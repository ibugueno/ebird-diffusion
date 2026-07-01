#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ebird.data import build_manifests


DATASET_ROOT = Path("/app/Rislab_Event_influence_volume/dataset/rgbe-gaze")
VOLUME_ROOT = Path("/app/Rislab_Event_influence_volume/rgbe-gaze")
CONFIG_ROOT = ROOT / "configs" / "rgbe_gaze" / "generalization_v2"
TRAINING_STEPS = ("generic-image", "generic-conditional", "specific-conditional")


def _users(first: int, last: int) -> list[str]:
    return [f"user_{index}" for index in range(first, last + 1)]


def _protocol(protocol: str) -> dict[str, object]:
    if protocol == "reduced":
        return {
            "generic_train_experiments": ["exp1"],
            "specific_train_experiments": ["exp2"],
            "train_stride": 5,
            "val_stride": 5,
        }
    return {
        "generic_train_experiments": ["exp1", "exp2", "exp3", "exp4"],
        "specific_train_experiments": ["exp1", "exp2", "exp3", "exp4"],
        "train_stride": 1,
        "val_stride": 1,
    }


def _paths(protocol: str) -> dict[str, Path]:
    return {
        "generic_manifest": VOLUME_ROOT
        / "manifests"
        / "generalization-v2"
        / protocol
        / "generic-1-50",
        "specific_manifest": VOLUME_ROOT
        / "manifests"
        / "generalization-v2"
        / protocol
        / "specific-51-66",
        "generic_config": CONFIG_ROOT / f"generic_1_50_{protocol}.yaml",
        "specific_config": CONFIG_ROOT / f"specific_51_66_{protocol}.yaml",
        "generic_run": VOLUME_ROOT
        / "runs"
        / "generalization-v2"
        / protocol
        / "generic-1-50",
        "specific_run": VOLUME_ROOT
        / "runs"
        / "generalization-v2"
        / protocol
        / "specific-51-66",
        "specific_validation_samples": VOLUME_ROOT
        / "samples"
        / "generalization-v2"
        / protocol
        / "specific-51-66"
        / "validation-checkpoints",
        "generic_validation_samples": VOLUME_ROOT
        / "samples"
        / "generalization-v2"
        / protocol
        / "generic-1-50"
        / "validation-checkpoints",
        "specific_test_samples": VOLUME_ROOT
        / "samples"
        / "generalization-v2"
        / protocol
        / "specific-51-66"
        / "test-best",
        "generic_test_samples": VOLUME_ROOT
        / "samples"
        / "generalization-v2"
        / protocol
        / "generic-1-50"
        / "test-best",
    }


def _manifest_summary(
    manifest_dir: Path,
    counts: dict[str, int],
    requested_users: list[str],
) -> dict[str, object]:
    split_users: dict[str, set[str]] = defaultdict(set)
    split_experiments: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "val", "test"):
        with (manifest_dir / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                split_users[split].add(row["user"])
                split_experiments[split].add(row["experiment"])
    included_users = set().union(*split_users.values())
    return {
        "counts": counts,
        "users": {key: sorted(value) for key, value in split_users.items()},
        "excluded_requested_users": sorted(set(requested_users) - included_users),
        "experiments": {
            key: sorted(value) for key, value in split_experiments.items()
        },
    }


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _manifest_compatibility(
    manifest_dir: Path,
    train_experiments: list[str],
    expected_spec: dict[str, object],
) -> tuple[bool, str]:
    spec_path = manifest_dir / "protocol_spec.json"
    if not spec_path.is_file():
        return False, f"{spec_path} is missing"
    current_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if current_spec != expected_spec:
        return False, f"{spec_path} does not match the selected protocol"
    expected = {
        "train": set(train_experiments),
        "val": {"exp5"},
        "test": {"exp6"},
    }
    for split, expected_experiments in expected.items():
        path = manifest_dir / f"{split}.csv"
        rows = _read_manifest_rows(path)
        if not rows:
            return False, f"{path} is missing or empty"
        experiments = {row["experiment"] for row in rows}
        if experiments != expected_experiments:
            return (
                False,
                f"{path} contains experiments {sorted(experiments)}; "
                f"expected {sorted(expected_experiments)}",
            )
    return True, "compatible"


def _existing_counts(manifest_dir: Path) -> dict[str, int]:
    split_counts = {
        split: len(_read_manifest_rows(manifest_dir / f"{split}.csv"))
        for split in ("train", "val", "test")
    }
    return {
        "paired": sum(split_counts.values()),
        "skipped_without_event": 0,
        "skipped_without_target": 0,
        **split_counts,
    }


def _prepare(protocol: str, dataset_root: Path, *, force: bool) -> None:
    settings = _protocol(protocol)
    paths = _paths(protocol)
    jobs = (
        (
            "generic",
            paths["generic_manifest"],
            _users(1, 50),
            settings["generic_train_experiments"],
        ),
        (
            "specific",
            paths["specific_manifest"],
            _users(51, 66),
            settings["specific_train_experiments"],
        ),
    )
    report: dict[str, object] = {"protocol": protocol, "dataset_root": str(dataset_root)}
    for name, output_dir, users, train_experiments in jobs:
        manifest_spec = {
            "protocol": protocol,
            "role": name,
            "users": users,
            "train_experiments": list(train_experiments),
            "val_experiments": ["exp5"],
            "test_experiments": ["exp6"],
            "train_stride": int(settings["train_stride"]),
            "val_stride": int(settings["val_stride"]),
            "test_stride": 1,
        }
        if (output_dir / "all.csv").is_file() and not force:
            compatible, reason = _manifest_compatibility(
                output_dir,
                list(train_experiments),
                manifest_spec,
            )
            if not compatible:
                raise FileExistsError(
                    f"Existing manifest is incompatible: {output_dir}. {reason}. "
                    "Pass --force-prepare to rebuild it intentionally."
                )
            counts = _existing_counts(output_dir)
            report[name] = _manifest_summary(output_dir, counts, users)
            print(f"Reusing compatible {name} manifests: {output_dir}")
            continue
        counts = build_manifests(
            dataset_root,
            output_dir,
            include_users=users,
            allow_missing_users=True,
            train_experiments=train_experiments,
            val_experiments=["exp5"],
            test_experiments=["exp6"],
            train_stride=int(settings["train_stride"]),
            val_stride=int(settings["val_stride"]),
            drop_users_without_train=True,
        )
        (output_dir / "protocol_spec.json").write_text(
            json.dumps(manifest_spec, indent=2) + "\n",
            encoding="utf-8",
        )
        report[name] = _manifest_summary(output_dir, counts, users)
        print(f"Prepared {name} manifests: {json.dumps(counts, sort_keys=True)}")
    report_path = (
        VOLUME_ROOT / "manifests" / "generalization-v2" / protocol / "summary.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Protocol summary: {report_path}")


def _display_command(command: list[str], env_updates: dict[str, str] | None = None) -> str:
    prefix = ""
    if env_updates:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_updates.items()) + " "
    return prefix + shlex.join(command)


def _run(
    command: list[str],
    *,
    execute: bool,
    env_updates: dict[str, str] | None = None,
) -> None:
    print(_display_command(command, env_updates))
    if not execute:
        return
    environment = os.environ.copy()
    if env_updates:
        environment.update(env_updates)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _compose_comparisons(samples_dir: Path, *, execute: bool) -> None:
    _run(
        [
            sys.executable,
            "scripts/compose_rgbe_triplets.py",
            "--input-dir",
            str(samples_dir),
            "--output-dir",
            str(samples_dir / "comparisons"),
            "--recursive",
            "--skip-incomplete",
        ],
        execute=execute,
    )


def _train_step(step: str, protocol: str, gpus: str, execute: bool) -> None:
    paths = _paths(protocol)
    config = paths["specific_config"] if step == "specific-conditional" else paths["generic_config"]
    stage = "image" if step == "generic-image" else "conditional"
    gpu_list = [value.strip() for value in gpus.split(",") if value.strip()]
    if not gpu_list:
        raise ValueError("At least one GPU must be provided")
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={len(gpu_list)}",
        "scripts/train_rgbe.py",
        "--stage",
        stage,
        "--config",
        str(config),
    ]
    _run(
        command,
        execute=execute,
        env_updates={"CUDA_VISIBLE_DEVICES": ",".join(gpu_list)},
    )


def _evaluate_snapshots(
    protocol: str,
    *,
    model: str,
    device: int,
    samples_per_user: int,
    limit: int,
    checkpoint_epochs: list[int],
    execute: bool,
) -> None:
    paths = _paths(protocol)
    if model not in ("generic", "specific"):
        raise ValueError(f"Invalid snapshot model: {model}")
    if limit == 0:
        raise ValueError("--validation-limit cannot be zero")
    invalid_epochs = [epoch for epoch in checkpoint_epochs if epoch < 0 or epoch > 40]
    if invalid_epochs:
        raise ValueError(f"Checkpoint epochs must be between 0 and 40: {invalid_epochs}")
    evaluations = []
    for epoch in dict.fromkeys(checkpoint_epochs):
        if epoch == 0 and model == "specific":
            evaluations.append(
                (
                    0,
                    paths["generic_run"]
                    / "conditional"
                    / "checkpoints"
                    / "best.pt",
                    "epoch_0000_generic",
                )
            )
        elif epoch > 0:
            evaluations.append(
                (
                    epoch,
                    paths[f"{model}_run"]
                    / "conditional"
                    / "checkpoints"
                    / f"epoch_{epoch:04d}.pt",
                    f"epoch_{epoch:04d}",
                )
            )
    if not evaluations:
        raise ValueError("No checkpoints were selected for evaluation")
    selection_name = f"limit-{limit}-per-user-{samples_per_user}"
    output_root = paths[f"{model}_validation_samples"] / selection_name
    aggregate_rows: list[dict[str, object]] = []
    for epoch, checkpoint, output_name in evaluations:
        output_dir = output_root / output_name
        sample_command = [
            sys.executable,
            "scripts/sample_rgbe.py",
            "--device",
            str(device),
            "--config",
            str(paths[f"{model}_config"]),
            "--conditional-checkpoint",
            str(checkpoint),
            "--split",
            "val",
            "--samples-per-user",
            str(samples_per_user),
            "--limit",
            str(limit),
            "--seed",
            "44",
            "--output-dir",
            str(output_dir),
        ]
        _run(sample_command, execute=execute)
        _run(
            [
                sys.executable,
                "scripts/evaluate_rgbe_metrics.py",
                "--samples-dir",
                str(output_dir),
            ],
            execute=execute,
        )
        _compose_comparisons(output_dir, execute=execute)
        if execute:
            summary_path = output_dir / "metrics" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            aggregate_rows.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "samples": summary["samples"],
                    "mse_mean": summary["mse"]["mean"],
                    "mse_std": summary["mse"]["std"],
                    "ssim_mean": summary["ssim"]["mean"],
                    "ssim_std": summary["ssim"]["std"],
                    "psnr_mean": summary["psnr"]["mean"],
                    "psnr_std": summary["psnr"]["std"],
                }
            )
    if execute:
        aggregate_path = output_root / "checkpoint_metrics.csv"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(aggregate_rows[0]))
            writer.writeheader()
            writer.writerows(aggregate_rows)
        print(f"Checkpoint metric summary: {aggregate_path}")


def _evaluate_test(
    protocol: str,
    *,
    model: str,
    device: int,
    limit: int,
    samples_per_user: int,
    execute: bool,
) -> None:
    paths = _paths(protocol)
    if model not in ("generic", "specific"):
        raise ValueError(f"Invalid test model: {model}")
    if limit == 0:
        raise ValueError("--test-limit cannot be zero")
    if samples_per_user < 0:
        raise ValueError("--test-samples-per-user cannot be negative")
    selection_name = (
        "all"
        if limit < 0 and samples_per_user == 0
        else f"limit-{limit}-per-user-{samples_per_user}"
    )
    config = paths[f"{model}_config"]
    output_dir = paths[f"{model}_test_samples"] / selection_name
    command = [
        sys.executable,
        "scripts/sample_rgbe.py",
        "--device",
        str(device),
        "--config",
        str(config),
        "--split",
        "test",
        "--limit",
        str(limit),
        "--seed",
        "44",
        "--output-dir",
        str(output_dir),
    ]
    if samples_per_user > 0:
        command.extend(["--samples-per-user", str(samples_per_user)])
    _run(command, execute=execute)
    _run(
        [
            sys.executable,
            "scripts/evaluate_rgbe_metrics.py",
            "--samples-dir",
            str(output_dir),
        ],
        execute=execute,
    )
    _compose_comparisons(output_dir, execute=execute)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and run the V2 cross-user generalization protocol"
    )
    parser.add_argument("--protocol", choices=("reduced", "full"), default="reduced")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=(
            "all",
            "prepare",
            "generic-image",
            "generic-conditional",
            "specific-conditional",
            "generic-evaluate",
            "specific-evaluate",
            "generic-test",
            "specific-test",
        ),
        default=["all"],
    )
    parser.add_argument("--gpus", default="1,2,4")
    parser.add_argument("--sampling-device", type=int, default=1)
    parser.add_argument("--samples-per-user", type=int, default=1)
    parser.add_argument("--validation-limit", type=int, default=100)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument(
        "--test-samples-per-user",
        type=int,
        default=7,
        help="Balanced test samples per user; use 0 with --test-limit -1 for all",
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="+",
        default=list(range(0, 41, 5)),
        help="Checkpoints to evaluate; epoch 0 is the generic zero-shot model",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Intentionally overwrite existing manifests for the selected protocol",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the pipeline; without this flag, only print the commands",
    )
    args = parser.parse_args()

    steps = list(args.steps)
    if "all" in steps:
        steps = ["prepare", *TRAINING_STEPS]
    if "prepare" in steps:
        if args.execute:
            _prepare(args.protocol, args.dataset_root, force=args.force_prepare)
        else:
            print(
                "DRY RUN: prepare manifests for the "
                f"{args.protocol} protocol from {args.dataset_root}"
            )
    for step in TRAINING_STEPS:
        if step in steps:
            _train_step(step, args.protocol, args.gpus, args.execute)
    if "generic-evaluate" in steps:
        _evaluate_snapshots(
            args.protocol,
            model="generic",
            device=args.sampling_device,
            samples_per_user=args.samples_per_user,
            limit=args.validation_limit,
            checkpoint_epochs=args.checkpoint_epochs,
            execute=args.execute,
        )
    if "specific-evaluate" in steps:
        _evaluate_snapshots(
            args.protocol,
            model="specific",
            device=args.sampling_device,
            samples_per_user=args.samples_per_user,
            limit=args.validation_limit,
            checkpoint_epochs=args.checkpoint_epochs,
            execute=args.execute,
        )
    if "specific-test" in steps:
        _evaluate_test(
            args.protocol,
            model="specific",
            device=args.sampling_device,
            limit=args.test_limit,
            samples_per_user=args.test_samples_per_user,
            execute=args.execute,
        )
    if "generic-test" in steps:
        _evaluate_test(
            args.protocol,
            model="generic",
            device=args.sampling_device,
            limit=args.test_limit,
            samples_per_user=args.test_samples_per_user,
            execute=args.execute,
        )
    if not args.execute:
        print("Dry run only. Add --execute to run these steps.")


if __name__ == "__main__":
    main()
