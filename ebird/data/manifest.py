from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MANIFEST_FIELDS = (
    "target_path",
    "event_path",
    "user",
    "experiment",
    "filename",
    "split",
)
USER_PATTERN = re.compile(r"^user_(\d+)$", re.IGNORECASE)
EXPERIMENT_PATTERN = re.compile(r"^exp(?:eriment)?_?(\d+)$", re.IGNORECASE)


def _image_map(root: Path) -> dict[Path, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    images: dict[Path, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            relative = path.relative_to(root)
            if relative in images:
                raise ValueError(f"Duplicate relative path: {relative}")
            images[relative] = path
    return images


def _parse_identity(relative_path: Path) -> tuple[str, str]:
    user = "unknown"
    experiment = "unknown"
    for part in relative_path.parts[:-1]:
        user_match = USER_PATTERN.match(part)
        experiment_match = EXPERIMENT_PATTERN.match(part)
        if user_match:
            user = f"user_{int(user_match.group(1))}"
        if experiment_match:
            experiment = f"exp{int(experiment_match.group(1))}"
    if user == "unknown":
        raise ValueError(
            f"Could not infer a user from path: {relative_path}"
        )
    return user, experiment


def _numeric_user_key(user: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", user)
    return (int(match.group(1)) if match else 10**9, user)


def _normalize_user(user: str) -> str:
    value = str(user).strip().lower()
    match = re.fullmatch(r"(?:user_?)?(\d+)", value)
    return f"user_{int(match.group(1))}" if match else value


def _normalize_experiment(experiment: str) -> str:
    value = str(experiment).strip().lower()
    match = re.fullmatch(r"(?:exp(?:eriment)?_?)?(\d+)", value)
    return f"exp{int(match.group(1))}" if match else value


def split_users(
    users: Iterable[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    unique_users = sorted(set(users), key=_numeric_user_key)
    if not unique_users:
        raise ValueError("No users were found")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must add up to less than 1")

    random.Random(seed).shuffle(unique_users)
    count = len(unique_users)
    if count == 1:
        return {unique_users[0]: "train"}

    test_count = max(1, round(count * test_ratio))
    val_count = max(1, round(count * val_ratio)) if count >= 3 and val_ratio > 0 else 0
    while count - test_count - val_count < 1:
        if val_count:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    assignment: dict[str, str] = {}
    for user in unique_users[:test_count]:
        assignment[user] = "test"
    for user in unique_users[test_count : test_count + val_count]:
        assignment[user] = "val"
    for user in unique_users[test_count + val_count :]:
        assignment[user] = "train"
    return assignment


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    target_dir: str = "gray_frames",
    event_dir: str = "event_accumulate_frames",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 44,
    include_users: Iterable[str] | None = None,
    allow_missing_users: bool = False,
    strict_pairs: bool = False,
    train_experiments: Iterable[str] | None = None,
    val_experiments: Iterable[str] | None = None,
    test_experiments: Iterable[str] | None = None,
    train_stride: int = 1,
    val_stride: int = 1,
    test_stride: int = 1,
    drop_users_without_train: bool = False,
) -> dict[str, int]:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    targets = _image_map(dataset_root / target_dir)
    events = _image_map(dataset_root / event_dir)

    missing_users: list[str] = []
    selected_users = (
        {_normalize_user(user) for user in include_users} if include_users else None
    )
    if selected_users:
        available_users = {
            _parse_identity(relative)[0] for relative in set(targets) | set(events)
        }
        missing_users = sorted(selected_users - available_users, key=_numeric_user_key)
        if missing_users and not allow_missing_users:
            raise ValueError(
                "Requested users were not found: " + ", ".join(missing_users)
            )
        targets = {
            relative: path
            for relative, path in targets.items()
            if _parse_identity(relative)[0] in selected_users
        }
        events = {
            relative: path
            for relative, path in events.items()
            if _parse_identity(relative)[0] in selected_users
        }

    missing_events = sorted(set(targets) - set(events))
    missing_targets = sorted(set(events) - set(targets))
    if strict_pairs and (missing_events or missing_targets):
        examples = [str(path) for path in (missing_events + missing_targets)[:10]]
        raise ValueError(
            "The dataset is not fully paired. "
            f"Missing event: {len(missing_events)}; missing target: {len(missing_targets)}; "
            f"examples: {examples}"
        )

    paired_paths = sorted(set(targets) & set(events))
    if not paired_paths:
        raise ValueError("No valid target/event pairs were found")

    rows: list[dict[str, str]] = []
    for relative in paired_paths:
        user, experiment = _parse_identity(relative)
        rows.append(
            {
                "target_path": str(targets[relative].relative_to(dataset_root)),
                "event_path": str(events[relative].relative_to(dataset_root)),
                "user": user,
                "experiment": experiment,
                "filename": relative.name,
                "split": "",
            }
        )

    training_experiments = {
        _normalize_experiment(value) for value in (train_experiments or [])
    }
    validation_experiments = {
        _normalize_experiment(value) for value in (val_experiments or [])
    }
    testing_experiments = {
        _normalize_experiment(value) for value in (test_experiments or [])
    }
    experiment_groups = {
        "train": training_experiments,
        "val": validation_experiments,
        "test": testing_experiments,
    }
    overlap = set()
    names = tuple(experiment_groups)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap.update(experiment_groups[first] & experiment_groups[second])
    if overlap:
        raise ValueError(
            "Experiment splits overlap: " + ", ".join(sorted(overlap))
        )

    explicit_experiment_split = any(experiment_groups.values())
    if explicit_experiment_split:
        available_experiments = {row["experiment"] for row in rows}
        requested = set().union(*experiment_groups.values())
        missing_experiments = sorted(requested - available_experiments)
        if missing_experiments:
            raise ValueError(
                "Requested experiments were not found: "
                + ", ".join(missing_experiments)
            )
        for row in rows:
            if row["experiment"] in testing_experiments:
                row["split"] = "test"
            elif row["experiment"] in validation_experiments:
                row["split"] = "val"
            elif training_experiments:
                row["split"] = "train" if row["experiment"] in training_experiments else "exclude"
            else:
                row["split"] = "train"
        rows = [row for row in rows if row["split"] != "exclude"]

        users_without_train: list[str] = []
        if drop_users_without_train:
            selected_row_users = {row["user"] for row in rows}
            users_with_train = {
                row["user"] for row in rows if row["split"] == "train"
            }
            users_without_train = sorted(
                selected_row_users - users_with_train,
                key=_numeric_user_key,
            )
            rows = [row for row in rows if row["user"] in users_with_train]
        if not any(row["split"] == "train" for row in rows):
            raise ValueError("The experiment split produced an empty training set")
    else:
        assignments = split_users(
            (row["user"] for row in rows),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        for row in rows:
            row["split"] = assignments[row["user"]]
        users_without_train = []

    strides = {
        "train": int(train_stride),
        "val": int(val_stride),
        "test": int(test_stride),
    }
    invalid_strides = {name: value for name, value in strides.items() if value < 1}
    if invalid_strides:
        values = ", ".join(f"{name}={value}" for name, value in invalid_strides.items())
        raise ValueError(f"Manifest strides must be at least 1: {values}")

    group_indices: Counter[tuple[str, str, str]] = Counter()
    sampled_rows: list[dict[str, str]] = []
    for row in rows:
        key = (row["split"], row["user"], row["experiment"])
        index = group_indices[key]
        group_indices[key] += 1
        if index % strides[row["split"]] == 0:
            sampled_rows.append(row)
    rows = sampled_rows

    _write_manifest(output_dir / "all.csv", rows)
    counts = Counter(row["split"] for row in rows)
    for split in ("train", "val", "test"):
        _write_manifest(
            output_dir / f"{split}.csv",
            [row for row in rows if row["split"] == split],
        )
    result = {
        "paired": len(rows),
        "skipped_without_event": len(missing_events),
        "skipped_without_target": len(missing_targets),
        **{split: counts.get(split, 0) for split in ("train", "val", "test")},
    }
    if selected_users is not None and (
        allow_missing_users
        or drop_users_without_train
        or any(value > 1 for value in strides.values())
    ):
        included_users = {row["user"] for row in rows}
        result.update(
            {
                "requested_users": len(selected_users),
                "included_users": len(included_users),
                "missing_users": len(missing_users),
                "users_without_train": len(users_without_train),
            }
        )
    return result


def validate_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    verify_images: bool = True,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root)
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    errors: list[str] = []
    users: set[str] = set()
    for row_index, row in enumerate(rows, start=2):
        users.add(row.get("user", "unknown"))
        for field in ("target_path", "event_path"):
            path = dataset_root / row[field]
            if not path.is_file():
                errors.append(f"line {row_index}: path does not exist: {path}")
                continue
            if verify_images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:  # Pillow may raise several exception types.
                    errors.append(f"line {row_index}: invalid image {path}: {exc}")
        if len(errors) >= 50:
            break
    return {
        "rows": len(rows),
        "users": len(users),
        "valid": not errors,
        "errors": errors,
    }
