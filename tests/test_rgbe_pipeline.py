#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ebird.data import RGBEGazeDataset, build_manifests, validate_manifest
from ebird.diffusion import LinearNoiseScheduler
from ebird.models.conditional import conditional_from_config
from ebird.models.unet import unet_from_config
from ebird.training.distributed import DistributedContext
from ebird.training.trainer import _run_epoch
from scripts.evaluate_rgbe_metrics import compute_metrics


SKIMAGE_AVAILABLE = importlib.util.find_spec("skimage") is not None


def _create_dataset(
    root: Path,
    users: int = 4,
    samples: int = 2,
    experiments: int = 1,
) -> None:
    for user in range(1, users + 1):
        for experiment in range(1, experiments + 1):
            for branch in ("gray_frames", "event_accumulate_frames"):
                directory = root / branch / f"user_{user}" / f"exp{experiment}"
                directory.mkdir(parents=True)
                for index in range(samples):
                    pixels = np.zeros((40, 40), dtype=np.uint8)
                    pixels[8:32, 10 + index : 30 + index] = 100 + user * 20
                    if branch == "event_accumulate_frames":
                        pixels = np.where(pixels > 0, 255, 0).astype(np.uint8)
                    filename = (
                        f"user_{user}_exp{experiment}_frame_{index}_rgbts_{index}_"
                        f"evts_{index}-{index + 33}_bbox_0-0-40-40_"
                        "size_512_winms_33_cropkb_20.png"
                    )
                    Image.fromarray(pixels).save(directory / filename)


class RGBEGazePipelineTest(unittest.TestCase):
    def test_validation_noise_is_deterministic(self):
        class ZeroNoiseModel(torch.nn.Module):
            def forward(self, noisy, timesteps):
                return torch.zeros_like(noisy)

        loader = DataLoader(
            [{"target": torch.zeros(1, 8, 8)} for _ in range(4)],
            batch_size=2,
        )
        context = DistributedContext(
            rank=1,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )
        arguments = {
            "model": ZeroNoiseModel(),
            "loader": loader,
            "optimizer": None,
            "scaler": torch.cuda.amp.GradScaler(enabled=False),
            "scheduler": LinearNoiseScheduler(10, 0.0001, 0.02),
            "stage": "image",
            "config": {
                "training": {
                    "gradient_accumulation_steps": 1,
                    "mixed_precision": False,
                }
            },
            "context": context,
            "random_seed": 10044,
        }
        first = _run_epoch(**arguments)
        second = _run_epoch(**arguments)
        self.assertEqual(first, second)

    def test_experiment_split_for_one_user(self):
        with tempfile.TemporaryDirectory(prefix="rgbe-split-") as temporary:
            root = Path(temporary)
            dataset_root = root / "rgbe-gaze"
            manifest_dir = root / "manifests"
            _create_dataset(dataset_root, users=1, samples=2, experiments=6)
            counts = build_manifests(
                dataset_root,
                manifest_dir,
                include_users=["user_1"],
                val_experiments=["5"],
                test_experiments=["exp6"],
            )
            self.assertEqual(counts, {
                "paired": 12,
                "skipped_without_event": 0,
                "skipped_without_target": 0,
                "train": 8,
                "val": 2,
                "test": 2,
            })
            expected = {
                "train": {"exp1", "exp2", "exp3", "exp4"},
                "val": {"exp5"},
                "test": {"exp6"},
            }
            for split, experiments in expected.items():
                with (manifest_dir / f"{split}.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual({row["experiment"] for row in rows}, experiments)
                self.assertEqual({row["user"] for row in rows}, {"user_1"})

    @unittest.skipUnless(SKIMAGE_AVAILABLE, "scikit-image is not installed")
    def test_reconstruction_metrics(self):
        target = np.zeros((16, 16), dtype=np.float32)
        identical = compute_metrics(target, target.copy())
        self.assertEqual(identical["mse"], 0.0)
        self.assertEqual(identical["ssim"], 1.0)
        self.assertTrue(np.isinf(identical["psnr"]))

        generated = target.copy()
        generated[4:12, 4:12] = 1.0
        changed = compute_metrics(target, generated)
        self.assertGreater(changed["mse"], 0.0)
        self.assertLess(changed["ssim"], 1.0)
        self.assertTrue(np.isfinite(changed["psnr"]))

    def test_user_filter(self):
        with tempfile.TemporaryDirectory(prefix="rgbe-filter-") as temporary:
            root = Path(temporary)
            dataset_root = root / "rgbe-gaze"
            manifest_dir = root / "manifests"
            _create_dataset(dataset_root, users=3, samples=2)
            unmatched = next(
                (dataset_root / "event_accumulate_frames" / "user_2").rglob("*.png")
            )
            unmatched.unlink()
            counts = build_manifests(
                dataset_root,
                manifest_dir,
                include_users=["1"],
            )
            self.assertEqual(counts["paired"], 2)
            self.assertEqual(counts["train"], 2)
            self.assertEqual(counts["skipped_without_event"], 0)
            self.assertEqual(counts["skipped_without_target"], 0)
            with (manifest_dir / "train.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["user"] for row in rows}, {"user_1"})

            all_counts = build_manifests(dataset_root, root / "all-manifests")
            self.assertEqual(all_counts["paired"], 5)
            self.assertEqual(all_counts["skipped_without_event"], 1)

    def test_manifest_dataset_and_models(self):
        with tempfile.TemporaryDirectory(prefix="rgbe-test-") as temporary:
            root = Path(temporary)
            dataset_root = root / "rgbe-gaze"
            manifest_dir = root / "manifests"
            _create_dataset(dataset_root)
            counts = build_manifests(
                dataset_root,
                manifest_dir,
                val_ratio=0.25,
                test_ratio=0.25,
                seed=7,
            )
            self.assertEqual(counts["paired"], 8)
            self.assertEqual(counts["train"] + counts["val"] + counts["test"], 8)
            self.assertGreater(counts["train"], 0)
            self.assertGreater(counts["val"], 0)
            self.assertGreater(counts["test"], 0)

            user_sets = {}
            for split in ("train", "val", "test"):
                with (manifest_dir / f"{split}.csv").open(newline="") as stream:
                    user_sets[split] = {row["user"] for row in csv.DictReader(stream)}
            self.assertTrue(user_sets["train"].isdisjoint(user_sets["val"]))
            self.assertTrue(user_sets["train"].isdisjoint(user_sets["test"]))
            self.assertTrue(user_sets["val"].isdisjoint(user_sets["test"]))

            validation = validate_manifest(manifest_dir / "all.csv", dataset_root)
            self.assertTrue(validation["valid"], validation["errors"])

            dataset = RGBEGazeDataset(
                manifest_dir / "train.csv",
                dataset_root,
                resolution=32,
                include_condition=True,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["target"].shape), (1, 32, 32))
            self.assertEqual(tuple(sample["condition"].shape), (1, 32, 32))

            model_config = {
                "image_size": 32,
                "in_channels": 1,
                "channels": [16, 32, 64],
                "attention_resolutions": [8],
                "time_embedding_dim": 64,
                "num_heads": 4,
                "dropout": 0.0,
                "gradient_checkpointing": False,
            }
            scheduler = LinearNoiseScheduler(10, 0.0001, 0.02)
            target = sample["target"].unsqueeze(0)
            condition = sample["condition"].unsqueeze(0)
            noise = torch.randn_like(target)
            timesteps = torch.tensor([3])
            noisy = scheduler.add_noise(target, noise, timesteps)

            base = unet_from_config(model_config)
            prediction = base(noisy, timesteps)
            self.assertEqual(tuple(prediction.shape), tuple(target.shape))
            prediction.square().mean().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in base.parameters()))

            base.zero_grad(set_to_none=True)
            conditional = conditional_from_config(base, model_config)
            prediction = conditional(noisy, condition, timesteps)
            self.assertEqual(tuple(prediction.shape), tuple(target.shape))
            prediction.square().mean().backward()
            self.assertFalse(any(parameter.grad is not None for parameter in conditional.base.parameters()))
            self.assertTrue(any(parameter.grad is not None for parameter in conditional.control.parameters()))


if __name__ == "__main__":
    unittest.main()
