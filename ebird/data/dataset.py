from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class RGBEGazeDataset(Dataset):
    """Manifest-backed paired grayscale-frame/accumulated-event dataset."""

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        *,
        resolution: int,
        include_condition: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.resolution = int(resolution)
        self.include_condition = include_condition
        with self.manifest_path.open(newline="", encoding="utf-8") as stream:
            self.rows = list(csv.DictReader(stream))
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        if self.resolution <= 0:
            raise ValueError("resolution debe ser positiva")

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, relative_path: str) -> torch.Tensor:
        path = self.dataset_root / relative_path
        with Image.open(path) as image:
            image = image.convert("L")
            image = image.resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS
            )
            tensor = TF.to_tensor(image)
        return tensor.mul(2.0).sub(1.0)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        sample: dict[str, object] = {
            "target": self._load(row["target_path"]),
            "filename": row["filename"],
            "user": row["user"],
            "experiment": row["experiment"],
        }
        if self.include_condition:
            sample["condition"] = self._load(row["event_path"])
        return sample
