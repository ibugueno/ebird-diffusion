#!/usr/bin/env python3
"""Smoke test del entorno Ebird sin utilizar el dataset real."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from Datasets import SingleImageDataset  # noqa: E402
from PairedDataSet_ajustable import PairedImageDataset  # noqa: E402
from Scheduler import LinearNoiseScheduler  # noqa: E402
from UnetClass2 import CombinedUnet, Unet  # noqa: E402


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero torch.cuda.is_available() es False")
    return torch.device(requested)


def create_synthetic_images(root: Path) -> tuple[Path, Path]:
    image_dir = root / "MNIST" / "Train" / "0"
    event_dir = root / "N-MNIST" / "33ms" / "Train" / "0"
    image_dir.mkdir(parents=True)
    event_dir.mkdir(parents=True)

    for index in range(2):
        image = np.zeros((28, 28), dtype=np.uint8)
        image[4 + index : 20 + index, 8:20] = 180 + index * 40
        event = np.where(image > 0, 255, 0).astype(np.uint8)
        filename = f"synthetic_{index}.png"
        Image.fromarray(image).save(image_dir / filename)
        Image.fromarray(event).save(event_dir / filename)

    return image_dir, event_dir


def check_datasets() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="ebird-smoke-") as temp_dir:
        image_dir, event_dir = create_synthetic_images(Path(temp_dir))

        single = SingleImageDataset([str(image_dir)])
        paired = PairedImageDataset(
            [str(image_dir)],
            [str(event_dir)],
            im_size=(28, 28),
            subset_ratio=1.0,
        )

        image = single[0]
        paired_image, paired_event = paired[0]
        expected_shape = (1, 28, 28)
        assert tuple(image.shape) == expected_shape
        assert tuple(paired_image.shape) == expected_shape
        assert tuple(paired_event.shape) == expected_shape
        assert image.min() >= -1 and image.max() <= 1

        return {
            "single_samples": len(single),
            "paired_samples": len(paired),
            "sample_shape": list(image.shape),
        }


def check_models(config: dict, device: torch.device) -> dict[str, object]:
    model_config = config["model_params"]
    diffusion_config = config["diffusion_params"]

    scheduler = LinearNoiseScheduler(**diffusion_config)
    clean = torch.randn(1, 1, 28, 28, device=device)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([10], device=device)
    noisy = scheduler.add_noise(clean, noise, timesteps)
    assert tuple(noisy.shape) == tuple(clean.shape)
    assert torch.isfinite(noisy).all()

    unet = Unet(model_config).to(device)
    prediction = unet(noisy, timesteps)
    assert tuple(prediction.shape) == tuple(clean.shape)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert any(parameter.grad is not None for parameter in unet.parameters())

    del prediction, unet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    combined = CombinedUnet(model_config, model_config).to(device)
    condition = torch.randn_like(clean)
    conditional_prediction = combined(noisy, condition, timesteps)
    assert tuple(conditional_prediction.shape) == tuple(clean.shape)
    assert torch.isfinite(conditional_prediction).all()
    conditional_prediction.square().mean().backward()

    frozen_gradients = [
        parameter.grad for parameter in combined.unet.parameters()
        if parameter.grad is not None
    ]
    trainable_gradients = [
        parameter.grad for parameter in combined.partial_unet.parameters()
        if parameter.grad is not None
    ]
    assert not frozen_gradients, "La U-Net base debería permanecer congelada"
    assert trainable_gradients, "La rama condicional debería recibir gradientes"

    return {
        "device": str(device),
        "output_shape": list(conditional_prediction.shape),
        "unet_parameters": sum(p.numel() for p in combined.unet.parameters()),
        "conditional_trainable_parameters": sum(
            p.numel() for p in combined.partial_unet.parameters() if p.requires_grad
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/Rislab_Event_influence_volume/smoke_test"),
        help="Directorio donde se escribirá smoke_test_report.json",
    )
    args = parser.parse_args()

    with (SRC / "default.yaml").open() as config_file:
        config = yaml.safe_load(config_file)

    device = select_device(args.device)
    report = {
        "status": "ok",
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "datasets": check_datasets(),
        "models": check_models(config, device),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "smoke_test_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(f"\nSMOKE TEST OK: {report_path}")


if __name__ == "__main__":
    main()
