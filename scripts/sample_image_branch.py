#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ebird.config import load_config, require_sections
from ebird.data import RGBEGazeDataset
from ebird.diffusion import LinearNoiseScheduler
from ebird.models.unet import unet_from_config


def _device(index: int | None) -> torch.device:
    if torch.cuda.is_available():
        selected = index if index is not None else 0
        if selected < 0 or selected >= torch.cuda.device_count():
            raise ValueError(
                f"GPU cuda:{selected} is unavailable; "
                f"PyTorch detects {torch.cuda.device_count()} GPUs"
            )
        device = torch.device("cuda", selected)
        torch.cuda.set_device(device)
        return device
    if index is not None:
        raise RuntimeError("--device was provided, but CUDA is unavailable")
    return torch.device("cpu")


def _load_model(
    config: dict,
    device: torch.device,
    checkpoint_path: Path,
) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Image-branch checkpoint does not exist: {checkpoint_path}. "
            "Run training with --stage image first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = unet_from_config(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate unconditional examples from the trained image branch"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", type=int, help="CUDA index; defaults to cuda:0")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Image checkpoint; defaults to training.base_checkpoint",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("--num-samples and --batch-size must be positive")

    config = load_config(args.config)
    require_sections(config, "data", "model", "diffusion", "training")
    device = _device(args.device)
    checkpoint_path = args.checkpoint or Path(
        config["training"]["base_checkpoint"]
    )
    model = _load_model(config, device, checkpoint_path)
    scheduler = LinearNoiseScheduler(**config["diffusion"])
    image_size = int(config["model"]["image_size"])
    channels = int(config["model"].get("in_channels", 1))
    output_dir = args.output_dir or (
        Path(config["training"]["output_dir"])
        / "image"
        / "samples"
        / f"seed-{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generated: list[torch.Tensor] = []
    with torch.inference_mode():
        while len(generated) < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - len(generated))
            current = torch.randn(
                current_batch,
                channels,
                image_size,
                image_size,
                device=device,
            )
            for timestep in tqdm(
                reversed(range(scheduler.num_timesteps)),
                total=scheduler.num_timesteps,
                desc=f"samples {len(generated) + 1}-{len(generated) + current_batch}",
                leave=False,
            ):
                timesteps = torch.full(
                    (current_batch,), timestep, device=device, dtype=torch.long
                )
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    predicted_noise = model(current, timesteps)
                current, _ = scheduler.sample_previous(
                    current, predicted_noise, timesteps
                )
            generated.extend(((current.cpu() + 1.0) / 2.0).clamp(0.0, 1.0))

    samples = torch.stack(generated)
    reference_dataset = RGBEGazeDataset(
        Path(config["data"]["manifest_dir"]) / "train.csv",
        config["data"]["dataset_root"],
        resolution=image_size,
        include_condition=False,
    )
    reference_count = min(len(samples), len(reference_dataset))
    reference_indices = random.Random(args.seed).sample(
        range(len(reference_dataset)), reference_count
    )
    references = torch.stack(
        [
            ((reference_dataset[index]["target"] + 1.0) / 2.0).clamp(0.0, 1.0)
            for index in reference_indices
        ]
    )
    grid_columns = math.ceil(math.sqrt(len(samples)))
    for index, sample in enumerate(samples):
        save_image(sample, output_dir / f"sample_{index:04d}.png")
    grid_path = output_dir / "grid.png"
    reference_grid_path = output_dir / "reference_grid.png"
    save_image(samples, grid_path, nrow=grid_columns)
    save_image(references, reference_grid_path, nrow=grid_columns)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "config": str(args.config),
        "device": str(device),
        "image_size": image_size,
        "num_samples": len(samples),
        "reference_indices": reference_indices,
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(samples)} image-branch samples in {output_dir}")
    print(f"Sample grid: {grid_path}")
    print(f"Unpaired training-reference grid: {reference_grid_path}")


if __name__ == "__main__":
    main()
