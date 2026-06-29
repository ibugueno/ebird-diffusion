#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ebird.config import load_config, require_sections
from ebird.data import RGBEGazeDataset
from ebird.diffusion import LinearNoiseScheduler
from ebird.models.conditional import conditional_from_config
from ebird.models.unet import unet_from_config


def _load_model(config: dict, device: torch.device):
    base_checkpoint = torch.load(config["training"]["base_checkpoint"], map_location="cpu")
    base = unet_from_config(config["model"])
    base.load_state_dict(base_checkpoint["model_state_dict"])
    model = conditional_from_config(base, config["model"])
    conditional_checkpoint = torch.load(
        config["training"]["conditional_checkpoint"], map_location="cpu"
    )
    model.control.load_state_dict(conditional_checkpoint["control_state_dict"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generación condicionada RGBE-Gaze")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--device",
        type=int,
        help="Índice CUDA que se utilizará; por defecto usa cuda:0",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(config, "data", "model", "diffusion", "training", "sampling")
    sampling = config["sampling"]
    split = sampling.get("split", "test")
    dataset = RGBEGazeDataset(
        Path(config["data"]["manifest_dir"]) / f"{split}.csv",
        config["data"]["dataset_root"],
        resolution=config["model"]["image_size"],
        include_condition=True,
    )
    loader = DataLoader(dataset, batch_size=int(sampling.get("batch_size", 1)))
    if torch.cuda.is_available():
        device_index = args.device if args.device is not None else 0
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(
                f"GPU cuda:{device_index} no disponible; "
                f"torch detecta {torch.cuda.device_count()} GPU"
            )
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)
    elif args.device is not None:
        raise RuntimeError("Se indicó --device, pero CUDA no está disponible")
    else:
        device = torch.device("cpu")
    model = _load_model(config, device)
    scheduler = LinearNoiseScheduler(**config["diffusion"])
    output_root = Path(sampling["output_dir"])
    limit = args.limit if args.limit is not None else int(sampling.get("limit", len(dataset)))
    generated = 0

    with torch.inference_mode():
        for batch in loader:
            condition = batch["condition"].to(device)
            current = torch.randn_like(condition)
            for timestep in tqdm(
                reversed(range(scheduler.num_timesteps)),
                total=scheduler.num_timesteps,
                desc=f"batch {generated}",
                leave=False,
            ):
                timesteps = torch.full(
                    (condition.shape[0],), timestep, device=device, dtype=torch.long
                )
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    predicted_noise = model(current, condition, timesteps)
                current, _ = scheduler.sample_previous(current, predicted_noise, timesteps)

            for index in range(current.shape[0]):
                if generated >= limit:
                    break
                sample_dir = output_root / batch["user"][index] / batch["experiment"][index]
                filename = batch["filename"][index]
                sample_dir.mkdir(parents=True, exist_ok=True)
                save_image((current[index].cpu() + 1) / 2, sample_dir / f"generated_{filename}")
                save_image((batch["target"][index] + 1) / 2, sample_dir / f"target_{filename}")
                save_image((batch["condition"][index] + 1) / 2, sample_dir / f"event_{filename}")
                generated += 1
            if generated >= limit:
                break
    print(f"Generadas {generated} muestras en {output_root}")


if __name__ == "__main__":
    main()
