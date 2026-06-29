#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ebird.config import load_config, require_sections
from ebird.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="RGBE-Gaze DDPM training")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("image", "conditional"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--device",
        type=int,
        help="CUDA index for single-GPU execution, for example --device 1",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    require_sections(config, "data", "model", "diffusion", "training")
    train(
        args.stage,
        config,
        resume=not args.no_resume,
        device_index=args.device,
    )


if __name__ == "__main__":
    main()
