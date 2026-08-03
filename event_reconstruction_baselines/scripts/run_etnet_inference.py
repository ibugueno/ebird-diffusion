#!/usr/bin/env python3
"""Run the unmodified ET-Net inference entry point with a device shim.

The pinned ET-Net source constructs ``torch.device('cuda: 0')`` (with a
space), which newer PyTorch versions reject. This adapter fixes only that
runtime string and then executes the official ``inference.py`` unchanged.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


_torch_device = torch.device


def _compatible_device(spec):
    if isinstance(spec, str):
        spec = spec.replace("cuda: ", "cuda:")
    return _torch_device(spec)


def main() -> None:
    torch.device = _compatible_device
    source = Path(__file__).resolve().parents[1] / "third_party" / "et-net" / "inference.py"
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
