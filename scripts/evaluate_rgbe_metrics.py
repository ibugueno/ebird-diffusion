#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def compute_metrics(target: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    try:
        from skimage.metrics import structural_similarity
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "scikit-image is required for SSIM; run this script in the EVDiff environment"
        ) from error
    if target.shape != generated.shape:
        raise ValueError(f"Shape mismatch: target={target.shape}, generated={generated.shape}")
    mse = float(np.mean((target - generated) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))
    ssim = float(structural_similarity(target, generated, data_range=1.0))
    return {"mse": mse, "ssim": ssim, "psnr": psnr}


def summarize(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return {
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "non_finite": len(values) - int(finite.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MSE, SSIM, and PSNR for generated RGBE-Gaze samples"
    )
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.samples_dir / "metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for generated_path in sorted(args.samples_dir.rglob("generated_*")):
        suffix = generated_path.name.removeprefix("generated_")
        target_path = generated_path.with_name(f"target_{suffix}")
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing target for {generated_path}: {target_path}")
        metrics = compute_metrics(
            load_grayscale(target_path), load_grayscale(generated_path)
        )
        rows.append(
            {
                "relative_path": str(generated_path.relative_to(args.samples_dir)),
                **metrics,
            }
        )
    if not rows:
        raise ValueError(f"No generated_* images found under {args.samples_dir}")

    csv_path = output_dir / "per_image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("relative_path", "mse", "ssim", "psnr")
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": len(rows),
        "mse": summarize([float(row["mse"]) for row in rows]),
        "ssim": summarize([float(row["ssim"]) for row in rows]),
        "psnr": summarize([float(row["psnr"]) for row in rows]),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Per-image metrics: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
