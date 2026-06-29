#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ebird.data import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate files from an RGBE-Gaze manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    result = validate_manifest(
        args.manifest,
        args.dataset_root,
        verify_images=not args.skip_image_check,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
