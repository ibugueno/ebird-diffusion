#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ebird.data import build_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye manifests pareados RGBE-Gaze")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--users",
        nargs="+",
        help="Usuarios a incluir, por ejemplo: --users user_1 user_2",
    )
    args = parser.parse_args()
    counts = build_manifests(
        args.dataset_root,
        args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        include_users=args.users,
    )
    print(json.dumps({"status": "ok", "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
