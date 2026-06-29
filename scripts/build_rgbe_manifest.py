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
    parser = argparse.ArgumentParser(description="Build paired RGBE-Gaze manifests")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--users",
        nargs="+",
        help="Users to include, for example: --users user_1 user_2",
    )
    parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="Fail when a target frame or event representation has no pair",
    )
    parser.add_argument(
        "--val-experiments",
        nargs="*",
        help="Experiments assigned to validation, for example: exp5",
    )
    parser.add_argument(
        "--test-experiments",
        nargs="*",
        help="Experiments assigned to testing, for example: exp6",
    )
    args = parser.parse_args()
    counts = build_manifests(
        args.dataset_root,
        args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        include_users=args.users,
        strict_pairs=args.strict_pairs,
        val_experiments=args.val_experiments,
        test_experiments=args.test_experiments,
    )
    print(json.dumps({"status": "ok", "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
