#!/usr/bin/env python3
"""Run a quick pre-eval data test suite for USR LRS2 manifests.

Suite:
1) Manifest structure checks
2) HF asset existence checks (existing script)
3) Transcript->token-id consistency against Hub .txt
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run pre-eval data tests.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/hoangbng/AVATAR/AVATAR/data/val_manifest.csv"),
    )
    ap.add_argument(
        "--repo-prefix",
        type=str,
        default="HBaoAL/LRS2",
    )
    ap.add_argument(
        "--units",
        type=Path,
        default=Path("/home/hoangbng/AVATAR/AVATAR/models/usr/utils/labels/unigram1000_units.txt"),
    )
    ap.add_argument(
        "--sample-count",
        type=int,
        default=200,
        help="Sampling size for tokenization and asset checks (0 = all where supported).",
    )
    return ap.parse_args()


def run_step(name: str, cmd: List[str]) -> int:
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    p = subprocess.run(cmd, check=False)
    print(f"[{name}] exit={p.returncode}")
    return p.returncode


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    manifest = str(args.manifest.resolve())
    units = str(args.units.resolve())

    steps = [
        (
            "manifest-structure",
            [
                sys.executable,
                str(root / "test_manifest_structure.py"),
                "--manifest",
                manifest,
                "--allow-ext",
                ".avi",
            ],
        ),
        (
            "hf-assets",
            [
                sys.executable,
                str(root / "check_val_manifest_hf_assets.py"),
                "--manifest",
                manifest,
                "--repo-prefix",
                args.repo_prefix,
                "--sample-count",
                str(args.sample_count),
                "--required-exts",
                ".avi,.wav,.txt",
            ],
        ),
        (
            "tokenization-vs-hub-transcript",
            [
                sys.executable,
                str(root / "test_manifest_tokenization_against_hub.py"),
                "--manifest",
                manifest,
                "--units",
                units,
                "--repo-prefix",
                args.repo_prefix,
                "--sample-count",
                str(args.sample_count),
            ],
        ),
    ]

    failures = 0
    for name, cmd in steps:
        failures += 1 if run_step(name, cmd) != 0 else 0

    print("\n=== SUITE RESULT ===")
    if failures:
        print(f"FAIL: {failures}/{len(steps)} checks failed.")
        return 1
    print("PASS: all checks succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

