#!/usr/bin/env python3
"""Check HF assets for stems listed in a USR manifest.

For sampled rows from a manifest (default: val manifest), verify that each stem
has all required files in the same HF repo folder:
  <stem>/<stem>.mp4, .wav, .npz, .avi
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate required files for sampled manifest stems on HF.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/hoangbng/AVATAR/AVATAR/data/val_manifest.csv"),
        help="Manifest CSV to sample from.",
    )
    ap.add_argument(
        "--repo-prefix",
        type=str,
        default="HBaoAL/LRS2",
        help="HF shard repo prefix, e.g. HBaoAL/LRS2.",
    )
    ap.add_argument(
        "--sample-count",
        type=int,
        default=50,
        help="How many manifest rows to sample (0 = all rows).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    ap.add_argument(
        "--required-exts",
        type=str,
        default=".mp4,.wav,.npz,.avi",
        help="Comma-separated required extensions.",
    )
    return ap.parse_args()


def read_manifest_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",", 3)
            if len(parts) < 2:
                continue
            tag = parts[0].strip()
            rel = parts[1].strip().replace("\\", "/")
            rows.append((tag, rel))
    return rows


def repo_from_tag(tag: str, repo_prefix: str) -> str:
    m = re.match(r"^lrs2_(\d{2})$", tag)
    if m:
        return f"{repo_prefix}_{m.group(1)}"
    if tag == "lrs2":
        return repo_prefix
    raise ValueError(f"Unsupported tag for LRS2 shard routing: {tag}")


def main() -> int:
    args = parse_args()
    rows = read_manifest_rows(args.manifest.resolve())
    if not rows:
        print(f"No rows found in {args.manifest}")
        return 1

    rng = random.Random(args.seed)
    if args.sample_count and args.sample_count > 0 and len(rows) > args.sample_count:
        rows = rng.sample(rows, args.sample_count)

    required_exts = [e.strip() for e in args.required_exts.split(",") if e.strip()]
    api = HfApi()

    # Group sampled rows by repo for one list_repo_files call per repo.
    rows_by_repo: dict[str, list[tuple[str, str]]] = {}
    for tag, rel in rows:
        repo = repo_from_tag(tag, args.repo_prefix)
        rows_by_repo.setdefault(repo, []).append((tag, rel))

    failures: list[str] = []
    checked = 0
    for repo_id, repo_rows in sorted(rows_by_repo.items()):
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        for _tag, rel in repo_rows:
            stem = Path(rel).stem
            folder = Path(rel).parent.as_posix()
            for ext in required_exts:
                target = f"{folder}/{stem}{ext}" if folder != "." else f"{stem}{ext}"
                if target not in files:
                    failures.append(f"{repo_id}: missing {target}")
            checked += 1

    print(f"Checked sampled stems: {checked}")
    print(f"Required extensions: {', '.join(required_exts)}")
    if failures:
        print(f"Failures: {len(failures)}")
        for msg in failures[:100]:
            print(f"  {msg}")
        if len(failures) > 100:
            print(f"  ... and {len(failures) - 100} more")
        return 1

    print("All sampled stems have all required files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

