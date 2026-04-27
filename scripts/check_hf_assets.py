#!/usr/bin/env python3
"""Check HF assets for stems listed in one or more USR manifests.

For sampled rows from manifests (default: train + val), verify that each stem
has all required files in the same HF repo folder:
  <stem>/<stem>.mp4, .wav, .npz, .avi, .txt, .json

Optionally verify local landmarks for each sampled row:
  <landmarks_root>/<manifest_rel_path>.npy
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate required files for sampled manifest stems on HF."
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Manifest CSV to sample from. Repeat flag to pass multiple files. "
            "If omitted, defaults to train + val manifests."
        ),
    )
    ap.add_argument(
        "--repo-prefix",
        type=str,
        default="HBaoAL/LRS2",
        help="HF shard repo prefix, e.g. HBaoAL/LRS2.",
    )
    ap.add_argument(
        "--from-repos",
        action="store_true",
        help="Ignore manifests and scan stems directly from HF shard repos.",
    )
    ap.add_argument(
        "--shards",
        type=str,
        default="00-14",
        help="Shard range/list for --from-repos, e.g. 00-14 or 00,01,03.",
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
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N checked stems (0 disables periodic logs).",
    )
    ap.add_argument(
        "--required-exts",
        type=str,
        default=".mp4,.wav,.npz,.avi,.txt,.json",
        help="Comma-separated required extensions.",
    )
    ap.add_argument(
        "--landmarks-root",
        type=Path,
        default=Path("/home/hoangbng/Data/usr/landmarks"),
        help="Local landmarks root for <manifest_rel_path>.npy checks.",
    )
    ap.add_argument(
        "--skip-local-landmarks-check",
        action="store_true",
        help="Disable local landmarks validation.",
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


def parse_shards(spec: str) -> list[str]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        start, end = int(a), int(b)
        return [f"{i:02d}" for i in range(start, end + 1)]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(f"{int(part):02d}")
    return out


def iter_repo_rows(api: HfApi, repo_prefix: str, shards: Iterable[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for shard in shards:
        repo_id = f"{repo_prefix}_{shard}"
        tag = f"lrs2_{shard}"
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        for f in files:
            p = f.replace("\\", "/")
            if p.endswith(".mp4") and "/" in p:
                rows.append((tag, p))
    return rows


def main() -> int:
    args = parse_args()
    api = HfApi()
    manifests: list[Path] = []
    if args.from_repos:
        all_rows = iter_repo_rows(api, args.repo_prefix, parse_shards(args.shards))
        if not all_rows:
            print(
                f"No rows found from repos for prefix={args.repo_prefix} shards={args.shards}"
            )
            return 1
    else:
        manifests = args.manifest or [
            Path("/home/hoangbng/AVATAR/AVATAR/data/train_manifest.csv"),
            Path("/home/hoangbng/AVATAR/AVATAR/data/val_manifest.csv"),
        ]
        all_rows = []
        for manifest in manifests:
            rows = read_manifest_rows(manifest.resolve())
            if not rows:
                print(f"No rows found in {manifest}")
                return 1
            all_rows.extend(rows)

    rng = random.Random(args.seed)
    if args.sample_count and args.sample_count > 0 and len(all_rows) > args.sample_count:
        all_rows = rng.sample(all_rows, args.sample_count)

    required_exts = [e.strip() for e in args.required_exts.split(",") if e.strip()]

    # Group sampled rows by repo for one list_repo_files call per repo.
    rows_by_repo: dict[str, list[tuple[str, str]]] = {}
    for tag, rel in all_rows:
        repo = repo_from_tag(tag, args.repo_prefix)
        rows_by_repo.setdefault(repo, []).append((tag, rel))

    failures: list[str] = []
    landmark_failures: list[str] = []
    missing_by_ext: Counter[str] = Counter()
    checked = 0
    total_to_check = sum(len(v) for v in rows_by_repo.values())
    for repo_id, repo_rows in sorted(rows_by_repo.items()):
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        for _tag, rel in repo_rows:
            stem = Path(rel).stem
            folder = Path(rel).parent.as_posix()
            for ext in required_exts:
                target = f"{folder}/{stem}{ext}" if folder != "." else f"{stem}{ext}"
                if target not in files:
                    failures.append(f"{repo_id}: missing {target}")
                    missing_by_ext[ext] += 1
            if not args.skip_local_landmarks_check:
                lm_path = (args.landmarks_root / Path(rel)).with_suffix(".npy")
                if not lm_path.is_file():
                    landmark_failures.append(f"missing local landmarks: {lm_path}")
            checked += 1
            if args.progress_every > 0 and checked % args.progress_every == 0:
                ext_bits = ", ".join(
                    f"{ext}={missing_by_ext.get(ext, 0)}" for ext in required_exts
                )
                print(
                    f"[progress] checked {checked}/{total_to_check} | "
                    f"local_landmark_failures={len(landmark_failures)} | "
                    f"missing_by_ext: {ext_bits}"
                )

    print(f"Checked sampled stems: {checked}")
    if args.from_repos:
        print(f"Input source: HF repos (prefix={args.repo_prefix}, shards={args.shards})")
    else:
        print(f"Input manifests: {', '.join(str(m.resolve()) for m in manifests)}")
    print(f"Required extensions: {', '.join(required_exts)}")
    if not args.skip_local_landmarks_check:
        print(f"Local landmarks root: {args.landmarks_root.resolve()}")
    if failures:
        print(f"Failures: {len(failures)}")
        print(
            "HF missing by extension: "
            + ", ".join(f"{ext}={missing_by_ext.get(ext, 0)}" for ext in required_exts)
        )
        for msg in failures[:100]:
            print(f"  {msg}")
        if len(failures) > 100:
            print(f"  ... and {len(failures) - 100} more")
    if landmark_failures:
        print(f"Local landmark failures: {len(landmark_failures)}")
        for msg in landmark_failures[:100]:
            print(f"  {msg}")
        if len(landmark_failures) > 100:
            print(f"  ... and {len(landmark_failures) - 100} more")
    if failures or landmark_failures:
        return 1

    print("All sampled stems have all required files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

