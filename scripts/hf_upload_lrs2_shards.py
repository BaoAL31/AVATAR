#!/usr/bin/env python3
"""
Upload LRS2 HF staging shards (folders LRS2_00, LRS2_01, ...) with upload_large_folder.

Standalone — copy to your server; does not require split_lrs2_for_hf_upload.py.

Examples:
  python3 hf_upload_lrs2_shards.py --staging ~/lrs2_hf_staging --repo-prefix HBaoAL/LRS2
  python3 hf_upload_lrs2_shards.py --staging ~/lrs2_hf_staging --repo-prefix HBaoAL/LRS2 --shard 3
  python3 hf_upload_lrs2_shards.py --staging ~/lrs2_hf_staging --repo-prefix HBaoAL/LRS2 --start 0 --end 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload LRS2_* shard folders to Hugging Face datasets.")
    ap.add_argument(
        "--staging",
        type=Path,
        required=True,
        help="Directory containing LRS2_00, LRS2_01, ...",
    )
    ap.add_argument(
        "--repo-prefix",
        type=str,
        required=True,
        help="Hub repo id prefix (shard i -> {prefix}_{i:02d}).",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=15,
        help="How many shards exist (default 15).",
    )
    ap.add_argument("--start", type=int, default=0, help="first shard index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="exclusive; default = num-shards")
    ap.add_argument(
        "--shard",
        type=int,
        default=None,
        metavar="N",
        help="upload only shard N; overrides --start/--end",
    )
    args = ap.parse_args()

    staging = args.staging.resolve()
    n = args.num_shards
    end = args.end if args.end is not None else n

    if args.shard is not None:
        if not 0 <= args.shard < n:
            raise SystemExit(f"--shard must be between 0 and {n - 1}")
        indices = [args.shard]
    else:
        indices = list(range(args.start, min(end, n)))

    api = HfApi()
    for i in indices:
        name = f"LRS2_{i:02d}"
        folder = staging / name
        if not folder.is_dir():
            raise SystemExit(f"Missing shard folder: {folder}")
        rid = f"{args.repo_prefix}_{i:02d}"
        print(f"=== Upload {rid} <- {folder} ===", flush=True)
        api.upload_large_folder(
            repo_id=rid,
            repo_type="dataset",
            folder_path=str(folder),
        )


if __name__ == "__main__":
    main()
