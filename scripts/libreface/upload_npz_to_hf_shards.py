#!/usr/bin/env python3
"""
Upload flat LibreFace *.npz files into Hugging Face dataset repo(s), as:
  <stem>/<stem>.npz  (next to <stem>/<stem>.mp4 on the Hub).

--staging is your local clip root, e.g. /home/hoangbng/lrs2_hf_staging

Layout A — sharded local tree (after split_lrs2_for_hf_upload.py):
  /home/hoangbng/lrs2_hf_staging/LRS2_00/<stem>/<stem>.mp4
  /home/hoangbng/lrs2_hf_staging/LRS2_01/...
  -> uploads to HBaoAL/LRS2_00, HBaoAL/LRS2_01, ... using --repo-prefix

Layout B — flat local tree (no LRS2_XX folders):
  /home/hoangbng/lrs2_hf_staging/<stem>/<stem>.mp4
  -> set --repo-id to the single Hub dataset that holds those clips

NPZ inputs (flat on disk):
  NPZ_DIR/<stem>.npz

Auth: huggingface-cli login  or  HF_TOKEN

Examples:
  # Sharded local folders + multi repos
  python3 scripts/upload_npz_to_hf_shards.py --dry-run \\
    --npz-dir /home/hoangbng/lrs2/libreface_out \\
    --staging /home/hoangbng/lrs2_hf_staging \\
    --repo-prefix HBaoAL/LRS2

  # Flat local clip root + one Hub repo
  python3 scripts/upload_npz_to_hf_shards.py \\
    --npz-dir /home/hoangbng/lrs2/libreface_out \\
    --staging /home/hoangbng/lrs2_hf_staging \\
    --repo-id HBaoAL/LRS2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm

SHARD_RE = re.compile(r"^LRS2_(\d{2})$")


def iter_shard_dirs(staging: Path) -> list[Path]:
    out = []
    for p in sorted(staging.iterdir()):
        if p.is_dir() and SHARD_RE.match(p.name):
            out.append(p)
    return out


def find_shard_for_stem(staging: Path, stem: str) -> Path | None:
    """Return shard directory Path (e.g. .../LRS2_03) or None."""
    mp4_name = f"{stem}.mp4"
    for shard in iter_shard_dirs(staging):
        if (shard / stem / mp4_name).is_file():
            return shard
    return None


def clip_mp4_flat(staging: Path, stem: str) -> Path:
    """Layout B: staging/<stem>/<stem>.mp4"""
    return staging / stem / f"{stem}.mp4"


def shard_to_repo_id(repo_prefix: str, shard_dir: Path) -> str:
    m = SHARD_RE.match(shard_dir.name)
    if not m:
        raise ValueError(f"Not a shard dir: {shard_dir}")
    return f"{repo_prefix}_{m.group(1)}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload flat stem.npz files into HF shard repos (stem/stem.npz)."
    )
    ap.add_argument(
        "--npz-dir",
        type=Path,
        required=True,
        help="Directory of flat *.npz (one per clip stem).",
    )
    ap.add_argument(
        "--staging",
        type=Path,
        required=True,
        help="Local clip root, e.g. /home/hoangbng/lrs2_hf_staging "
        "(either LRS2_XX/<stem>/… or flat <stem>/<stem>.mp4).",
    )
    ap.add_argument(
        "--repo-prefix",
        type=str,
        default="HBaoAL/LRS2",
        help="With sharded local LRS2_XX dirs: repos {prefix}_00, {prefix}_01, …",
    )
    ap.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="With flat local layout (no LRS2_XX under --staging): single dataset id for all NPZs.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads only.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip upload if stem/stem.npz already exists on the Hub (extra API checks).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N npz files (sorted by path).",
    )
    args = ap.parse_args()

    npz_dir = args.npz_dir.resolve()
    staging = args.staging.resolve()

    if not npz_dir.is_dir():
        raise SystemExit(f"Not a directory: {npz_dir}")
    if not staging.is_dir():
        raise SystemExit(f"Not a directory: {staging}")

    npz_paths = sorted(npz_dir.glob("*.npz"))
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]

    if not npz_paths:
        raise SystemExit(f"No *.npz under {npz_dir}")

    shard_dirs = iter_shard_dirs(staging)
    use_sharded = len(shard_dirs) > 0
    if use_sharded and args.repo_id is not None:
        print(
            "Note: LRS2_XX shard dirs found under --staging; --repo-id ignored "
            "(using --repo-prefix for routing).",
            file=sys.stderr,
        )
    if not use_sharded and args.repo_id is None:
        raise SystemExit(
            "No LRS2_XX directories under --staging (flat clip layout). "
            "Set --repo-id to the Hub dataset that contains <stem>/<stem>.mp4, "
            "or restore sharded folders LRS2_00, LRS2_01, … under that path."
        )

    api = HfApi()

    missing_shard: list[str] = []
    uploaded = 0
    skipped_exist = 0
    errors: list[tuple[str, str]] = []

    for npz_path in tqdm(npz_paths, desc="NPZ uploads"):
        stem = npz_path.stem
        if use_sharded:
            shard_dir = find_shard_for_stem(staging, stem)
            if shard_dir is None:
                missing_shard.append(stem)
                continue
            repo_id = shard_to_repo_id(args.repo_prefix, shard_dir)
        else:
            if not clip_mp4_flat(staging, stem).is_file():
                missing_shard.append(stem)
                continue
            assert args.repo_id is not None
            repo_id = args.repo_id
        path_in_repo = f"{stem}/{stem}.npz"

        if args.dry_run:
            print(f"DRY  {npz_path.name} -> {repo_id}  @  {path_in_repo}")
            continue

        if args.skip_existing:
            try:
                if api.file_exists(
                    repo_id=repo_id,
                    filename=path_in_repo,
                    repo_type="dataset",
                ):
                    skipped_exist += 1
                    continue
            except Exception as e:
                errors.append((stem, f"file_exists check: {e}"))
                continue

        try:
            api.upload_file(
                path_or_fileobj=str(npz_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            uploaded += 1
        except Exception as e:
            errors.append((stem, str(e)))

    print("\n--- Summary ---")
    print(f"NPZ files considered: {len(npz_paths)}")
    if args.dry_run:
        print("Dry run only; no uploads.")
        return

    print(f"Uploaded: {uploaded}")
    print(f"Skipped (already on Hub): {skipped_exist}")
    print(
        f"No matching clip (no mp4 next to stem under --staging): {len(missing_shard)}"
    )
    if missing_shard and len(missing_shard) <= 25:
        print(f"  Missing: {missing_shard}")
    elif missing_shard:
        print(f"  (first 20) {missing_shard[:20]} ...")

    if errors:
        print(f"Errors: {len(errors)}")
        for stem, msg in errors[:15]:
            print(f"  {stem}: {msg}")
        if len(errors) > 15:
            print(f"  ... and {len(errors) - 15} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
