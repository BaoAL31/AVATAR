#!/usr/bin/env python3
"""
Prepare LRS2-style clip folders for Hugging Face Hub upload (hardlink staging).

Expects layout under SRC:
  <SRC>/<stem>/<stem>.mp4  (+ json, txt, wav)

Skips known non-clip dirs (libreface_out, temp, etc.).

By default clips are **hardlinked** into staging (`cp -al`): originals stay under --src, almost no extra disk.
Use **--move** to **mv** each clip folder into staging instead (true “cut”; frees space under --src).

Two modes:

  --mode multi-repo (default)
    <STAGING>/LRS2_00/<stem>/...  -> dataset repo HBaoAL/LRS2_00 (and ... LRS2_14).

  --mode single-repo
    <STAGING>/shard_00/<stem>/... under one repo (e.g. HBaoAL/LRS2).

Hub notes:
  - Directory guidance: <=10k entries per folder — sharding fixes the “144k in one folder” issue.
  - ~144k clips × ~4 files ≈ ~580k files total; Hub also recommends ~100k *files* per repo.
    One repo may still trigger a warning; if the Hub complains, use --mode multi-repo or pack
    clips into fewer files (e.g. WebDataset tar shards).

Usage:
  python scripts/split_lrs2_for_hf_upload.py \\
    --src /home/you/lrs2/lrs2_webdataset \\
    --staging /scratch/you/lrs2_hf_staging \\
    --repo-prefix HBaoAL/LRS2

After linking:
  python3 /scratch/you/lrs2_hf_staging/upload_all_shards.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SKIP_DIR_NAMES = frozenset(
    {
        "libreface_out",
        "temp",
        "libreface_weights",
        "cache",
        ".git",
    }
)


def discover_clip_dirs(src: Path) -> list[Path]:
    if not src.is_dir():
        raise SystemExit(f"Not a directory: {src}")
    out: list[Path] = []
    for p in sorted(src.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.name in SKIP_DIR_NAMES:
            continue
        mp4 = p / f"{p.name}.mp4"
        if mp4.is_file():
            out.append(p)
    return out


def split_into_chunks(items: list[Path], n: int) -> list[list[Path]]:
    if n < 1:
        raise ValueError("n >= 1")
    if not items:
        return [[] for _ in range(n)]
    k, m = divmod(len(items), n)
    chunks: list[list[Path]] = []
    i = 0
    for j in range(n):
        sz = k + (1 if j < m else 0)
        chunks.append(items[i : i + sz])
        i += sz
    return chunks


def hardlink_tree(src: Path, dst: Path) -> None:
    """GNU cp -al: hardlink copy of a directory tree (same filesystem)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-al", str(src), str(dst)],
        check=True,
    )


def move_tree(src: Path, dst: Path) -> None:
    """Move directory (mv). Same filesystem is fast; cross-FS behaves like copy+delete."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"Refusing to mv: destination already exists: {dst}")
    subprocess.run(["mv", str(src), str(dst)], check=True)


def _upload_one_repo_script(staging: Path, repo_id: str) -> str:
    st = str(staging)
    return "\n".join(
        [
            "#!/usr/bin/env python3",
            "'''Upload staged tree (single repo) via upload_large_folder.'''",
            "from huggingface_hub import HfApi",
            "",
            f"FOLDER = {st!r}",
            f"REPO_ID = {repo_id!r}",
            "",
            "def main() -> None:",
            "    print(f'Uploading {FOLDER!r} -> {REPO_ID!r} (dataset)', flush=True)",
            "    HfApi().upload_large_folder(",
            "        repo_id=REPO_ID,",
            "        repo_type='dataset',",
            "        folder_path=FOLDER,",
            "    )",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )


def _upload_multi_repo_script(repo_prefix: str, n: int, staging: Path) -> str:
    st = str(staging)
    lines = [
        "#!/usr/bin/env python3",
        "'''Upload staged LRS2_* shards via upload_large_folder.",
        "Examples:",
        "  python3 upload_all_shards.py                    # all shards",
        "  python3 upload_all_shards.py --shard 3          # only LRS2_03 -> {prefix}_03",
        "  python3 upload_all_shards.py --start 0 --end 1  # only shard 0",
        "'''",
        "import argparse",
        "from pathlib import Path",
        "from huggingface_hub import HfApi",
        "",
        f"STAGING = Path({st!r})",
        f"REPO_PREFIX = {repo_prefix!r}",
        f"NUM_SHARDS = {n}",
        "",
        "def main() -> None:",
        "    ap = argparse.ArgumentParser(description='Upload HF dataset shards from STAGING.')",
        "    ap.add_argument('--start', type=int, default=0, help='first shard index (inclusive)')",
        "    ap.add_argument('--end', type=int, default=NUM_SHARDS, help='last shard index (exclusive)')",
        "    ap.add_argument('--shard', type=int, default=None, metavar='N',",
        f"                    help='upload only shard N (0..{n - 1}); overrides --start/--end')",
        "    args = ap.parse_args()",
        "    api = HfApi()",
        "    if args.shard is not None:",
        "        if not 0 <= args.shard < NUM_SHARDS:",
        "            raise SystemExit(f'--shard must be 0..{NUM_SHARDS - 1}')",
        "        indices = [args.shard]",
        "    else:",
        "        indices = list(range(args.start, min(args.end, NUM_SHARDS)))",
        "    for i in indices:",
        "        name = f'LRS2_{i:02d}'",
        "        folder = STAGING / name",
        "        rid = f'{REPO_PREFIX}_{i:02d}'",
        "        print(f'=== Upload {rid} from {folder} ===', flush=True)",
        "        api.upload_large_folder(",
        "            repo_id=rid,",
        "            repo_type='dataset',",
        "            folder_path=str(folder),",
        "        )",
        "",
        "if __name__ == '__main__':",
        "    main()",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shard LRS2 clip folders for Hugging Face (hardlink or move into staging)."
    )
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Dataset root with clip subdirs (<stem>/<stem>.mp4).",
    )
    ap.add_argument(
        "--staging",
        type=Path,
        required=True,
        help="Where to build shard folders. With --move, clips are moved here (removed from --src).",
    )
    ap.add_argument(
        "--move",
        action="store_true",
        help="Move (mv) each clip folder into staging instead of hardlinking. "
        "Frees space under --src; irreversible. Interrupting mid-run can leave clips split.",
    )
    ap.add_argument(
        "--mode",
        choices=("single-repo", "multi-repo"),
        default="multi-repo",
        help="multi-repo (default): LRS2_00.. -> HBaoAL/LRS2_00.. "
        "single-repo: shard_XX -> one --repo-id.",
    )
    ap.add_argument(
        "--num-splits",
        type=int,
        default=15,
        help="Number of shard folders (default 15, ~<=10k clips per shard for ~144k clips).",
    )
    ap.add_argument(
        "--repo-id",
        type=str,
        default="HBaoAL/LRS2",
        help="Full dataset repo id for --mode single-repo.",
    )
    ap.add_argument(
        "--repo-prefix",
        type=str,
        default="HBaoAL/LRS2",
        help="Repo id prefix for --mode multi-repo -> {prefix}_00, ...",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Remove existing --staging (entire tree) before building.",
    )
    args = ap.parse_args()

    src = args.src.resolve()
    staging = args.staging.resolve()
    n = args.num_splits

    clips = discover_clip_dirs(src)
    if not clips:
        raise SystemExit(f"No clip folders found under {src} (expected <stem>/<stem>.mp4).")

    chunks = split_into_chunks(clips, n)
    max_shard = max(len(c) for c in chunks)
    if max_shard > 10_000:
        print(
            f"WARNING: largest shard has {max_shard} folders (>10k Hub guidance). "
            f"Increase --num-splits (try {n * 2}).",
            file=sys.stderr,
        )

    approx_files = len(clips) * 4
    if args.mode == "single-repo" and approx_files > 100_000:
        print(
            f"NOTE: ~{approx_files} files total — Hub recommends ~100k files/repo (soft limit). "
            "If upload warns or fails, use --mode multi-repo or fewer files per clip.",
            file=sys.stderr,
        )

    print(f"Source: {src}")
    print(f"Staging: {staging}")
    print(f"Transfer: {'move (cut)' if args.move else 'hardlink (originals stay in --src)'}")
    print(f"Mode: {args.mode}")
    print(f"Clip folders found: {len(clips)}")
    print(f"Shards: {n}, sizes: {[len(c) for c in chunks]}")
    print()

    if args.dry_run:
        for i, ch in enumerate(chunks):
            if args.mode == "single-repo":
                name = f"shard_{i:02d}"
            else:
                name = f"LRS2_{i:02d}"
            print(f"  {name}: {len(ch)} clips -> {staging / name}")
        print("\nDry run done.")
        return

    if staging.exists():
        if args.force:
            subprocess.run(["rm", "-rf", str(staging)], check=True)
        else:
            raise SystemExit(
                f"Staging already exists: {staging}\n"
                "Use --force to delete it and rebuild."
            )

    staging.mkdir(parents=True)

    for i, ch in enumerate(chunks):
        if args.mode == "single-repo":
            shard_name = f"shard_{i:02d}"
        else:
            shard_name = f"LRS2_{i:02d}"
        shard_root = staging / shard_name
        shard_root.mkdir(parents=True)

        for clip in ch:
            dest = shard_root / clip.name
            if args.move:
                move_tree(clip, dest)
            else:
                hardlink_tree(clip, dest)

        print(f"OK {shard_name}: {len(ch)} clips -> {shard_root}")

    if args.mode == "single-repo":
        helper = staging / "upload_large_folder.py"
        helper.write_text(
            _upload_one_repo_script(staging, args.repo_id),
            encoding="utf-8",
        )
        helper.chmod(0o755)
        print()
        print(f"Wrote: {helper}")
        print(f"Upload one repo: python3 {helper}")
        print(f"  -> {args.repo_id} (dataset)")
    else:
        helper = staging / "upload_all_shards.py"
        helper.write_text(
            _upload_multi_repo_script(args.repo_prefix, n, staging),
            encoding="utf-8",
        )
        helper.chmod(0o755)
        print()
        print(f"Wrote: {helper}")
        print(f"Run: python3 {helper}                    # all shards")
        print(f"     python3 {helper} --shard 3           # only LRS2_03")
        print(f"     python3 {helper} --start 0 --end 3  # shards 0,1,2")


if __name__ == "__main__":
    main()
