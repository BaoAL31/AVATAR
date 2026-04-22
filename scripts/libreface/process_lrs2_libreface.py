#!/usr/bin/env python3
"""
LibreFace LRS2 tools: export *.npz from Hub videos, and/or upload flat NPZs to HF shard repos.

Shared data root (override with flags): ``/data/hoangbng`` — NPZs default to
``/data/hoangbng/libreface_out``, weights to ``/data/hoangbng/libreface_weights``.

**push** (default) — run **process**, then **upload** from ``--output-dir`` (same ``--repo-prefix`` / ``--shards``). On a fully successful Hub upload, local ``*.npz`` are removed by default and ``--temp-dir`` is emptied.

**process** — snapshot_download shard .mp4 from HF, run LibreFace, write flat ``<stem>.npz`` only.

**upload** — read flat ``NPZ_DIR/<stem>.npz``, commit as ``<stem>/<stem>.npz`` on dataset repo(s). By default removes each successfully uploaded local ``.npz`` and empties ``--temp-dir`` after a successful run (use ``--no-delete-after-upload`` to keep NPZs; ``--dry-run`` skips temp cleanup).

Auth for upload: ``huggingface-cli login`` or ``HF_TOKEN``.

Examples:
  # Export + upload to HF (default if you omit the subcommand)
  python process_lrs2_libreface.py --repo-prefix HBaoAL/LRS2
  python process_lrs2_libreface.py push --repo-prefix HBaoAL/LRS2 --shards 05 06 07

  # Export only (no upload)
  python process_lrs2_libreface.py process --repo-prefix HBaoAL/LRS2

  # Two parallel export+upload instances (each uploads only its NPZs under shared output-dir)
  python process_lrs2_libreface.py push --instance-id 0 --total-instances 2 &
  python process_lrs2_libreface.py push --instance-id 1 --total-instances 2 &

  # Upload only (uses /data/hoangbng/libreface_out by default)
  python process_lrs2_libreface.py upload --dry-run --repo-prefix HBaoAL/LRS2
  python process_lrs2_libreface.py upload --repo-prefix HBaoAL/LRS2 --shards 00 01 02
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

_REPO_USR = Path(__file__).resolve().parents[2] / "models" / "usr"
sys.path.insert(0, str(_REPO_USR))
from utils.hf_env import ensure_hf_env

ensure_hf_env()

# Shared SSH data root — NPZ / weights / upload source (override with CLI).
_DATA_DIR = Path("/data/hoangbng")
_DEFAULT_STEMS_FILE = Path("/home/hoangbng/AVATAR/AVATAR/data/hf_npz_stems.txt")


def _delete_local_npz_after_upload(args: argparse.Namespace) -> bool:
    """Default True unless ``--no-delete-after-upload`` was passed."""
    return not getattr(args, "no_delete_after_upload", False)


def _clean_dir_contents(path: Path) -> None:
    """Remove all files and subdirectories under ``path``; keep ``path`` itself."""
    path = path.resolve()
    if not path.is_dir():
        return
    for child in list(path.iterdir()):
        try:
            if child.is_symlink() or child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        except OSError as e:
            print(f"  WARNING: could not remove {child}: {e}", file=sys.stderr)


def _results_to_npz_dict(results) -> dict:
    import pandas as pd

    if isinstance(results, pd.DataFrame):
        return {str(c): results[c].to_numpy() for c in results.columns}
    if isinstance(results, dict):
        return {str(k): np.asarray(v) for k, v in results.items()}
    raise TypeError(
        f"Unexpected get_facial_attributes return type {type(results).__name__!r}; "
        "expected DataFrame or dict"
    )


def download_shard_mp4s(
    repo_prefix: str, suffix: str, *, max_workers: int = 16
) -> Path | None:
    """Download all .mp4 files for a shard repo. Returns local cache dir."""
    from huggingface_hub import snapshot_download

    repo_id = f"{repo_prefix}_{suffix}"
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*/*.mp4"],
            max_workers=max_workers,
        )
        return Path(local_dir)
    except Exception as e:
        print(f"  WARNING: could not download {repo_id}: {e}")
        return None


def discover_mp4s_in_cache(cache_dir: Path) -> list[Path]:
    """Find all <stem>/<stem>.mp4 in the cached snapshot dir."""
    videos = []
    for d in sorted(cache_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        candidate = d / f"{d.name}.mp4"
        if candidate.is_file():
            videos.append(candidate)
    return videos


def run_process(args: argparse.Namespace) -> int:
    try:
        import libreface  # noqa: F401
    except ImportError:
        raise SystemExit("libreface not found: pip install libreface")

    output_dir = args.output_dir.resolve()
    weights_dir = str(args.weights_dir.resolve())
    temp_dir = str(args.temp_dir.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes = (
        [s.zfill(2) for s in args.shards]
        if args.shards is not None
        else [f"{i:02d}" for i in range(15)]
    )

    processed = {f.stem for f in output_dir.glob("*.npz")}

    if args.stems_file is not None and args.stems_file.is_file():
        with args.stems_file.open(encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                stem = s.split("\t")[1] if "\t" in s else s
                processed.add(stem)
        print(f"Stems file: {len(processed)} stems already done (local + HF)")

    print(f"Instance {args.instance_id}/{args.total_instances} | device: {args.device}")
    print(f"Already processed: {len(processed)} stems")
    print(f"Shards to process: {suffixes}")
    print(
        f"Batch size: {args.batch_size} | LibreFace workers: {args.num_workers} | "
        f"HF download threads: {args.hf_max_workers}"
    )
    print(f"Compression: {'on' if args.compress else 'off (faster)'}")

    save_fn = np.savez_compressed if args.compress else np.savez
    succeeded = 0
    failed = 0
    interrupted = False

    for suffix in suffixes:
        repo_id = f"{args.repo_prefix}_{suffix}"
        print(f"\n--- Downloading .mp4 from {repo_id} ...", end=" ", flush=True)
        cache_dir = download_shard_mp4s(
            args.repo_prefix, suffix, max_workers=args.hf_max_workers
        )
        if cache_dir is None:
            continue
        print("done")

        all_videos = discover_mp4s_in_cache(cache_dir)
        remaining = [v for v in all_videos if v.stem not in processed]
        remaining = remaining[args.instance_id :: args.total_instances]

        if not remaining:
            print(f"  {repo_id}: all {len(all_videos)} clips already processed, skipping")
            continue

        print(f"  {repo_id}: {len(remaining)} clips to process ({len(all_videos)} total)")

        for video_path in tqdm(remaining, desc=f"LRS2_{suffix}"):
            output_path = output_dir / f"{video_path.stem}.npz"
            try:
                results = libreface.get_facial_attributes(
                    str(video_path),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    weights_download_dir=weights_dir,
                    temp_dir=temp_dir,
                    device=args.device,
                )
                save_fn(output_path, **_results_to_npz_dict(results))
                succeeded += 1
                processed.add(video_path.stem)
            except KeyboardInterrupt:
                print("\nInterrupted. Run again to resume.")
                interrupted = True
                break
            except Exception as e:
                print(f"\nFailed: {video_path.name}: {e}")
                failed += 1

        if interrupted:
            break

    print(f"\nDone. Succeeded: {succeeded}, Failed: {failed}")
    if interrupted:
        return 130
    return 0


# --- Upload (HF shard repos) ---


def list_repo_stems(api, repo_id: str, ext: str = ".mp4") -> set[str]:
    """Fetch the full file list for a repo once; return stems matching ext."""
    stems: set[str] = set()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"  WARNING: could not list {repo_id}: {e}", file=sys.stderr)
        return stems
    for f in files:
        if f.endswith(ext):
            stems.add(f.split("/")[0] if "/" in f else Path(f).stem)
    return stems


def build_hub_shard_map(api, repo_prefix: str, suffixes: list[str]) -> Dict[str, str]:
    """Query Hub repos to build stem -> shard suffix mapping from .mp4 files."""
    mapping: Dict[str, str] = {}
    for suffix in suffixes:
        repo_id = f"{repo_prefix}_{suffix}"
        print(f"  Listing clips in {repo_id} ...", end=" ", flush=True)
        hub_stems = list_repo_stems(api, repo_id, ext=".mp4")
        print(f"{len(hub_stems)} stems")
        for stem in hub_stems:
            mapping[stem] = suffix
    return mapping


def upload_batch(
    api,
    repo_id: str,
    operations: list,
    batch_num: int,
    total_batches: int,
) -> int:
    """Commit a batch of add operations. Returns number uploaded."""
    msg = f"Add {len(operations)} LibreFace .npz files (batch {batch_num}/{total_batches})"
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=msg,
    )
    return len(operations)


def run_upload(args: argparse.Namespace) -> int:
    from huggingface_hub import CommitOperationAdd, HfApi

    delete_npz = _delete_local_npz_after_upload(args)

    npz_dir = args.npz_dir.resolve()
    if not npz_dir.is_dir():
        raise SystemExit(f"Not a directory: {npz_dir}")

    npz_paths = sorted(npz_dir.glob("*.npz"))
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise SystemExit(f"No *.npz under {npz_dir}")

    api = HfApi()
    stem_to_shard: Dict[str, str] = {}

    if args.repo_id is not None:
        repo_groups: Dict[str, list[Path]] = {args.repo_id: npz_paths}
        print(f"Single-repo mode: all {len(npz_paths)} NPZs -> {args.repo_id}")
    else:
        suffixes = (
            [s.zfill(2) for s in args.shards]
            if args.shards is not None
            else [f"{i:02d}" for i in range(15)]
        )
        print(
            f"Building stem-to-shard map from Hub "
            f"({args.repo_prefix}_XX, {len(suffixes)} repos) ...",
            flush=True,
        )
        stem_to_shard = build_hub_shard_map(api, args.repo_prefix, suffixes)
        print(f"Mapped {len(stem_to_shard)} stems across {len(suffixes)} Hub repos.\n")

    all_npz_entries: set[tuple[str, str]] = set()
    already_done: set[str] = set()

    if args.stems_file is not None:
        args.stems_file.parent.mkdir(parents=True, exist_ok=True)
        if args.stems_file.is_file():
            needs_migrate = False
            with args.stems_file.open(encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if "\t" in s:
                        suf, stem = s.split("\t", 1)
                        all_npz_entries.add((suf, stem))
                        already_done.add(stem)
                    else:
                        shard = stem_to_shard.get(s)
                        if shard:
                            all_npz_entries.add((shard, s))
                        else:
                            all_npz_entries.add(("??", s))
                        already_done.add(s)
                        needs_migrate = True
            if needs_migrate:
                migrated = sum(1 for suf, _ in all_npz_entries if suf != "??")
                orphan = sum(1 for suf, _ in all_npz_entries if suf == "??")
                print(f"Migrated stems file: {migrated} with shard info, {orphan} unresolved")

        before = len(npz_paths)
        if delete_npz:
            skipped_paths = [p for p in npz_paths if p.stem in already_done]
            for p in skipped_paths:
                if p.is_file():
                    p.unlink()
            if skipped_paths:
                print(f"Deleted {len(skipped_paths)} local .npz already in stems file")
        npz_paths = [p for p in npz_paths if p.stem not in already_done]
        print(
            f"Stems file: {len(already_done)} stems already done, "
            f"skipping {before - len(npz_paths)} of {before} NPZs"
        )

    def flush_stems_file() -> None:
        if args.stems_file is None:
            return
        with args.stems_file.open("w", encoding="utf-8") as sf:
            for suf, stem in sorted(all_npz_entries):
                sf.write(f"{suf}\t{stem}\n")

    flush_stems_file()

    if not npz_paths and not args.dry_run:
        print("Nothing new to upload.")
        return 0

    if args.repo_id is not None:
        repo_groups = {args.repo_id: npz_paths}
    else:
        repo_groups = defaultdict(list)
        missing_shard: list[str] = []

        for npz_path in npz_paths:
            suffix = stem_to_shard.get(npz_path.stem)
            if suffix is None:
                missing_shard.append(npz_path.stem)
                continue
            repo_groups[f"{args.repo_prefix}_{suffix}"].append(npz_path)

        total_to_upload = sum(len(v) for v in repo_groups.values())
        print(f"Files to upload: {total_to_upload} across {len(repo_groups)} repos")
        if missing_shard:
            print(f"No matching clip on Hub for {len(missing_shard)} stems (skipped)")
            if len(missing_shard) <= 25:
                print(f"  Missing: {missing_shard}")
            else:
                print(f"  (first 20) {missing_shard[:20]} ...")

    if args.dry_run:
        for repo_id, paths in sorted(repo_groups.items()):
            print(f"  {repo_id}: {len(paths)} files")
            for p in paths[:5]:
                print(f"    {p.stem}/{p.stem}.npz")
            if len(paths) > 5:
                print(f"    ... and {len(paths) - 5} more")
        print("\nDry run only; no uploads.")
        return 0

    uploaded = 0
    skipped_exist = 0
    deleted = 0
    errors: list[tuple[str, str]] = []

    for repo_id, paths in sorted(repo_groups.items()):
        repo_suffix = repo_id.rsplit("_", 1)[-1] if "_" in repo_id else "00"

        existing_stems: set[str] = set()
        if args.skip_existing:
            print(f"Listing existing .npz in {repo_id} ...", flush=True)
            existing_stems = list_repo_stems(api, repo_id, ext=".npz")
            print(f"  {len(existing_stems)} already on Hub")
            new_existing = {s for s in existing_stems if (repo_suffix, s) not in all_npz_entries}
            if new_existing:
                for s in new_existing:
                    all_npz_entries.add((repo_suffix, s))
                flush_stems_file()

        to_upload: list[Path] = []
        for p in paths:
            if p.stem in existing_stems:
                skipped_exist += 1
                if delete_npz and p.is_file():
                    p.unlink()
                    deleted += 1
            else:
                to_upload.append(p)

        if not to_upload:
            print(f"{repo_id}: nothing new to upload")
            continue

        batches = [
            to_upload[i : i + args.batch_size]
            for i in range(0, len(to_upload), args.batch_size)
        ]
        print(
            f"{repo_id}: uploading {len(to_upload)} files in {len(batches)} "
            f"batches of <={args.batch_size}"
        )

        for batch_num, batch in enumerate(
            tqdm(batches, desc=repo_id, unit="batch"), start=1
        ):
            operations = [
                CommitOperationAdd(
                    path_in_repo=f"{p.stem}/{p.stem}.npz",
                    path_or_fileobj=str(p),
                )
                for p in batch
            ]
            try:
                n = upload_batch(api, repo_id, operations, batch_num, len(batches))
                uploaded += n
                for p in batch:
                    all_npz_entries.add((repo_suffix, p.stem))
                    if delete_npz and p.is_file():
                        p.unlink()
                        deleted += 1
                flush_stems_file()
            except Exception as e:
                for p in batch:
                    errors.append((p.stem, str(e)))

    print("\n--- Summary ---")
    print(f"NPZ files considered: {len(npz_paths)}")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped (already on Hub): {skipped_exist}")
    if delete_npz:
        print(f"Local .npz deleted: {deleted}")

    if errors:
        print(f"Errors: {len(errors)}")
        for stem, msg in errors[:15]:
            print(f"  {stem}: {msg}")
        if len(errors) > 15:
            print(f"  ... and {len(errors) - 15} more")
        return 1
    return 0


def _add_process_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-prefix", type=str, default="HBaoAL/LRS2",
                   help="HF repo prefix (repos are {prefix}_00 .. {prefix}_14).")
    p.add_argument("--shards", nargs="+", type=str, default=None, metavar="NN",
                   help="Shard suffixes to process (default: 00..14).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DATA_DIR / "libreface_out",
        help=f"Where to write *.npz (default: {_DATA_DIR / 'libreface_out'})",
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=_DATA_DIR / "libreface_weights",
        help=f"LibreFace weights download dir (default: {_DATA_DIR / 'libreface_weights'})",
    )
    p.add_argument(
        "--temp-dir",
        type=Path,
        default=_DATA_DIR / "libreface_out" / "temp",
        help=f"LibreFace temp dir (default: {_DATA_DIR / 'libreface_out' / 'temp'})",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=16,
                   help="LibreFace dataloader workers (not HF downloads).")
    p.add_argument(
        "--hf-max-workers",
        type=int,
        default=16,
        metavar="N",
        help="Concurrent file downloads per shard (huggingface_hub snapshot_download).",
    )
    p.add_argument("--instance-id", type=int, default=0)
    p.add_argument("--total-instances", type=int, default=1)
    p.add_argument("--compress", action="store_true",
                   help="Use np.savez_compressed (slower writes, smaller files).")
    p.add_argument(
        "--stems-file",
        type=Path,
        default=_DEFAULT_STEMS_FILE,
        help=f"Stems from upload script (default: {_DEFAULT_STEMS_FILE}); used for resume/skip.",
    )


def _add_push_upload_flags(p: argparse.ArgumentParser) -> None:
    """HF upload phase options (after LibreFace export). ``--output-dir`` is used as ``--npz-dir``."""
    p.add_argument(
        "--upload-batch-size",
        type=int,
        default=50,
        metavar="N",
        dest="upload_hf_commit_batch",
        help="NPZs per HF create_commit during upload (default: 50).",
    )
    p.add_argument(
        "--upload-dry-run",
        action="store_true",
        dest="upload_dry_run",
        help="Only print what would be uploaded (no HF commits).",
    )
    p.add_argument(
        "--upload-skip-existing",
        action="store_true",
        default=True,
        dest="upload_skip_existing",
        help="Skip stems whose .npz already exists on the Hub (default: enabled).",
    )
    p.add_argument(
        "--no-upload-skip-existing",
        action="store_false",
        dest="upload_skip_existing",
        help="Disable Hub-side existence checks before upload.",
    )
    p.add_argument(
        "--upload-limit",
        type=int,
        default=None,
        metavar="N",
        dest="upload_limit",
        help="Upload at most N local .npz files (sorted by path).",
    )
    p.add_argument(
        "--no-delete-after-upload",
        action="store_true",
        help="Keep local *.npz after upload (default: delete after each successful commit).",
    )
    p.add_argument(
        "--upload-repo-id",
        type=str,
        default=None,
        dest="upload_repo_id",
        help="Single HF dataset repo for upload (non-sharded). Else shard repos from --repo-prefix.",
    )


def namespace_for_upload_after_push(args: argparse.Namespace) -> argparse.Namespace:
    """Build ``run_upload`` namespace from ``push`` subparser args."""
    return argparse.Namespace(
        npz_dir=args.output_dir,
        repo_prefix=args.repo_prefix,
        repo_id=args.upload_repo_id,
        shards=args.shards,
        dry_run=args.upload_dry_run,
        skip_existing=args.upload_skip_existing,
        batch_size=args.upload_hf_commit_batch,
        limit=args.upload_limit,
        stems_file=args.stems_file,
        no_delete_after_upload=args.no_delete_after_upload,
    )


def _add_upload_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--npz-dir",
        type=Path,
        default=_DATA_DIR / "libreface_out",
        help=f"Directory of flat *.npz (default: {_DATA_DIR / 'libreface_out'})",
    )
    p.add_argument(
        "--repo-prefix",
        type=str,
        default="HBaoAL/LRS2",
        help="HF repo prefix; repos are {prefix}_00, {prefix}_01, etc.",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Single HF dataset repo (non-sharded). Mutually exclusive with shard mode.",
    )
    p.add_argument(
        "--shards",
        nargs="+",
        type=str,
        default=None,
        metavar="NN",
        help="Shard suffixes to scan, e.g. 00 01 14. Default: 00..14.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned uploads only.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip stems whose .npz already exists on the Hub (default: enabled).",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Disable Hub-side existence checks before upload.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of .npz files per HF commit (default: 50).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N npz files (sorted by path).",
    )
    p.add_argument(
        "--stems-file",
        type=Path,
        default=_DEFAULT_STEMS_FILE,
        help=f"Track uploaded stems (default: {_DEFAULT_STEMS_FILE}).",
    )
    p.add_argument(
        "--no-delete-after-upload",
        action="store_true",
        help="Keep local *.npz after upload (default: delete after each successful commit).",
    )
    p.add_argument(
        "--temp-dir",
        type=Path,
        default=_DATA_DIR / "libreface_out" / "temp",
        help="LibreFace temp directory to empty after a successful upload (default: export temp).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LibreFace NPZ export + HF upload for LRS2 (see module docstring).",
        epilog="Subcommands: push (default) — LibreFace then HF upload; process — export only; upload — HF only.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser(
        "push",
        help="Process (LibreFace export) then upload NPZs from --output-dir to HF.",
    )
    _add_process_args(p_push)
    _add_push_upload_flags(p_push)

    p_proc = sub.add_parser("process", help="Download shard .mp4, run LibreFace, write flat .npz only")
    _add_process_args(p_proc)

    p_up = sub.add_parser("upload", help="Upload flat .npz to HF as stem/stem.npz")
    _add_upload_args(p_up)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["push"]
    elif argv[0] not in ("process", "upload", "push"):
        argv = ["push"] + argv

    args = build_parser().parse_args(argv)
    if args.cmd == "push":
        proc_rc = run_process(args)
        if proc_rc != 0:
            print("\nSkipping upload (export did not finish normally).", file=sys.stderr)
            return proc_rc
        print("\n========== Upload phase ==========\n", flush=True)
        up_rc = run_upload(namespace_for_upload_after_push(args))
        if up_rc == 0 and not args.upload_dry_run:
            tdir = Path(args.temp_dir).resolve()
            print(f"\nCleaning LibreFace temp dir: {tdir}", flush=True)
            _clean_dir_contents(tdir)
        return up_rc
    if args.cmd == "process":
        return run_process(args)
    if args.cmd == "upload":
        up_rc = run_upload(args)
        if up_rc == 0 and not args.dry_run:
            tdir = Path(args.temp_dir).resolve()
            print(f"\nCleaning LibreFace temp dir: {tdir}", flush=True)
            _clean_dir_contents(tdir)
        return up_rc
    raise AssertionError(f"unknown cmd {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
