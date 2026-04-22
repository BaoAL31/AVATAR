#!/usr/bin/env python3
"""
Build USR fine-tuning manifest CSVs for LRS2.

Two modes:
  --from-hub   Build entirely from HF Hub repos (no local files needed).
               Reads --npz-stems-file (required, format: suffix<TAB>stem per line)
               to get the clip list and shard mapping. Per shard: snapshot_download
               (.txt + .mp4), then manifest rows for that shard (no waiting for all
               shards to finish downloading). Frame count = 0 unless .mp4 is present.

  (default)    Build from local --dataset-root with .txt + .mp4 pairs.

Examples:
  # Hub-only (recommended — no local files needed):
  python build_lrs2_usr_manifest.py \\
    --from-hub \\
    --repo-prefix HBaoAL/LRS2 \\
    --units /home/hoangbng/AVATAR/AVATAR/models/usr/utils/labels/unigram1000_units.txt \\
    --out-dir /home/hoangbng/AVATAR/AVATAR/data \\
    --npz-stems-file /home/hoangbng/AVATAR/AVATAR/data/hf_npz_stems.txt \\
    --val-ratio 0.01 --fresh

  Small manifests (debug):
  --max-train-rows 5000 --max-val-rows 200
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "AVATAR" / "AVATAR" / "models" / "usr"))
from utils.hf_env import ensure_hf_env

ensure_hf_env()

from typing import Dict, Optional, Set

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    def _tqdm(iterable, **_kwargs):  # type: ignore[misc,no-redef]
        return iterable

SHARD_RE = re.compile(r"^LRS2_(\d{2})$")


def load_unigram_table(units_path: Path) -> tuple[dict, int]:
    table = {}
    with units_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token, tid = line.rsplit(None, 1)
            table[token] = int(tid)
    unk_id = table.get("<unk>", 1)
    return table, unk_id


def words_to_token_strings(upper_sentence: str) -> list[str]:
    words = re.findall(r"[A-Z0-9']+", upper_sentence.upper())
    return ["\u2581" + w for w in words]


def text_to_ids(text: str, table: dict, unk_id: int) -> list[int]:
    toks = words_to_token_strings(text)
    return [table.get(t, unk_id) for t in toks]


def parse_transcript_txt(txt_path: Path) -> str | None:
    with txt_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("text:"):
                return s.split(":", 1)[1].strip()
    return None



def video_frame_count(mp4_path: Path) -> int | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n if n > 0 else None


def is_validation_split(key: str, val_ratio: float, seed: int) -> bool:
    if val_ratio <= 0:
        return False
    h = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(2**64)
    return u < val_ratio


def format_manifest_line(tag: str, rel_mp4: str, frame_count: int, ids: list[int]) -> str:
    label = " ".join(str(i) for i in ids)
    return f"{tag},{rel_mp4},{frame_count},{label}"


def load_checkpoint(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                done.add(s)
    return done


def append_checkpoint(path: Path, key: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(key + "\n")
        f.flush()


# ── Local mode helpers ──

def build_shard_map(staging: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for d in sorted(staging.iterdir()):
        m = SHARD_RE.match(d.name)
        if not m or not d.is_dir():
            continue
        suffix = m.group(1)
        for clip_dir in d.iterdir():
            if clip_dir.is_dir() and (clip_dir / f"{clip_dir.name}.mp4").is_file():
                mapping[clip_dir.name] = suffix
    return mapping


def discover_samples(dataset_root: Path, recursive: bool) -> list[Path]:
    txts = sorted(dataset_root.rglob("*.txt") if recursive else dataset_root.glob("*.txt"))
    return [tp for tp in txts if tp.with_suffix(".mp4").is_file()]


# ── Hub mode helpers ──

def load_stems_file(path: Path) -> list[tuple[str, str]]:
    """Read stems file. Format: suffix<TAB>stem per line (or bare stem)."""
    entries: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "\t" in s:
                suffix, stem = s.split("\t", 1)
                entries.append((suffix, stem))
            else:
                entries.append(("??", s))
    return entries



def run_hub_mode(args, table: dict, unk_id: int) -> None:
    """Build manifest from HF Hub. Reads stems file for clip list + shard info."""
    from huggingface_hub import snapshot_download

    if args.npz_stems_file is None:
        print("ERROR: --from-hub requires --npz-stems-file (format: suffix<TAB>stem)", file=sys.stderr)
        sys.exit(1)

    sf = args.npz_stems_file.resolve()
    if not sf.is_file():
        print(f"ERROR: npz-stems-file not found: {sf}", file=sys.stderr)
        sys.exit(1)

    entries = load_stems_file(sf)
    bad = [stem for suf, stem in entries if suf == "??"]
    if bad:
        print(
            f"WARNING: {len(bad)} stems have no shard info (old format). "
            f"Re-run snapshot_hf_npz_stems.py to fix.",
            file=sys.stderr,
        )
    candidates = [(suf, stem) for suf, stem in entries if suf != "??"]
    print(f"Loaded {len(candidates)} stems with shard info from {sf}")

    # Group by shard so we can bulk-download .txt files per repo
    from collections import defaultdict
    by_shard: dict[str, list[str]] = defaultdict(list)
    for suf, stem in candidates:
        by_shard[suf].append(stem)
    unique_suffixes = sorted(by_shard.keys())
    print(f"Shards to fetch transcripts from: {unique_suffixes}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_manifest.csv"
    val_path = args.out_dir / "val_manifest.csv"
    checkpoint_path = args.out_dir / ".usr_manifest_checkpoint.txt"

    if args.fresh:
        for p in (train_path, val_path, checkpoint_path):
            if p.exists():
                p.unlink()
        done: set[str] = set()
        print("--fresh: removed previous outputs and checkpoint.")
    else:
        done = load_checkpoint(checkpoint_path)
        print(f"Resume: {len(done)} keys already in checkpoint.")

    mode_train = "a" if train_path.exists() and not args.fresh else "w"
    mode_val = "a" if val_path.exists() and not args.fresh else "w"
    if args.fresh:
        mode_train = mode_val = "w"

    n_new_train = n_new_val = n_skip = n_already = 0
    f_train = train_path.open(mode_train, encoding="utf-8")
    f_val = val_path.open(mode_val, encoding="utf-8")

    use_pbar = _TQDM_AVAILABLE and not args.no_progress_bar
    total_clips = len(candidates)
    global_idx = 0
    stop = False

    try:
        for suffix in unique_suffixes:
            repo_id = f"{args.repo_prefix}_{suffix}"
            n_stems = len(by_shard[suffix])
            print(f"\n=== {repo_id} ({n_stems} stems) ===")
            print("Downloading .txt + .mp4 ...", end=" ", flush=True)
            try:
                local_dir = Path(
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        allow_patterns=["*/*.txt", "*/*.mp4"],
                        max_workers=args.hf_max_workers,
                    )
                )
                print("done")
            except Exception as e:
                print(f"FAILED: {e}", file=sys.stderr)
                continue

            print(f"Building manifest for {repo_id} ...")
            shard_pairs = [(suffix, stem) for stem in by_shard[suffix]]
            inner_iter = (
                _tqdm(shard_pairs, desc=f"Manifest LRS2_{suffix}", unit="clip")
                if use_pbar
                else shard_pairs
            )

            for suffix, stem in inner_iter:
                global_idx += 1
                try:
                    key = f"{stem}/{stem}.mp4"

                    if key in done:
                        n_already += 1
                        continue

                    txt_path = local_dir / stem / f"{stem}.txt"
                    if not txt_path.is_file():
                        n_skip += 1
                        continue

                    text = parse_transcript_txt(txt_path)
                    if not text:
                        n_skip += 1
                        continue

                    ids = text_to_ids(text, table, unk_id)
                    if not ids:
                        n_skip += 1
                        continue

                    mp4_path = local_dir / stem / f"{stem}.mp4"
                    fc = video_frame_count(mp4_path) if mp4_path.is_file() else None
                    if fc is None or fc <= 0:
                        if args.skip_missing_frames:
                            n_skip += 1
                            continue
                        fc = 0

                    clip_tag = f"{args.tag}_{suffix}"
                    manifest_path = f"{stem}/{stem}.mp4"
                    line = format_manifest_line(clip_tag, manifest_path, fc, ids)

                    if is_validation_split(key, args.val_ratio, args.seed):
                        f_val.write(line + "\n")
                        f_val.flush()
                        n_new_val += 1
                    else:
                        f_train.write(line + "\n")
                        f_train.flush()
                        n_new_train += 1

                    append_checkpoint(checkpoint_path, key)
                    done.add(key)
                    if (args.max_train_rows is not None and n_new_train >= args.max_train_rows) or (
                        args.max_val_rows is not None and n_new_val >= args.max_val_rows
                    ):
                        stop = True
                finally:
                    if use_pbar:
                        inner_iter.set_postfix(
                            new=n_new_train + n_new_val,
                            skip=n_skip,
                            done=n_already,
                            refresh=False,
                        )
                    elif args.progress_every and global_idx % args.progress_every == 0:
                        print(
                            f"scan {global_idx}/{total_clips} | "
                            f"new {n_new_train + n_new_val} (tr {n_new_train} val {n_new_val}) | "
                            f"skip {n_skip} | ckpt_hit {n_already}",
                            flush=True,
                        )
                if stop:
                    break
            if stop:
                break
    finally:
        f_train.close()
        f_val.close()

    print()
    print(f"Done. New rows: train +{n_new_train}, val +{n_new_val}")
    print(f"Skipped (no transcript / empty ids): {n_skip}")
    print(f"Skipped (already in checkpoint): {n_already}")
    print(f"Outputs: {train_path}")
    print(f"         {val_path}")


def run_local_mode(args, table: dict, unk_id: int) -> None:
    """Build manifest from local .txt + .mp4 pairs."""
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: dataset-root not a directory: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    npz_dir = None
    if args.require_npz:
        if args.npz_dir is None:
            print("ERROR: --require-npz requires --npz-dir", file=sys.stderr)
            sys.exit(1)
        npz_dir = args.npz_dir.resolve()
        if not npz_dir.is_dir():
            print(f"ERROR: npz-dir not a directory: {npz_dir}", file=sys.stderr)
            sys.exit(1)

    stems_file_set: Optional[Set[str]] = None
    if args.npz_stems_file is not None:
        sf = args.npz_stems_file.resolve()
        if not sf.is_file():
            print(f"ERROR: npz-stems-file not found: {sf}", file=sys.stderr)
            sys.exit(1)
        entries = load_stems_file(sf)
        stems_file_set = {stem for _, stem in entries}
        print(f"Loaded {len(stems_file_set)} stems from {sf}")

    shard_map: Optional[Dict[str, str]] = None
    if args.staging is not None:
        staging_dir = args.staging.resolve()
        if not staging_dir.is_dir():
            print(f"ERROR: staging not a directory: {staging_dir}", file=sys.stderr)
            sys.exit(1)
        shard_map = build_shard_map(staging_dir)
        if not shard_map:
            shard_map = None
        else:
            print(f"Shard map: {len(shard_map)} stems across staging shards.")

    txts = discover_samples(dataset_root, args.recursive)
    if not txts:
        print("ERROR: No paired .txt + .mp4 found.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_manifest.csv"
    val_path = args.out_dir / "val_manifest.csv"
    checkpoint_path = args.out_dir / ".usr_manifest_checkpoint.txt"

    if args.fresh:
        for p in (train_path, val_path, checkpoint_path):
            if p.exists():
                p.unlink()
        done: set[str] = set()
        print("--fresh: removed previous outputs and checkpoint.")
    else:
        done = load_checkpoint(checkpoint_path)
        print(f"Resume: {len(done)} keys already in checkpoint.")

    mode_train = "a" if train_path.exists() and not args.fresh else "w"
    mode_val = "a" if val_path.exists() and not args.fresh else "w"
    if args.fresh:
        mode_train = mode_val = "w"

    n_new_train = n_new_val = n_skip = n_already = 0
    f_train = train_path.open(mode_train, encoding="utf-8")
    f_val = val_path.open(mode_val, encoding="utf-8")

    use_pbar = _TQDM_AVAILABLE and not args.no_progress_bar
    bar_iter = _tqdm(txts, total=len(txts), desc="Manifest", unit="clip") if use_pbar else txts
    stop = False

    try:
        for idx, tp in enumerate(bar_iter):
            try:
                mp4 = tp.with_suffix(".mp4")
                rel = mp4.relative_to(dataset_root)
                rel_posix = rel.as_posix()
                key = rel_posix

                if key in done:
                    n_already += 1
                    continue

                if args.require_npz and npz_dir is not None:
                    if not (npz_dir / f"{mp4.stem}.npz").is_file():
                        n_skip += 1
                        continue

                if stems_file_set is not None:
                    if mp4.stem not in stems_file_set:
                        n_skip += 1
                        continue

                text = parse_transcript_txt(tp)
                if not text:
                    n_skip += 1
                    continue

                ids = text_to_ids(text, table, unk_id)
                if not ids:
                    n_skip += 1
                    continue

                fc = video_frame_count(mp4)
                if fc is None or fc <= 0:
                    if args.skip_missing_frames:
                        n_skip += 1
                        continue
                    fc = 0

                clip_tag = args.tag
                manifest_path = rel_posix
                if shard_map is not None:
                    shard_suffix = shard_map.get(mp4.stem)
                    if shard_suffix is not None:
                        clip_tag = f"{args.tag}_{shard_suffix}"
                        shard_dir = dataset_root / f"LRS2_{shard_suffix}"
                        try:
                            manifest_path = mp4.relative_to(shard_dir).as_posix()
                        except ValueError:
                            pass

                line = format_manifest_line(clip_tag, manifest_path, fc, ids)
                if is_validation_split(rel_posix, args.val_ratio, args.seed):
                    f_val.write(line + "\n")
                    f_val.flush()
                    n_new_val += 1
                else:
                    f_train.write(line + "\n")
                    f_train.flush()
                    n_new_train += 1

                append_checkpoint(checkpoint_path, key)
                done.add(key)
                if (args.max_train_rows is not None and n_new_train >= args.max_train_rows) or (
                    args.max_val_rows is not None and n_new_val >= args.max_val_rows
                ):
                    stop = True
            finally:
                if use_pbar:
                    bar_iter.set_postfix(
                        new=n_new_train + n_new_val, skip=n_skip, done=n_already, refresh=False,
                    )
                elif args.progress_every and (idx + 1) % args.progress_every == 0:
                    print(
                        f"scan {idx + 1}/{len(txts)} | "
                        f"new {n_new_train + n_new_val} | skip {n_skip} | ckpt {n_already}",
                        flush=True,
                    )
            if stop:
                break
    finally:
        f_train.close()
        f_val.close()

    print()
    print(f"Done. New rows: train +{n_new_train}, val +{n_new_val}")
    print(f"Skipped: {n_skip} | Already in checkpoint: {n_already}")
    print(f"Outputs: {train_path}\n         {val_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build resumable USR lrs2 manifest CSVs.")
    ap.add_argument("--from-hub", action="store_true",
                    help="Build entirely from HF Hub (no local files needed). "
                    "Requires --npz-stems-file with format: suffix<TAB>stem.")
    ap.add_argument("--repo-prefix", type=str, default="HBaoAL/LRS2",
                    help="HF repo prefix (repos are {prefix}_00, {prefix}_01, …).")
    ap.add_argument("--npz-stems-file", type=Path, default=None,
                    help="Stems file (suffix<TAB>stem per line from upload script).")
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--units", type=Path, required=True, help="unigram1000_units.txt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--tag", default="lrs2")
    ap.add_argument("--staging", type=Path, default=None)
    ap.add_argument("--skip-missing-frames", action="store_true")
    ap.add_argument("--require-npz", action="store_true")
    ap.add_argument("--npz-dir", type=Path, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--no-progress-bar", action="store_true")
    ap.add_argument(
        "--hf-max-workers",
        type=int,
        default=16,
        metavar="N",
        help="Concurrent file downloads per shard (snapshot_download). Default: 16.",
    )
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop after writing N train rows (val also stops if --max-val-rows hit first).",
    )
    ap.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop after writing N val rows.",
    )
    args = ap.parse_args()

    table, unk_id = load_unigram_table(args.units.resolve())
    print(f"Loaded {len(table)} tokens; <unk> id = {unk_id}")

    if args.from_hub:
        run_hub_mode(args, table, unk_id)
    else:
        if args.dataset_root is None:
            print("ERROR: --dataset-root required in local mode (or use --from-hub)", file=sys.stderr)
            sys.exit(1)
        run_local_mode(args, table, unk_id)


if __name__ == "__main__":
    main()
