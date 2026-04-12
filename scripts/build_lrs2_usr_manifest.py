#!/usr/bin/env python3
"""
Build USR fine-tuning manifest CSVs from LRS2-style samples (single file; copy to D: etc.).

Expected layout per sample (same basename):
  000000.mp4
  000000.txt   (line: Text: ...)
  000000.json  (optional)

Each output row:
  lrs2,<relative_path_to_mp4>,<frame_count>,<space-separated unigram1000 token ids>

With --staging (HF Hub upload staging tree):
  Tag becomes shard-aware (lrs2_00, lrs2_01, ...) matching the repo each clip was uploaded to.
  E.g.:  lrs2_03,<stem>/<stem>.mp4,<frame_count>,<label>
  Training resolves lrs2_03 + hub.lrs2_repo_prefix HBaoAL/LRS2 → repo HBaoAL/LRS2_03

Outputs (UTF-8):
  train_manifest.csv
  val_manifest.csv
  .usr_manifest_checkpoint.txt   (one completed sample key per line — for resume)

Resume: re-run the same command; completed keys are skipped. Use --fresh to rebuild from scratch.

With --require-npz, expects a flat npz dir: <npz-dir>/<stem>.npz matching each mp4 basename
(stem collisions across folders are not supported).

Do not delete the CSVs while keeping the checkpoint — use --fresh to reset both.

Install tqdm for a progress bar: pip install tqdm

Example (local, plain lrs2 tag):
  python build_lrs2_usr_manifest.py \\
    --dataset-root /mnt/d/lrs2_webdataset \\
    --units /path/to/unigram1000_units.txt \\
    --out-dir /mnt/d/lrs2_manifests \\
    --require-npz \\
    --npz-dir /mnt/d/libreface_out \\
    --val-ratio 0.01

Example (HF sharded, shard-aware tags):
  python build_lrs2_usr_manifest.py \\
    --dataset-root /mnt/d/lrs2_webdataset \\
    --staging /home/hoangbng/lrs2_hf_staging \\
    --units /path/to/unigram1000_units.txt \\
    --out-dir /mnt/d/lrs2_manifests \\
    --require-npz \\
    --npz-dir /mnt/d/libreface_out \\
    --val-ratio 0.01
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from tqdm import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

    def _tqdm(iterable, **_kwargs):  # type: ignore[misc,no-redef]
        return iterable


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


def parse_transcript_txt(txt_path: Path) -> str | None:
    text = None
    with txt_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("text:"):
                text = s.split(":", 1)[1].strip()
                break
    return text


def words_to_token_strings(upper_sentence: str) -> list[str]:
    words = re.findall(r"[A-Z0-9']+", upper_sentence.upper())
    return ["▁" + w for w in words]


def text_to_ids(text: str, table: dict, unk_id: int) -> list[int]:
    toks = words_to_token_strings(text)
    return [table.get(t, unk_id) for t in toks]


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


SHARD_RE = re.compile(r"^LRS2_(\d{2})$")


def build_shard_map(staging: Path) -> Dict[str, str]:
    """Map clip stem → shard suffix (e.g. '03') from staging/LRS2_XX/<stem>/<stem>.mp4."""
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
    out: list[Path] = []
    for tp in txts:
        if tp.with_suffix(".mp4").is_file():
            out.append(tp)
    return out


def sample_key(rel_mp4_posix: str) -> str:
    """Stable id for checkpointing (relative mp4 path from dataset root)."""
    return rel_mp4_posix


def is_validation_split(rel_mp4_posix: str, val_ratio: float, seed: int) -> bool:
    """Deterministic train/val assignment (reproducible across resume runs)."""
    if val_ratio <= 0:
        return False
    h = hashlib.sha256(f"{seed}|{rel_mp4_posix}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(2**64)
    return u < val_ratio


def format_manifest_line(tag: str, rel_mp4_posix: str, frame_count: int, ids: list[int]) -> str:
    label = " ".join(str(i) for i in ids)
    return f"{tag},{rel_mp4_posix},{frame_count},{label}"


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.is_file():
        return set()
    done = set()
    with checkpoint_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                done.add(s)
    return done


def append_checkpoint(checkpoint_path: Path, key: str) -> None:
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(key + "\n")
        f.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build resumable USR lrs2 manifest CSVs.")
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--units", type=Path, required=True, help="unigram1000_units.txt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--tag", default="lrs2")
    ap.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="HF upload staging dir with LRS2_XX/<stem>/... folders. "
        "When set, tags become shard-aware (e.g. lrs2_03) matching the upload repos.",
    )
    ap.add_argument(
        "--skip-missing-frames",
        action="store_true",
        help="Skip sample if OpenCV cannot read frame count.",
    )
    ap.add_argument(
        "--require-npz",
        action="store_true",
        help="Only include clips whose LibreFace output exists: <npz-dir>/<mp4_stem>.npz",
    )
    ap.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        help="Directory of *.npz (flat; one file per video stem). Required with --require-npz.",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoint and truncate CSVs (start over).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Without tqdm: print status every N scanned transcripts (0 = quiet).",
    )
    ap.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm bar; use --progress-every for text updates instead.",
    )
    args = ap.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: dataset-root not a directory: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    units_path = args.units.resolve()
    if not units_path.is_file():
        print(f"ERROR: units not found: {units_path}", file=sys.stderr)
        sys.exit(1)

    if args.require_npz:
        if args.npz_dir is None:
            print("ERROR: --require-npz requires --npz-dir", file=sys.stderr)
            sys.exit(1)
        npz_dir = args.npz_dir.resolve()
        if not npz_dir.is_dir():
            print(f"ERROR: npz-dir not a directory: {npz_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        npz_dir = None

    shard_map: Optional[Dict[str, str]] = None
    if args.staging is not None:
        staging_dir = args.staging.resolve()
        if not staging_dir.is_dir():
            print(f"ERROR: staging not a directory: {staging_dir}", file=sys.stderr)
            sys.exit(1)
        shard_map = build_shard_map(staging_dir)
        if not shard_map:
            print(
                f"WARNING: no LRS2_XX/<stem>/<stem>.mp4 found under --staging {staging_dir}; "
                "all clips will use plain --tag.",
                file=sys.stderr,
            )
            shard_map = None
        else:
            print(f"Shard map: {len(shard_map)} stems across staging shards.")

    table, unk_id = load_unigram_table(units_path)
    print(f"Loaded {len(table)} tokens; <unk> id = {unk_id}")

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
        done = set()
        print("--fresh: removed previous outputs and checkpoint.")
    else:
        done = load_checkpoint(checkpoint_path)
        print(f"Resume: {len(done)} keys already in checkpoint.")

    mode_train = "a" if train_path.exists() and not args.fresh else "w"
    mode_val = "a" if val_path.exists() and not args.fresh else "w"
    if args.fresh:
        mode_train = mode_val = "w"

    n_new_train = n_new_val = n_skip = n_already = 0

    try:
        f_train = train_path.open(mode_train, encoding="utf-8")
        f_val = val_path.open(mode_val, encoding="utf-8")
    except OSError as e:
        print(f"ERROR opening outputs: {e}", file=sys.stderr)
        sys.exit(1)

    total = len(txts)
    use_pbar = _TQDM_AVAILABLE and not args.no_progress_bar
    if use_pbar:
        bar_iter = _tqdm(
            txts,
            total=total,
            desc="Manifest",
            unit="clip",
            dynamic_ncols=True,
        )
    else:
        bar_iter = txts
        if not _TQDM_AVAILABLE and not args.no_progress_bar:
            print("Tip: pip install tqdm for a progress bar.", file=sys.stderr)

    try:
        for idx, tp in enumerate(bar_iter):
            try:
                mp4 = tp.with_suffix(".mp4")
                rel = mp4.relative_to(dataset_root)
                rel_posix = rel.as_posix()
                key = sample_key(rel_posix)

                if key in done:
                    n_already += 1
                    continue

                if args.require_npz and npz_dir is not None:
                    npz_path = npz_dir / f"{mp4.stem}.npz"
                    if not npz_path.is_file():
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
                if shard_map is not None:
                    shard_suffix = shard_map.get(mp4.stem)
                    if shard_suffix is not None:
                        clip_tag = f"{args.tag}_{shard_suffix}"
                line = format_manifest_line(clip_tag, rel_posix, fc, ids)
                is_val = is_validation_split(rel_posix, args.val_ratio, args.seed)
                if is_val:
                    f_val.write(line + "\n")
                    f_val.flush()
                    n_new_val += 1
                else:
                    f_train.write(line + "\n")
                    f_train.flush()
                    n_new_train += 1

                append_checkpoint(checkpoint_path, key)
                done.add(key)
            finally:
                if use_pbar:
                    bar_iter.set_postfix(
                        new=n_new_train + n_new_val,
                        skip=n_skip,
                        done=n_already,
                        refresh=False,
                    )
                else:
                    pe = args.progress_every
                    if pe and (idx + 1) % pe == 0:
                        print(
                            f"scan {idx + 1}/{total} | "
                            f"new_rows {n_new_train + n_new_val} "
                            f"(tr {n_new_train} val {n_new_val}) | "
                            f"skip {n_skip} | ckpt_hit {n_already}",
                            flush=True,
                        )
    finally:
        f_train.close()
        f_val.close()

    print()
    print(f"Done. New rows: train +{n_new_train}, val +{n_new_val}")
    print(f"Skipped (no text / empty ids / no npz / etc.): {n_skip}")
    print(f"Skipped (already in checkpoint): {n_already}")
    print(f"Outputs: {train_path}")
    print(f"         {val_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print()
    if shard_map is not None:
        print("Hydra (HF sharded, example):")
        print(f"  data.dataset.train_csv={train_path}")
        print(f"  data.dataset.val_csv={val_path}")
        print(f"  data.hub.lrs2_repo_prefix=HBaoAL/LRS2")
    else:
        print("Hydra (local, example):")
        print(f"  data.dataset.train_csv={train_path}")
        print(f"  data.dataset.val_csv={val_path}")
        print(f"  data.lrs2_video_dir={dataset_root}")
        print(f"  data.lrs2_audio_dir={dataset_root}")


if __name__ == "__main__":
    main()
