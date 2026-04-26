#!/usr/bin/env python3
"""Extract mouth-crop AVIs from HF repos using a stems file (no manifest required).

Stems file format (same as hf_npz_stems.txt):
  <shard_suffix>\t<stem>
Example:
  00\t0000218
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

USR_ROOT = Path("/home/hoangbng/AVATAR/AVATAR/models/usr")
sys.path.insert(0, str(USR_ROOT))
from utils.hf_media import hub_local_path  # noqa: E402

sys.path.insert(0, str(USR_ROOT / "preprocessing"))
from extract_mouths import crop_patch, get_video_clip, save_video_lossless  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract mouth crops from HF repos + local landmarks (no manifest).")
    ap.add_argument(
        "--stems-file",
        type=Path,
        default=Path("/home/hoangbng/AVATAR/AVATAR/data/hf_npz_stems.txt"),
        help="suffix<TAB>stem file.",
    )
    ap.add_argument("--repo-prefix", type=str, default="HBaoAL/LRS2")
    ap.add_argument("--landmarks-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument(
        "--mean-face",
        type=Path,
        default=USR_ROOT / "preprocessing" / "20words_mean_face.npy",
    )
    ap.add_argument("--max-clips", type=int, default=0, help="0 means all")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_stems(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "\t" in s:
                suf, stem = s.split("\t", 1)
                rows.append((suf.strip().zfill(2), stem.strip()))
    return rows


def main() -> int:
    args = parse_args()
    stems = read_stems(args.stems_file.resolve())
    if args.max_clips > 0:
        stems = stems[: args.max_clips]

    landmarks_root = args.landmarks_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    em_args = SimpleNamespace(
        crop_width=96,
        crop_height=96,
        start_idx=48,
        stop_idx=68,
        window_margin=12,
    )
    reference = np.load(args.mean_face.resolve())

    done = skipped = failed = 0
    for suffix, stem in tqdm(stems, desc="mouth-crops", unit="clip"):
        rel_mp4 = f"{stem}/{stem}.mp4"
        rel_no_ext = Path(stem) / stem
        out_no_ext = out_root / rel_no_ext
        out_avi = Path(str(out_no_ext) + ".avi")
        lm_path = landmarks_root / f"{stem}/{stem}.npy"

        if out_avi.exists() and not args.overwrite:
            skipped += 1
            continue
        if not lm_path.is_file():
            failed += 1
            continue

        try:
            repo_id = f"{args.repo_prefix}_{suffix}"
            mp4_local = hub_local_path(repo_id, rel_mp4, repo_type="dataset")
            video = get_video_clip(mp4_local)
            landmarks = np.load(lm_path)
            seq = crop_patch(video, landmarks, reference, em_args)
            out_no_ext.parent.mkdir(parents=True, exist_ok=True)
            save_video_lossless(str(out_no_ext), seq, 25)
            done += 1
        except Exception:
            failed += 1

    print(f"Done. generated={done} skipped_existing={skipped} failed={failed} out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

