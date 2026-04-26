#!/usr/bin/env python3
"""Extract mouth-crop AVI files directly from HF-hosted MP4s using a USR manifest.

This avoids requiring a local videos root. It resolves each manifest MP4 from HF,
loads matching landmarks from landmarks_root/<rel_mp4>.npy, and writes:
  out_root/<rel_mp4_without_ext>.avi
"""

from __future__ import annotations

import argparse
import re
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


def repo_from_tag(tag: str, repo_prefix: str) -> str:
    m = re.match(r"^lrs2_(\d{2})$", tag)
    if m:
        return f"{repo_prefix}_{m.group(1)}"
    if tag == "lrs2":
        return repo_prefix
    raise ValueError(f"Unsupported tag for LRS2 shard routing: {tag}")


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
            rel_mp4 = parts[1].strip().replace("\\", "/")
            rows.append((tag, rel_mp4))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract mouth crops from HF MP4s + local landmarks")
    ap.add_argument("--manifest", type=Path, required=True)
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
    args = ap.parse_args()

    rows = read_manifest_rows(args.manifest.resolve())
    if args.max_clips > 0:
        rows = rows[: args.max_clips]

    landmarks_root = args.landmarks_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Match extract_mouths.py defaults exactly.
    em_args = SimpleNamespace(
        crop_width=96,
        crop_height=96,
        start_idx=48,
        stop_idx=68,
        window_margin=12,
    )
    reference = np.load(args.mean_face.resolve())

    done = skipped = failed = 0
    for tag, rel_mp4 in tqdm(rows, desc="mouth-crops", unit="clip"):
        rel_path = Path(rel_mp4)
        out_no_ext = out_root / rel_path.with_suffix("")
        out_avi = Path(str(out_no_ext) + ".avi")
        lm_path = landmarks_root / rel_path.with_suffix(".npy")

        if out_avi.exists() and not args.overwrite:
            skipped += 1
            continue
        if not lm_path.is_file():
            failed += 1
            continue

        try:
            repo_id = repo_from_tag(tag, args.repo_prefix)
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

