#!/usr/bin/env python3
"""Generate per-frame facial landmarks from a USR manifest.

Writes one .npy per clip under out_root, preserving manifest relative paths:
  rel/path/clip.mp4 -> <out_root>/rel/path/clip.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm
import torch

import face_alignment

import sys

USR_ROOT = Path("/home/hoangbng/AVATAR/AVATAR/models/usr")
sys.path.insert(0, str(USR_ROOT))
from utils.hf_media import hub_local_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate landmarks from manifest + Hub media.")
    ap.add_argument("--manifest", type=Path, default=None, help="USR manifest CSV")
    ap.add_argument(
        "--repo-prefix",
        type=str,
        default="HBaoAL/LRS2",
        help="Hub repo prefix, e.g. HBaoAL/LRS2 (shards become _00, _01...)",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/hoangbng/Data/usr/landmarks"),
        help="Output root for .npy files",
    )
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-clips", type=int, default=0, help="0 means no limit")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for landmark inference")
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local HF cache (skip remote checks/downloads)",
    )
    ap.add_argument(
        "--from-repos",
        action="store_true",
        help="Ignore manifest and iterate all mp4 files in HF shard repos.",
    )
    ap.add_argument(
        "--shards",
        type=str,
        default="00-14",
        help="Shard range for --from-repos, e.g. 00-14 or 00,01,03",
    )
    ap.add_argument(
        "--stems-file",
        type=Path,
        default=None,
        help="Optional suffix<TAB>stem list (e.g. hf_npz_stems.txt). If set, this is used as the clip source.",
    )
    return ap.parse_args()


def repo_from_tag(tag: str, repo_prefix: str) -> str:
    m = re.match(r"^lrs2_(\d{2})$", tag)
    if m:
        return f"{repo_prefix}_{m.group(1)}"
    if tag == "lrs2":
        return repo_prefix
    raise ValueError(f"Unsupported tag for LRS2 manifest: {tag}")


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 3)
            if len(parts) < 4:
                continue
            tag, rel_mp4 = parts[0].strip(), parts[1].strip().replace("\\", "/")
            rows.append((tag, rel_mp4))
    return rows


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


def iter_repo_mp4_rows(repo_prefix: str, shards: Iterable[str]) -> list[tuple[str, str]]:
    from huggingface_hub import list_repo_files

    rows: list[tuple[str, str]] = []
    for shard in shards:
        tag = f"lrs2_{shard}"
        repo_id = f"{repo_prefix}_{shard}"
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        for f in files:
            if f.endswith(".mp4") and "/" in f:
                rows.append((tag, f.replace("\\", "/")))
    return rows


def read_stems_rows(path: Path) -> list[tuple[str, str]]:
    """Read suffix<TAB>stem entries and return (tag, rel_mp4)."""
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or "\t" not in s:
                continue
            suffix, stem = s.split("\t", 1)
            suffix = suffix.strip().zfill(2)
            stem = stem.strip()
            if not stem:
                continue
            rows.append((f"lrs2_{suffix}", f"{stem}/{stem}.mp4"))
    return rows


def fill_missing(landmarks: list[np.ndarray | None]) -> np.ndarray:
    # Fill missing detections by nearest-neighbor propagation.
    first_valid = next((x for x in landmarks if x is not None), None)
    if first_valid is None:
        return np.zeros((len(landmarks), 68, 2), dtype=np.float32)

    out: list[np.ndarray] = []
    prev = first_valid
    for lm in landmarks:
        if lm is None:
            out.append(prev)
        else:
            out.append(lm)
            prev = lm

    # Backward pass for leading missing frames.
    nxt = out[-1]
    for i in range(len(out) - 1, -1, -1):
        if landmarks[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]

    return np.stack(out, axis=0).astype(np.float32)


def extract_landmarks(
    fa: face_alignment.FaceAlignment, mp4_path: Path, batch_size: int = 16
) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    frames_rgb: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames_rgb:
        return np.zeros((0, 68, 2), dtype=np.float32)

    frames_lm: list[np.ndarray | None] = []
    # Batch inference is much faster than per-frame calls.
    for i in range(0, len(frames_rgb), batch_size):
        chunk = frames_rgb[i : i + batch_size]
        arr = np.stack(chunk, axis=0)  # [B,H,W,C], uint8
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # [B,C,H,W]
        if fa.device != "cpu":
            t = t.to(fa.device, non_blocking=True)
        preds_batch = fa.get_landmarks_from_batch(t)
        for preds in preds_batch:
            if preds is None or len(preds) == 0:
                frames_lm.append(None)
            else:
                # API returns list/array per face; choose first face.
                lm = preds[0] if isinstance(preds, list) else preds
                frames_lm.append(np.asarray(lm, dtype=np.float32))

    return fill_missing(frames_lm)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.local_files_only:
        import os

        os.environ["HF_MEDIA_LOCAL_ONLY"] = "1"

    torch.set_grad_enabled(False)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.stems_file is not None:
        rows = read_stems_rows(args.stems_file.resolve())
    elif args.from_repos:
        rows = iter_repo_mp4_rows(args.repo_prefix, parse_shards(args.shards))
    else:
        if args.manifest is None:
            raise ValueError("--manifest is required unless --from-repos is set")
        rows = read_manifest(args.manifest.resolve())
    if args.max_clips > 0:
        rows = rows[: args.max_clips]

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=args.device,
    )

    n_done = n_skip = n_fail = 0
    for tag, rel_mp4 in tqdm(rows, desc="landmarks", unit="clip"):
        out_path = args.out_root / Path(rel_mp4).with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            repo_id = repo_from_tag(tag, args.repo_prefix)
            local_mp4 = Path(
                hub_local_path(
                    repo_id=repo_id,
                    repo_filename=rel_mp4,
                    repo_type="dataset",
                )
            )
            arr = extract_landmarks(fa, local_mp4, batch_size=args.batch_size)
            np.save(out_path, arr)
            n_done += 1
        except Exception:
            n_fail += 1

    print(
        f"Done. generated={n_done} skipped_existing={n_skip} failed={n_fail} out_root={args.out_root}"
    )


if __name__ == "__main__":
    main()
