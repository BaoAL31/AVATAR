#!/usr/bin/env python3
"""Preflight checks for landmark + mouth-crop readiness.

Validates sampled rows from a USR manifest against local preprocessing outputs:
  - landmarks_root/<rel_video>.npy exists and looks valid (T,68,2)
  - mouth_root/<rel_video>.avi exists
  - mouth AVI is 96x96 and ~25 fps
  - optional frame-count consistency checks
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Preflight validate mouth-crop preprocessing outputs.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/hoangbng/AVATAR/AVATAR/data/train_manifest.csv"),
    )
    ap.add_argument(
        "--landmarks-root",
        type=Path,
        default=Path("/home/hoangbng/Data/usr/landmarks"),
    )
    ap.add_argument(
        "--mouth-root",
        type=Path,
        default=Path("/home/hoangbng/Data/usr/mouth_crops"),
    )
    ap.add_argument("--sample-count", type=int, default=100, help="0 means all rows.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expect-width", type=int, default=96)
    ap.add_argument("--expect-height", type=int, default=96)
    ap.add_argument("--expect-fps", type=float, default=25.0)
    ap.add_argument(
        "--fps-tol",
        type=float,
        default=0.25,
        help="Absolute tolerance for fps check.",
    )
    ap.add_argument(
        "--strict-frame-match",
        action="store_true",
        help="Require exact landmark_len == avi_frame_count.",
    )
    ap.add_argument("--show-limit", type=int, default=20)
    return ap.parse_args()


def read_manifest_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split(",", 3)
            if len(parts) != 4:
                continue
            tag, rel_video, frame_count_s, _label_ids = parts
            try:
                frame_count = int(frame_count_s.strip())
            except ValueError:
                frame_count = -1
            rows.append((ln, tag.strip(), rel_video.strip().replace("\\", "/"), frame_count))
    return rows


def check_avi(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, None, "cannot_open"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, nframes


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    lm_root = args.landmarks_root.resolve()
    mouth_root = args.mouth_root.resolve()

    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}")
        return 1
    if not lm_root.is_dir():
        print(f"ERROR: landmarks root not found: {lm_root}")
        return 1
    if not mouth_root.is_dir():
        print(f"ERROR: mouth root not found: {mouth_root}")
        return 1

    rows = read_manifest_rows(manifest)
    if not rows:
        print(f"ERROR: no parseable rows in {manifest}")
        return 1

    rng = random.Random(args.seed)
    if args.sample_count > 0 and len(rows) > args.sample_count:
        rows = rng.sample(rows, args.sample_count)

    problems = []
    ok = 0
    for ln, _tag, rel_video, manifest_frames in rows:
        rel = Path(rel_video)
        npy = lm_root / rel.with_suffix(".npy")
        avi = mouth_root / rel.with_suffix(".avi")

        if not npy.is_file():
            problems.append(f"L{ln} missing landmarks: {npy}")
            continue
        if not avi.is_file():
            problems.append(f"L{ln} missing mouth crop: {avi}")
            continue

        try:
            arr = np.load(npy, mmap_mode="r")
        except Exception as e:
            problems.append(f"L{ln} bad landmarks load: {npy} ({e})")
            continue

        if arr.ndim != 3 or arr.shape[1:] != (68, 2):
            problems.append(f"L{ln} landmarks shape {tuple(arr.shape)} expected (T,68,2)")
            continue
        if arr.shape[0] <= 0:
            problems.append(f"L{ln} landmarks empty: {npy}")
            continue

        w, h, fps, nframes = check_avi(avi)
        if nframes == "cannot_open":
            problems.append(f"L{ln} cannot open avi: {avi}")
            continue
        if w != args.expect_width or h != args.expect_height:
            problems.append(
                f"L{ln} bad avi size {w}x{h}, expected {args.expect_width}x{args.expect_height}: {avi}"
            )
            continue
        if abs(float(fps) - args.expect_fps) > args.fps_tol:
            problems.append(
                f"L{ln} bad fps {fps:.3f}, expected {args.expect_fps}±{args.fps_tol}: {avi}"
            )
            continue
        if nframes <= 0:
            problems.append(f"L{ln} non-positive avi frame count: {avi}")
            continue

        lm_t = int(arr.shape[0])
        if args.strict_frame_match:
            if lm_t != nframes:
                problems.append(f"L{ln} landmark T={lm_t} != avi frames={nframes}: {rel_video}")
                continue
        else:
            if abs(lm_t - nframes) > 2:
                problems.append(
                    f"L{ln} landmark/avi frame drift too large (T={lm_t}, avi={nframes}): {rel_video}"
                )
                continue
        if manifest_frames > 0 and abs(manifest_frames - nframes) > 2:
            problems.append(
                f"L{ln} manifest/avi frame drift too large (manifest={manifest_frames}, avi={nframes}): {rel_video}"
            )
            continue

        ok += 1

    print(f"Manifest: {manifest}")
    print(f"Sampled rows checked: {len(rows)}")
    print(f"Passed: {ok}")
    print(f"Failed: {len(problems)}")
    if problems:
        print("\nExamples:")
        for p in problems[: args.show_limit]:
            print(f"- {p}")
        if len(problems) > args.show_limit:
            print(f"... and {len(problems) - args.show_limit} more")
        return 1
    print("PASS: preprocessing outputs are consistent for sampled rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

