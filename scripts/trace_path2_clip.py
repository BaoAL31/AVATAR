#!/usr/bin/env python3
"""Run full AV diarization once and print Path 2 (audio-residual) diagnostics.

Computes overlap between reference RTTM speech frames and:
  global audio VAD, per-track subtraction residual, Path 1 coverage / winner,
  Path 2 skip reasons (covered vs cosine-unknown), and Path 2 emissions.

Example:
  python scripts/trace_path2_clip.py \\
    --video /path/to/1j20qq1JyX4_c_01.mp4 \\
    --ref-rttm data/eval/rttms_clip/1j20qq1JyX4_c_01.rttm \\
    --work-dir /tmp/path2_trace_1j20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_AVDIAR = REPO_ROOT / "models" / "av-diarization"
if str(_AVDIAR) not in sys.path:
    sys.path.insert(0, str(_AVDIAR))


def _parse_rttm_segments(path: Path) -> list[tuple[float, float]]:
    segs: list[tuple[float, float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            segs.append((start, start + dur))
    return segs


def _safe_default_device():
    import torch

    if os.environ.get("AVATAR_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True, help="Input clip .mp4")
    ap.add_argument(
        "--ref-rttm",
        type=Path,
        default=None,
        help="Reference RTTM (SPEAKER lines); default data/eval/rttms_clip/<stem>.rttm",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Output/cache directory (same layout as Diarizer wrapper)",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="If set, write path2_diag JSON here",
    )
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"error: missing video: {args.video}", file=sys.stderr)
        return 1

    stem = args.video.stem
    ref = args.ref_rttm
    if ref is None:
        ref = REPO_ROOT / "data" / "eval" / "rttms_clip" / f"{stem}.rttm"
    if not ref.is_file():
        print(f"warning: no ref RTTM at {ref}; ref_* fractions will be absent", file=sys.stderr)
        ref_segs = None
    else:
        ref_segs = _parse_rttm_segments(ref)

    work = args.work_dir.resolve()
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    from voxconverse.avdiarizer import AVDiarizer  # noqa: E402

    ns = Namespace(
        input=str(args.video),
        out_dir=str(cache),
        cache_dir=str(cache),
        ckpt_dir=None,
        visualize=False,
        vad="silero",
        speaker_model="ecapa-tdnn",
    )

    path2_diag: dict = {}
    extras = {
        "path2_diag": path2_diag,
        "ref_segments_sec": ref_segs,
    }

    device = _safe_default_device()
    pipe = AVDiarizer(ns)
    pipe.run(
        str(args.video),
        str(cache),
        device,
        str(cache),
        False,
        diarizer_extras=extras,
    )

    print(json.dumps(path2_diag, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(path2_diag, f, indent=2)
        print(f"wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
