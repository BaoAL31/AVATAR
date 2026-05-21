"""Diarizer-only re-run for fast eval iteration.

`run_eval_pipeline.run_clip` invokes the full AVATAR `Pipeline` (diarize +
mouth-crop + USR). When tuning VAD / clustering parameters we only need a
fresh `result.rttm` per clip - the rest is irrelevant for DER/JER scoring and
USR alone dominates wall time.

This module re-runs **only** `src.diarization.run_av_diarization.Diarizer` for
each clip, salvages `cache/result.rttm`, and rewrites the URI column - same
output contract as `run_eval_pipeline` but ~2x faster per clip.

Existing per-clip `cache/` (frames, tracks.pkl, faceidx.pkl) is wiped before
the re-run so the new VAD / clustering settings actually take effect.
"""

from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path
from typing import List, Optional

from src.evaluation.run_eval_pipeline import ClipRunResult, _rewrite_uri


def _safe_default_device():
    """Match Pipeline._safe_default_device(): honor AVATAR_FORCE_CPU + catch
    partial-CUDA double-frees on WSL."""
    import torch
    if os.environ.get("AVATAR_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


def run_clip_diar_only(
    clip_id: str,
    clip_video: Path,
    work_dir: Path,
    hyp_dir: Path,
    visualize: bool = False,
    reset_cache: bool = True,
) -> ClipRunResult:
    clip_work = work_dir / clip_id
    hyp_rttm = hyp_dir / f"{clip_id}.rttm"
    hyp_dir.mkdir(parents=True, exist_ok=True)

    if reset_cache and clip_work.exists():
        shutil.rmtree(clip_work)
    clip_work.mkdir(parents=True, exist_ok=True)

    from src.diarization.run_av_diarization import Diarizer

    pipe_err: Optional[Exception] = None
    try:
        device = _safe_default_device()
        diar = Diarizer(str(clip_video), str(clip_work), device)
        diar.run(visualize=visualize)
    except Exception as e:
        traceback.print_exc()
        pipe_err = e

    src_rttm = clip_work / "cache" / "result.rttm"
    if src_rttm.exists() and src_rttm.stat().st_size > 0:
        _rewrite_uri(src_rttm, hyp_rttm, clip_id)
        return ClipRunResult(
            clip_id=clip_id,
            hyp_rttm=hyp_rttm,
            ok=pipe_err is None,
            error=None if pipe_err is None else f"{type(pipe_err).__name__}: {pipe_err}",
        )

    hyp_rttm.write_text("")
    return ClipRunResult(
        clip_id=clip_id,
        hyp_rttm=hyp_rttm,
        ok=False,
        error=(f"{type(pipe_err).__name__}: {pipe_err}" if pipe_err
               else "no result.rttm written by diarizer"),
    )


def run_all_diar_only(
    clips_dir: str | os.PathLike,
    work_dir: str | os.PathLike,
    hyp_dir: str | os.PathLike,
    clip_ids: Optional[List[str]] = None,
    skip_existing: bool = False,
    visualize: bool = False,
    reset_cache: bool = True,
) -> List[ClipRunResult]:
    clips_dir = Path(clips_dir)
    work_dir = Path(work_dir)
    hyp_dir = Path(hyp_dir)

    if clip_ids is None:
        clip_ids = sorted(p.stem for p in clips_dir.glob("*.mp4"))

    results: List[ClipRunResult] = []
    for clip_id in clip_ids:
        clip_video = clips_dir / f"{clip_id}.mp4"
        hyp_rttm = hyp_dir / f"{clip_id}.rttm"

        if skip_existing and hyp_rttm.exists() and hyp_rttm.stat().st_size > 0:
            results.append(ClipRunResult(clip_id, hyp_rttm, ok=True))
            continue
        if not clip_video.exists():
            print(f"[rerun_diarize_only] missing clip video: {clip_video}")
            hyp_rttm.parent.mkdir(parents=True, exist_ok=True)
            hyp_rttm.write_text("")
            results.append(ClipRunResult(clip_id, hyp_rttm, ok=False, error="missing video"))
            continue

        print(f"[rerun_diarize_only] {clip_id}")
        results.append(run_clip_diar_only(clip_id, clip_video, work_dir, hyp_dir, visualize, reset_cache))

    n_ok = sum(1 for r in results if r.ok)
    print(f"[rerun_diarize_only] {n_ok}/{len(results)} clips succeeded")
    return results
