"""Loop the AVATAR `Pipeline` over a directory of prepared eval clips.

For each clip in ``clips_dir``:
    - run `src.pipeline.Pipeline` with `output_dir=<work_dir>/<clip_id>`,
    - copy the diarizer's `cache/result.rttm` to `<hyp_dir>/<clip_id>.rttm`,
    - rewrite the RTTM `<file-id>` field to equal `<clip_id>` so it matches
      the reference URI in scoring.

Failures (no face, ffmpeg, etc.) are isolated per clip: a stub empty RTTM is
written instead of crashing the sweep. An empty RTTM is scored as 100% miss
for that clip, which is the desired behaviour for a held-out benchmark.
"""

from __future__ import annotations

import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# `Pipeline` pulls in the full ML stack (cv2, torch, voxconverse). Import
# lazily inside `run_clip` so utility helpers like `_rewrite_uri` are usable
# from lighter envs (e.g. the scoring venv that only has pyannote.metrics).


@dataclass
class ClipRunResult:
    clip_id: str
    hyp_rttm: Path
    ok: bool
    error: Optional[str] = None


def _rewrite_uri(rttm_src: Path, rttm_dst: Path, uri: str) -> None:
    """Copy `rttm_src` -> `rttm_dst` and force column 1 (<file-id>) to `uri`.

    Drops any lines that aren't valid SPEAKER rows. Truncated/corrupt trailing
    lines in `cache/result.rttm` (we have seen `... <NA> <NA>\\n` in repo
    fixtures) are skipped instead of failing the whole eval.
    """
    rttm_dst.parent.mkdir(parents=True, exist_ok=True)
    if not rttm_src.exists():
        rttm_dst.write_text("")
        return
    with open(rttm_src) as fin, open(rttm_dst, "w") as fout:
        for raw in fin:
            parts = raw.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            parts[1] = uri
            try:
                float(parts[3])
                float(parts[4])
            except ValueError:
                continue
            fout.write(" ".join(parts) + "\n")


def run_clip(
    clip_id: str,
    clip_video: Path,
    work_dir: Path,
    hyp_dir: Path,
    visualize: bool = False,
) -> ClipRunResult:
    clip_work = work_dir / clip_id
    hyp_rttm = hyp_dir / f"{clip_id}.rttm"
    hyp_dir.mkdir(parents=True, exist_ok=True)

    from src.pipeline import Pipeline  # heavy: cv2/torch/voxconverse

    pipe_err: Optional[Exception] = None
    try:
        pipe = Pipeline(
            video_path=str(clip_video),
            output_dir=str(clip_work),
            visualize=visualize,
        )
        pipe.run()
    except Exception as e:
        traceback.print_exc()
        pipe_err = e

    # The diarizer writes `cache/result.rttm` BEFORE mouth crop / USR run,
    # so a late-stage failure (e.g. USR OOM) must not discard those segments.
    # Always try to salvage result.rttm if it exists on disk.
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


def run_all(
    clips_dir: str | os.PathLike,
    work_dir: str | os.PathLike,
    hyp_dir: str | os.PathLike,
    clip_ids: Optional[List[str]] = None,
    skip_existing: bool = True,
    visualize: bool = False,
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
            print(f"[run_eval_pipeline] missing clip video: {clip_video}")
            hyp_rttm.parent.mkdir(parents=True, exist_ok=True)
            hyp_rttm.write_text("")
            results.append(ClipRunResult(clip_id, hyp_rttm, ok=False, error="missing video"))
            continue

        print(f"[run_eval_pipeline] {clip_id}")
        results.append(run_clip(clip_id, clip_video, work_dir, hyp_dir, visualize))

    n_ok = sum(1 for r in results if r.ok)
    print(f"[run_eval_pipeline] {n_ok}/{len(results)} clips succeeded")
    return results
