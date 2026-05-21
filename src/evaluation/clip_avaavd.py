"""AVA-AVD test split preparation.

Given an AVA-AVD download root (containing ``videos/``, ``rttms/`` and
``split/`` directories), this module:

1. Resolves the parent YouTube video file for each clip id in
   ``split/test.list``.
2. Computes the clip time window ``[mins, maxs]`` from the union of segments
   in ``rttms/<clip_id>.rttm`` (AVA-AVD's own preprocessing convention - see
   ``dataset/scripts/preprocessing.py:split_waves``).
3. Cuts the parent video with ffmpeg into ``clips/<clip_id>.mp4`` so the
   resulting file has 0-relative time.
4. Emits a shifted-and-renamed RTTM under ``rttms_clip/<clip_id>.rttm`` whose
   timestamps are clip-relative and whose ``<file-id>`` field equals
   ``<clip_id>`` (required for pyannote.metrics URI matching).

The downloader entrypoint defers to the upstream
``dataset/scripts/download.py`` script when a clone of
``zcxu-eric/AVA-AVD`` is available; otherwise it prints clear instructions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


@dataclass
class ClipSpec:
    clip_id: str
    parent_video: Path
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def read_test_list(split_path: Path) -> List[str]:
    with open(split_path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def parent_video_id(clip_id: str) -> str:
    """``<ytid>_c_NN`` -> ``<ytid>``."""
    if "_c_" not in clip_id:
        return clip_id
    return clip_id.rsplit("_c_", 1)[0]


def find_parent_video(videos_dir: Path, ytid: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        candidate = videos_dir / f"{ytid}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(videos_dir.glob(f"{ytid}.*"))
    return matches[0] if matches else None


def parse_rttm_bounds(rttm_path: Path) -> Tuple[float, float]:
    """Mirror AVA-AVD's mins/maxs computation across all rows."""
    mins = float("inf")
    maxs = float("-inf")
    with open(rttm_path) as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 5 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            mins = min(mins, start)
            maxs = max(maxs, start + dur)
    if mins == float("inf") or maxs == float("-inf"):
        raise ValueError(f"No SPEAKER rows in {rttm_path}")
    return mins, maxs


def write_shifted_rttm(src: Path, dst: Path, offset: float, uri: str) -> None:
    """Rewrite RTTM with clip-relative times and ``uri`` as <file-id>."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src) as fin, open(dst, "w") as fout:
        for raw in fin:
            parts = raw.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            parts[1] = uri
            parts[3] = f"{max(0.0, float(parts[3]) - offset):.3f}"
            fout.write(" ".join(parts) + "\n")


def ffmpeg_cut(src: Path, dst: Path, start: float, duration: float) -> None:
    """Re-encode cut. ``-c copy`` is unsafe here - AVA-AVD windows rarely
    align with keyframes and the downstream face detector requires clean
    frames at t=0."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", "16000", "-ac", "1",
        "-loglevel", "error",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def build_clip_specs(
    avaavd_root: Path,
    clip_ids: Iterable[str],
) -> List[ClipSpec]:
    videos_dir = avaavd_root / "videos"
    rttm_dir = avaavd_root / "rttms"
    specs: List[ClipSpec] = []
    missing: List[str] = []
    for clip_id in clip_ids:
        rttm_path = rttm_dir / f"{clip_id}.rttm"
        if not rttm_path.exists():
            missing.append(f"rttm:{clip_id}")
            continue
        parent = find_parent_video(videos_dir, parent_video_id(clip_id))
        if parent is None:
            missing.append(f"video:{parent_video_id(clip_id)}")
            continue
        mins, maxs = parse_rttm_bounds(rttm_path)
        specs.append(ClipSpec(clip_id, parent, mins, maxs))
    if missing:
        print(f"[clip_avaavd] WARNING: missing assets for {len(missing)} clips: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return specs


def prepare_clips(
    avaavd_root: str | os.PathLike,
    out_root: str | os.PathLike,
    split: str = "test",
    overwrite: bool = False,
) -> List[ClipSpec]:
    """Slice clips + write shifted GT RTTMs.

    Returns the list of clips successfully prepared.
    """
    avaavd_root = Path(avaavd_root)
    out_root = Path(out_root)
    clips_dir = out_root / "clips"
    ref_dir = out_root / "rttms_clip"
    clips_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    split_path = avaavd_root / "split" / f"{split}.list"
    clip_ids = read_test_list(split_path)
    specs = build_clip_specs(avaavd_root, clip_ids)

    prepared: List[ClipSpec] = []
    for spec in specs:
        clip_video = clips_dir / f"{spec.clip_id}.mp4"
        clip_rttm = ref_dir / f"{spec.clip_id}.rttm"

        if overwrite or not clip_video.exists():
            try:
                ffmpeg_cut(spec.parent_video, clip_video, spec.start, spec.duration)
            except subprocess.CalledProcessError as e:
                print(f"[clip_avaavd] ffmpeg failed for {spec.clip_id}: {e}")
                continue

        if overwrite or not clip_rttm.exists():
            src_rttm = avaavd_root / "rttms" / f"{spec.clip_id}.rttm"
            write_shifted_rttm(src_rttm, clip_rttm, spec.start, spec.clip_id)

        prepared.append(spec)

    print(f"[clip_avaavd] prepared {len(prepared)}/{len(specs)} clips at {out_root}")
    return prepared


def download(avaavd_root: str | os.PathLike, avaavd_repo: Optional[str] = None) -> None:
    """Best-effort downloader. Defers to upstream ``dataset/scripts/download.py``.

    If ``avaavd_repo`` is given it must point at a local clone of
    https://github.com/zcxu-eric/AVA-AVD - the upstream script downloads
    videos from S3 + annotations from gdrive into ``dataset/``.
    """
    avaavd_root = Path(avaavd_root)
    avaavd_root.mkdir(parents=True, exist_ok=True)
    if avaavd_repo is None:
        print(
            "[clip_avaavd] No AVA-AVD repo clone provided.\n"
            "  Clone https://github.com/zcxu-eric/AVA-AVD and run:\n"
            "    cd AVA-AVD && python dataset/scripts/download.py\n"
            f"  Then move/symlink dataset/ -> {avaavd_root}"
        )
        return
    repo = Path(avaavd_repo)
    script = repo / "dataset" / "scripts" / "download.py"
    if not script.exists():
        raise FileNotFoundError(f"Upstream download script not found at {script}")
    subprocess.run(["python", str(script)], cwd=repo, check=True)
    upstream_dataset = repo / "dataset"
    if upstream_dataset.resolve() != avaavd_root.resolve():
        for sub in ("videos", "rttms", "split", "labs", "tracks"):
            src = upstream_dataset / sub
            dst = avaavd_root / sub
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
