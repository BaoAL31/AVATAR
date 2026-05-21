"""Render a demo MP4 from a finished `Pipeline.run()` work dir.

Consumes per-clip artifacts that already exist on disk:

    <work_dir>/cache/pyframes/*.jpg         per-frame source images (25 fps)
    <work_dir>/cache/pyavi/audio.wav        16 kHz mono audio
    <work_dir>/cache/tracks.pkl             face track boxes per frame
    <work_dir>/cache/faceidx.pkl            track_idx -> face_cluster_id
    <work_dir>/cache/result.rttm            speaker diarization (used for time->speaker)
    <work_dir>/transcript.srt               speaker-attributed transcription

Output: an MP4 with:
  - Face bounding boxes colored by speaker (face_cluster_id), labeled `SPK_N`.
  - Caption strip burned into the bottom of the frame, showing the active
    transcription line(s) at the current timestamp.
  - Original audio muxed in.

Existing `voxconverse.visualize.Visualizer` only labels face track index +
face identity (no speaker color, no captions). This module fills the gap.

Run with:

    python -m src.evaluation.render_demo \
        --work-dir data/eval/work/<clip_id> \
        --out reports/eval/demo_<clip_id>.mp4 \
        [--max-seconds 300]
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_FPS = 25
_PALETTE = [
    (255, 100, 100),   # red
    (100, 200, 255),   # light blue
    (100, 255, 150),   # green
    (255, 200, 100),   # orange
    (200, 100, 255),   # purple
    (255, 255, 100),   # yellow
    (100, 255, 255),   # cyan
    (255, 150, 200),   # pink
    (180, 180, 180),   # gray (fallback)
]


def _color_for(spk_id: int) -> Tuple[int, int, int]:
    if spk_id < 0:
        return _PALETTE[-1]
    return _PALETTE[spk_id % (len(_PALETTE) - 1)]


@dataclass
class FaceBox:
    track: int
    speaker: int
    x: float
    y: float
    s: float


@dataclass
class Caption:
    start: float
    end: float
    speaker: str
    text: str


def _word_set(text: str) -> set:
    return {w.lower() for w in re.findall(r"[A-Za-z']+", text) if len(w) >= 2}


def _dedup_overlapping(captions: List[Caption], min_jaccard: float = 0.4) -> List[Caption]:
    """Collapse captions that overlap in time AND share text content.

    Pipeline-level cause: voxconverse emits one ASD row per visible face per
    utterance, and `_run_usr` lipreads each mouth crop independently. Silent
    bystanders end up with garbled transcripts derived from the same audio
    window as the real speaker. After parsing the SRT we group captions by
    time overlap, drop entries whose word-Jaccard against a higher-scoring
    sibling exceeds `min_jaccard`, and keep the longest-text survivor.
    """
    if not captions:
        return captions
    sorted_caps = sorted(captions, key=lambda c: (c.start, -len(c.text)))
    keep: List[Caption] = []
    dropped = 0
    for c in sorted_caps:
        c_words = _word_set(c.text)
        merged = False
        for i, k in enumerate(keep):
            if c.start >= k.end or c.end <= k.start:
                continue
            k_words = _word_set(k.text)
            union = c_words | k_words
            if not union:
                continue
            jac = len(c_words & k_words) / len(union)
            if jac >= min_jaccard:
                if len(c.text) > len(k.text):
                    keep[i] = c
                dropped += 1
                merged = True
                break
        if not merged:
            keep.append(c)
    keep.sort(key=lambda c: c.start)
    print(f"[render_demo] dedup: kept {len(keep)} / {len(captions)} captions "
          f"(dropped {dropped} duplicates, jaccard>={min_jaccard})")
    return keep


def _parse_srt(path: Path) -> List[Caption]:
    """SRT parser: only needs start/end timestamps + body. Body is
    `<speaker>: <text>`."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    caps: List[Caption] = []
    for blk in blocks:
        lines = [ln for ln in blk.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        m = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            lines[1],
        )
        if not m:
            continue
        h1, mi1, s1, ms1, h2, mi2, s2, ms2 = (int(x) for x in m.groups())
        start = h1 * 3600 + mi1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + mi2 * 60 + s2 + ms2 / 1000
        body = " ".join(lines[2:])
        if ":" in body:
            speaker, _, body_text = body.partition(":")
            speaker = speaker.strip()
            body_text = body_text.strip()
        else:
            speaker, body_text = "", body
        caps.append(Caption(start=start, end=end, speaker=speaker, text=body_text))
    return caps


def _build_face_index(
    tracks: List[Dict],
    faceidx: List[int],
    num_frames: int,
) -> List[List[FaceBox]]:
    """For each output frame, list of face boxes labeled with speaker id (=
    face_cluster_id from faceidx)."""
    faces: List[List[FaceBox]] = [[] for _ in range(num_frames)]
    for tidx, track in enumerate(tracks):
        try:
            speaker = int(faceidx[tidx]) if tidx < len(faceidx) else -1
            frames = track["track"]["frame"].tolist()
            xs = track["proc_track"]["x"]
            ys = track["proc_track"]["y"]
            ss = track["proc_track"]["s"]
            for i, fr in enumerate(frames):
                if 0 <= fr < num_frames:
                    faces[fr].append(FaceBox(
                        track=tidx,
                        speaker=speaker,
                        x=float(xs[i]),
                        y=float(ys[i]),
                        s=float(ss[i]),
                    ))
        except (IndexError, KeyError, ValueError):
            continue
    return faces


def _active_captions(captions: List[Caption], t: float) -> List[Caption]:
    return [c for c in captions if c.start <= t <= c.end]


def _wrap_text(text: str, max_chars: int = 60) -> List[str]:
    """Naive char-budget wrap: split into lines so each <= max_chars."""
    words = text.split()
    out: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur = cur + " " + w
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def _write_srt_window(captions: List[Caption], path: Path, max_seconds: Optional[float]) -> None:
    """Write a clipped SRT next to the demo MP4. Caption rows whose start
    >= max_seconds are dropped; rows that straddle max_seconds are truncated.
    Timestamps remain in source-clip time so a player can sync against the
    rendered video (which also starts at t=0)."""
    def _fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        if ms == 1000:
            s += 1
            ms = 0
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    keep: List[Caption] = []
    for c in captions:
        if max_seconds is not None:
            if c.start >= max_seconds:
                continue
            end = min(c.end, max_seconds)
        else:
            end = c.end
        keep.append(Caption(start=c.start, end=end, speaker=c.speaker, text=c.text))

    with path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(keep, start=1):
            speaker = f"{c.speaker}: " if c.speaker else ""
            f.write(f"{i}\n")
            f.write(f"{_fmt(c.start)} --> {_fmt(c.end)}\n")
            f.write(f"{speaker}{c.text}\n\n")


def render(
    work_dir: Path,
    out_path: Path,
    max_seconds: Optional[float] = None,
    max_width: int = 1280,
    captions_mode: str = "sidecar",  # one of: sidecar | burn | none
    dedup_captions: bool = True,
    dedup_jaccard: float = 0.4,
) -> Path:
    import cv2
    import numpy as np

    cache_dir = work_dir / "cache"
    frames_dir = cache_dir / "pyframes"
    audio_path = cache_dir / "pyavi" / "audio.wav"
    tracks_path = cache_dir / "tracks.pkl"
    faceidx_path = cache_dir / "faceidx.pkl"
    srt_path = work_dir / "transcript.srt"

    if not frames_dir.exists():
        raise FileNotFoundError(f"missing frames dir: {frames_dir}")

    flist = sorted(frames_dir.glob("*.jpg"))
    if max_seconds is not None:
        flist = flist[: int(max_seconds * _FPS)]
    if not flist:
        raise RuntimeError(f"no frames in {frames_dir}")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)
    with open(faceidx_path, "rb") as f:
        faceidx = pickle.load(f)
    captions = _parse_srt(srt_path)
    print(f"[render_demo] {len(flist)} frames, {len(tracks)} tracks, "
          f"{len(captions)} captions, audio={audio_path.exists()}")
    if dedup_captions:
        captions = _dedup_overlapping(captions, min_jaccard=dedup_jaccard)

    faces_per_frame = _build_face_index(tracks, faceidx, num_frames=len(flist))

    first = cv2.imread(str(flist[0]))
    if first is None:
        raise RuntimeError(f"failed to read {flist[0]}")
    fh0, fw0 = first.shape[:2]
    scale = min(1.0, max_width / fw0)
    fw, fh = int(fw0 * scale), int(fh0 * scale)
    burn = (captions_mode == "burn")
    caption_strip_h = max(60, int(fh * 0.12)) if burn else 0
    out_h = fh + caption_strip_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vonly = out_path.with_suffix(".vonly.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vonly), fourcc, _FPS, (fw, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 VideoWriter failed for {vonly}")

    for fidx, fp in enumerate(flist):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if scale != 1.0:
            img = cv2.resize(img, (fw, fh))

        for face in faces_per_frame[fidx]:
            x1 = int((face.x - face.s) * scale)
            y1 = int((face.y - face.s) * scale)
            x2 = int((face.x + face.s) * scale)
            y2 = int((face.y + face.s) * scale)
            color = _color_for(face.speaker)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"SPK_{face.speaker}" if face.speaker >= 0 else "SPK_?"
            cv2.putText(img, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        if burn:
            strip = np.zeros((caption_strip_h, fw, 3), dtype=np.uint8)
            canvas = np.vstack([img, strip])
            t = fidx / _FPS
            active = _active_captions(captions, t)
            if active:
                lines: List[str] = []
                for c in active[:2]:
                    head = f"[{c.speaker}] " if c.speaker else ""
                    wrapped = _wrap_text(head + c.text, max_chars=80)
                    lines.extend(wrapped)
                line_h = 26
                y = fh + 28
                for ln in lines[:3]:
                    cv2.putText(canvas, ln, (16, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    y += line_h
        else:
            canvas = img

        writer.write(canvas)

        if fidx % (5 * _FPS) == 0:
            print(f"[render_demo] {fidx}/{len(flist)} frames "
                  f"({100 * fidx / len(flist):.1f}%)")

    writer.release()

    if audio_path.exists():
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(vonly),
            "-i", str(audio_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
        ]
        if max_seconds is not None:
            cmd += ["-t", str(max_seconds)]
        cmd.append(str(out_path))
        print(f"[render_demo] muxing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        vonly.unlink(missing_ok=True)
    else:
        vonly.rename(out_path)
    print(f"[render_demo] wrote {out_path}")

    if captions_mode == "sidecar":
        sidecar = out_path.with_suffix(".srt")
        _write_srt_window(captions, sidecar, max_seconds)
        print(f"[render_demo] wrote {sidecar} ({len(captions)} src captions)")

    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True,
                   help="Pipeline.run() output directory for one clip")
    p.add_argument("--out", required=True, help="output MP4 path")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="truncate to N seconds (None = full clip)")
    p.add_argument("--max-width", type=int, default=1280,
                   help="downscale frames to this width if larger")
    p.add_argument("--captions", choices=("sidecar", "burn", "none"), default="sidecar",
                   help="how to handle captions: sidecar SRT next to MP4 (default), "
                        "burn into caption strip, or skip entirely")
    p.add_argument("--no-dedup-captions", dest="dedup_captions", action="store_false",
                   help="disable overlap dedup pass on parsed SRT")
    p.add_argument("--dedup-jaccard", type=float, default=0.4,
                   help="word-Jaccard threshold for caption dedup (default 0.4)")
    p.set_defaults(dedup_captions=True)
    args = p.parse_args(argv)

    render(
        work_dir=Path(args.work_dir),
        out_path=Path(args.out),
        max_seconds=args.max_seconds,
        max_width=args.max_width,
        captions_mode=args.captions,
        dedup_captions=args.dedup_captions,
        dedup_jaccard=args.dedup_jaccard,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
