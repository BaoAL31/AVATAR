"""Split AVA-AVD reference RTTMs into on-screen / off-screen subsets.

The audio-visual diarizer can only emit speakers whose face is visible in
the video. Any ground-truth utterance whose time window overlaps NO face
track is effectively un-attributable for an AV-only system, so scoring
against it inflates DER through miss alone.

This module reads `tracks.pkl` (output of voxconverse `Preprocessor`) for
each clip, builds the union of all face-track time spans, and writes two
sibling RTTM dirs:

    <out>/onscreen/<clip>.rttm   - ref rows whose [start, end] overlaps any
                                   face track for >= `min_overlap_ratio`
                                   of the row's duration.
    <out>/offscreen/<clip>.rttm  - the remaining rows.

The hyp side stays unchanged. Scoring against `onscreen/` measures the
diarizer's quality on the AV-feasible portion of the GT; scoring against
`offscreen/` quantifies the unrecoverable structural miss.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class RefRow:
    file_id: str
    start: float
    duration: float
    speaker: str
    raw: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def _parse_rttm(path: Path) -> List[RefRow]:
    rows: List[RefRow] = []
    if not path.exists():
        return rows
    for raw in path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        try:
            start = float(parts[3])
            dur = float(parts[4])
        except ValueError:
            continue
        rows.append(RefRow(
            file_id=parts[1],
            start=start,
            duration=dur,
            speaker=parts[7],
            raw=raw,
        ))
    return rows


def _track_intervals(tracks_pkl: Path, fps: float = 25.0) -> List[Tuple[float, float]]:
    """Read tracks.pkl, return [(start_sec, end_sec), ...] for every track."""
    if not tracks_pkl.exists():
        return []
    with open(tracks_pkl, "rb") as f:
        tracks = pickle.load(f)
    intervals: List[Tuple[float, float]] = []
    for tr in tracks:
        try:
            frames = tr["track"]["frame"]
            if len(frames) == 0:
                continue
            s = float(frames[0]) / fps
            e = float(frames[-1] + 1) / fps
            intervals.append((s, e))
        except (KeyError, TypeError, IndexError):
            continue
    return intervals


def _union(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: List[Tuple[float, float]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _row_overlap(row: RefRow, union: List[Tuple[float, float]]) -> float:
    total = 0.0
    for iv in union:
        total += _overlap((row.start, row.end), iv)
    return total


def split_clip(
    ref_path: Path,
    tracks_pkl: Path,
    onscreen_path: Path,
    offscreen_path: Path,
    min_overlap_ratio: float = 0.5,
) -> Tuple[int, int]:
    """Split ref rows into on/off-screen. Returns (n_on, n_off)."""
    rows = _parse_rttm(ref_path)
    union = _union(_track_intervals(tracks_pkl))

    on_rows: List[RefRow] = []
    off_rows: List[RefRow] = []
    for r in rows:
        if r.duration <= 0:
            continue
        overlap = _row_overlap(r, union)
        if overlap / r.duration >= min_overlap_ratio:
            on_rows.append(r)
        else:
            off_rows.append(r)

    onscreen_path.parent.mkdir(parents=True, exist_ok=True)
    offscreen_path.parent.mkdir(parents=True, exist_ok=True)
    onscreen_path.write_text("\n".join(r.raw for r in on_rows) + ("\n" if on_rows else ""))
    offscreen_path.write_text("\n".join(r.raw for r in off_rows) + ("\n" if off_rows else ""))
    return len(on_rows), len(off_rows)


def split_directory(
    ref_dir: Path,
    work_dir: Path,
    out_root: Path,
    clip_ids: List[str] | None = None,
    min_overlap_ratio: float = 0.5,
) -> None:
    """For each clip with a tracks.pkl in <work_dir>/<clip>/cache, split its
    ref RTTM into onscreen/offscreen sibling files under <out_root>."""
    on_dir = out_root / "onscreen"
    off_dir = out_root / "offscreen"
    on_dir.mkdir(parents=True, exist_ok=True)
    off_dir.mkdir(parents=True, exist_ok=True)

    if clip_ids is None:
        clip_ids = sorted(p.stem for p in ref_dir.glob("*.rttm"))

    total_on = total_off = 0
    for clip_id in clip_ids:
        ref_path = ref_dir / f"{clip_id}.rttm"
        tracks_pkl = work_dir / clip_id / "cache" / "tracks.pkl"
        on_path = on_dir / f"{clip_id}.rttm"
        off_path = off_dir / f"{clip_id}.rttm"

        if not tracks_pkl.exists():
            print(f"[filter_onscreen] WARN: missing tracks.pkl for {clip_id} - "
                  f"falling back to empty face-track set (everything off-screen)")
        n_on, n_off = split_clip(ref_path, tracks_pkl, on_path, off_path,
                                 min_overlap_ratio=min_overlap_ratio)
        total_on += n_on
        total_off += n_off
        print(f"[filter_onscreen] {clip_id}: on={n_on}  off={n_off}  "
              f"(ratio>={min_overlap_ratio})")

    print(f"[filter_onscreen] total: on={total_on}  off={total_off}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-dir", required=True, help="reference RTTM dir (full GT)")
    p.add_argument("--work-dir", required=True,
                   help="Pipeline.run() work dir (must contain <clip>/cache/tracks.pkl)")
    p.add_argument("--out", required=True,
                   help="output root; writes <out>/onscreen/ and <out>/offscreen/")
    p.add_argument("--clip", action="append", default=None,
                   help="restrict to specific clip id (repeatable)")
    p.add_argument("--min-overlap-ratio", type=float, default=0.5,
                   help="minimum fraction of a ref row that must overlap a face track "
                        "for the row to be classified on-screen (default 0.5)")
    args = p.parse_args(argv)

    split_directory(
        ref_dir=Path(args.ref_dir),
        work_dir=Path(args.work_dir),
        out_root=Path(args.out),
        clip_ids=args.clip,
        min_overlap_ratio=args.min_overlap_ratio,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
