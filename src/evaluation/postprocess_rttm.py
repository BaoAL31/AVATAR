"""RTTM post-processing utilities for cleaning voxconverse diarizer output.

The voxconverse `result.rttm` is produced as the *union* of two emit paths:

1. Per-face-track ASD rows (one row per visible face per utterance).
2. VAD-cluster rows that re-emit speech via cosine similarity against the
   speaker embedding bank, falling back to label `"unknown"` when no cluster
   is above `spk_thres`.

Both paths write into the same RTTM, producing massive stacked overlap and
~50-100% of reference speech duration tagged as `"unknown"`. Without
post-processing the file reports DER >> 100% (false alarm dominated).

This module implements two minimally invasive cleanups:

- `drop_unknown`: discard rows with speaker label `"unknown"` (or any label
  in the configurable drop-list).
- `merge_same_speaker`: collapse overlapping/adjacent rows that share a
  speaker label into a single non-overlapping segment.

Both run independently and are idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class RTTMRow:
    file_id: str
    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_line(self) -> str:
        return (
            f"SPEAKER {self.file_id} 1 {self.start:.3f} "
            f"{self.duration:.3f} <NA> <NA> {self.speaker} <NA> <NA>"
        )


def parse_rttm(path: Path) -> List[RTTMRow]:
    rows: List[RTTMRow] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                dur = float(parts[4])
            except ValueError:
                continue
            if dur <= 0:
                continue
            rows.append(RTTMRow(parts[1], start, dur, parts[7]))
    return rows


def drop_unknown(rows: Sequence[RTTMRow], drop: Iterable[str] = ("unknown",)) -> List[RTTMRow]:
    drop_set = {d.lower() for d in drop}
    return [r for r in rows if r.speaker.lower() not in drop_set]


def merge_same_speaker(rows: Sequence[RTTMRow]) -> List[RTTMRow]:
    """Collapse overlapping/adjacent rows that share a speaker label.

    Different-speaker overlaps are preserved (they represent true crosstalk).
    """
    by_spk: dict[str, list[RTTMRow]] = {}
    for r in rows:
        by_spk.setdefault(r.speaker, []).append(r)

    merged: List[RTTMRow] = []
    for spk, items in by_spk.items():
        items.sort(key=lambda r: r.start)
        cur = None
        for r in items:
            if cur is None:
                cur = RTTMRow(r.file_id, r.start, r.duration, spk)
                continue
            if r.start <= cur.end:
                new_end = max(cur.end, r.end)
                cur.duration = new_end - cur.start
            else:
                merged.append(cur)
                cur = RTTMRow(r.file_id, r.start, r.duration, spk)
        if cur is not None:
            merged.append(cur)
    merged.sort(key=lambda r: r.start)
    return merged


def single_speaker_per_frame(rows: Sequence[RTTMRow]) -> List[RTTMRow]:
    """Collapse overlapping different-speaker rows to a single non-overlapping
    timeline. At each time slice the winning speaker is the one with the
    longest total speech in this clip (a proxy for ``trustworthy'' identity).
    Use cases: kill ASD-per-track stacked overlap. Cost: loses ability to
    score genuine multi-speaker overlap regions.
    """
    if not rows:
        return []
    spk_total: dict[str, float] = {}
    for r in rows:
        spk_total[r.speaker] = spk_total.get(r.speaker, 0.0) + r.duration

    boundaries = sorted({r.start for r in rows} | {r.end for r in rows})
    out: List[RTTMRow] = []
    file_id = rows[0].file_id

    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        if t1 - t0 < 1e-6:
            continue
        active = [r.speaker for r in rows if r.start <= t0 and r.end >= t1]
        if not active:
            continue
        winner = max(active, key=lambda s: spk_total.get(s, 0.0))
        if out and out[-1].speaker == winner and abs(out[-1].end - t0) < 1e-6:
            out[-1].duration = t1 - out[-1].start
        else:
            out.append(RTTMRow(file_id, t0, t1 - t0, winner))
    return out


def postprocess_file(
    src: Path,
    dst: Path,
    drop_speakers: Iterable[str] = ("unknown",),
    single_speaker: bool = False,
) -> int:
    rows = parse_rttm(src)
    rows = drop_unknown(rows, drop=drop_speakers)
    if single_speaker:
        rows = single_speaker_per_frame(rows)
    else:
        rows = merge_same_speaker(rows)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for r in rows:
            f.write(r.to_line() + "\n")
    return len(rows)


def postprocess_dir(
    src_dir: Path,
    dst_dir: Path,
    drop_speakers: Iterable[str] = ("unknown",),
    single_speaker: bool = False,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src in sorted(src_dir.glob("*.rttm")):
        counts[src.stem] = postprocess_file(
            src, dst_dir / src.name, drop_speakers, single_speaker=single_speaker,
        )
    return counts


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="input RTTM dir")
    p.add_argument("--dst", required=True, help="output RTTM dir")
    p.add_argument("--drop", default="unknown",
                   help="comma-separated speaker labels to drop")
    p.add_argument("--single-speaker", action="store_true",
                   help="collapse to single speaker per time frame (kills overlap detection)")
    args = p.parse_args()
    drop = [s.strip() for s in args.drop.split(",") if s.strip()]
    counts = postprocess_dir(
        Path(args.src), Path(args.dst),
        drop_speakers=drop, single_speaker=args.single_speaker,
    )
    total = sum(counts.values())
    print(f"[postprocess_rttm] wrote {len(counts)} clips, {total} total rows -> {args.dst}")
