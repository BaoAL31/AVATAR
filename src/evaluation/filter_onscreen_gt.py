"""Split AVA-AVD reference RTTMs into onscreen / offscreen using AVA-AVD
ground-truth face tracks (NOT our pipeline's predicted tracks.pkl).

For each clip we parse:
    <avaavd_root>/rttms/<clip_id>.rttm                   - absolute-time ref
    <avaavd_root>/tracks/<video_id>-activespeaker.csv    - per-frame bboxes
                                                            labeled by spkid

The activespeaker CSV uses spkids of the form `<clip_suffix><spkXX>` where
`<clip_suffix>` is `01`, `02`, `03` matching `_c_01`, `_c_02`, `_c_03`.

A ref row (start, dur, spk) is classified ONSCREEN iff the bbox annotations
for THAT SAME speaker cover >= `min_overlap_ratio` of the row's duration.
Speakers that never appear in the CSV (e.g. off-screen narrators) end up
entirely in the offscreen subset, which is the structural floor an AV-only
diarizer can never address.

Output RTTMs use clip-local timestamps (matching AVATAR's rttms_clip), so
they can be scored directly against our hyp RTTMs.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SPKID_RE = re.compile(r"^(\d{2})(spk\d+)$")


@dataclass
class RefRow:
    file_id: str
    abs_start: float
    duration: float
    speaker: str

    @property
    def abs_end(self) -> float:
        return self.abs_start + self.duration


def _parse_abs_rttm(path: Path) -> List[RefRow]:
    """Parse upstream AVA-AVD rttm with absolute video timestamps."""
    rows: List[RefRow] = []
    if not path.exists():
        return rows
    for raw in path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        try:
            rows.append(RefRow(
                file_id=parts[1],
                abs_start=float(parts[3]),
                duration=float(parts[4]),
                speaker=parts[7],
            ))
        except ValueError:
            continue
    return rows


def _parse_tracks_csv(csv_path: Path, clip_suffix: str
                     ) -> Dict[str, List[Tuple[float, float]]]:
    """Parse AVA-AVD activespeaker.csv -> {spk_label: [(t_start, t_end), ...]}
    where t is absolute video time and intervals are merged across consecutive
    bbox samples (gap <= 0.5s)."""

    if not csv_path.exists():
        return {}

    per_spk_times: Dict[str, List[float]] = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 9:
                continue
            spkid = row[8]
            m = SPKID_RE.match(spkid)
            if not m:
                continue
            suffix, spk = m.group(1), m.group(2)
            if suffix != clip_suffix:
                continue
            try:
                t = float(row[1])
            except ValueError:
                continue
            per_spk_times[spk].append(t)

    intervals: Dict[str, List[Tuple[float, float]]] = {}
    for spk, times in per_spk_times.items():
        times.sort()
        if not times:
            continue
        merged: List[Tuple[float, float]] = []
        cur_s = cur_e = times[0]
        for t in times[1:]:
            if t - cur_e <= 0.5:
                cur_e = t
            else:
                merged.append((cur_s, cur_e))
                cur_s = cur_e = t
        merged.append((cur_s, cur_e))
        intervals[spk] = merged
    return intervals


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _row_overlap(row: RefRow, spk_intervals: List[Tuple[float, float]]) -> float:
    total = 0.0
    for iv in spk_intervals:
        total += _overlap((row.abs_start, row.abs_end), iv)
    return total


def _clip_offset(abs_rows: List[RefRow], local_path: Path) -> float:
    """Recover the clip start offset by aligning the first ref row's absolute
    timestamp against the first row in the clip-local RTTM."""
    if not local_path.exists() or not abs_rows:
        return 0.0
    local_first: float | None = None
    for raw in local_path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        try:
            local_first = float(parts[3])
            local_dur = float(parts[4])
            local_spk = parts[7]
        except ValueError:
            continue
        for r in abs_rows:
            if r.speaker == local_spk and abs(r.duration - local_dur) < 1e-3:
                return r.abs_start - local_first
        break
    return 0.0


def split_clip(
    clip_id: str,
    avaavd_root: Path,
    local_ref_path: Path,
    onscreen_path: Path,
    offscreen_path: Path,
    min_overlap_ratio: float = 0.5,
) -> Tuple[int, int, Dict[str, Tuple[float, float]]]:
    """Returns (n_on, n_off, per_spk_summary)."""
    abs_ref_path = avaavd_root / "rttms" / f"{clip_id}.rttm"
    abs_rows = _parse_abs_rttm(abs_ref_path)
    if not abs_rows:
        print(f"[filter_onscreen_gt] WARN: empty abs ref for {clip_id}", file=sys.stderr)

    m = re.match(r"(.+)_c_(\d{2})$", clip_id)
    if not m:
        raise ValueError(f"unrecognised clip id: {clip_id}")
    video_id, clip_suffix = m.group(1), m.group(2)

    csv_path = avaavd_root / "tracks" / f"{video_id}-activespeaker.csv"
    spk_intervals = _parse_tracks_csv(csv_path, clip_suffix)

    offset = _clip_offset(abs_rows, local_ref_path)

    on_rows: List[Tuple[float, float, str]] = []
    off_rows: List[Tuple[float, float, str]] = []
    per_spk: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

    for r in abs_rows:
        if r.duration <= 0:
            continue
        ivals = spk_intervals.get(r.speaker, [])
        ov = _row_overlap(r, ivals)
        local_start = r.abs_start - offset
        record = (local_start, r.duration, r.speaker)
        on_d, off_d = per_spk[r.speaker]
        if ivals and ov / r.duration >= min_overlap_ratio:
            on_rows.append(record)
            per_spk[r.speaker] = (on_d + r.duration, off_d)
        else:
            off_rows.append(record)
            per_spk[r.speaker] = (on_d, off_d + r.duration)

    def _write(path: Path, rows: List[Tuple[float, float, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"SPEAKER {clip_id} 1 {s:.3f} {d:.3f} <NA> <NA> {spk} <NA> <NA>"
            for s, d, spk in rows
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))

    _write(onscreen_path, on_rows)
    _write(offscreen_path, off_rows)
    return len(on_rows), len(off_rows), dict(per_spk)


def split_directory(
    avaavd_root: Path,
    local_ref_dir: Path,
    out_root: Path,
    clip_ids: List[str] | None = None,
    min_overlap_ratio: float = 0.5,
) -> None:
    on_dir = out_root / "onscreen"
    off_dir = out_root / "offscreen"
    on_dir.mkdir(parents=True, exist_ok=True)
    off_dir.mkdir(parents=True, exist_ok=True)

    if clip_ids is None:
        clip_ids = sorted(p.stem for p in local_ref_dir.glob("*.rttm"))

    total_on = total_off = 0
    for clip_id in clip_ids:
        local_ref = local_ref_dir / f"{clip_id}.rttm"
        n_on, n_off, per_spk = split_clip(
            clip_id=clip_id,
            avaavd_root=avaavd_root,
            local_ref_path=local_ref,
            onscreen_path=on_dir / f"{clip_id}.rttm",
            offscreen_path=off_dir / f"{clip_id}.rttm",
            min_overlap_ratio=min_overlap_ratio,
        )
        total_on += n_on
        total_off += n_off
        on_dur = sum(o for o, _ in per_spk.values())
        off_dur = sum(off for _, off in per_spk.values())
        unseen = [spk for spk, (o, off) in per_spk.items() if o == 0 and off > 0]
        print(f"[filter_onscreen_gt] {clip_id}: on={n_on} ({on_dur:.1f}s) "
              f"off={n_off} ({off_dur:.1f}s)  never-on-screen-speakers="
              f"{sorted(unseen)}")

    print(f"[filter_onscreen_gt] total: on={total_on}  off={total_off}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--avaavd-root", required=True,
                   help="AVA-AVD dataset root (contains rttms/ and tracks/)")
    p.add_argument("--local-ref-dir", required=True,
                   help="AVATAR clip-local ref RTTM dir (rttms_clip/)")
    p.add_argument("--out", required=True,
                   help="output root; writes <out>/onscreen/ and <out>/offscreen/")
    p.add_argument("--clip", action="append", default=None,
                   help="restrict to specific clip id (repeatable)")
    p.add_argument("--min-overlap-ratio", type=float, default=0.5,
                   help="minimum fraction of a ref row that must overlap the "
                        "speaker's own bbox annotations (default 0.5)")
    args = p.parse_args(argv)

    split_directory(
        avaavd_root=Path(args.avaavd_root),
        local_ref_dir=Path(args.local_ref_dir),
        out_root=Path(args.out),
        clip_ids=args.clip,
        min_overlap_ratio=args.min_overlap_ratio,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
