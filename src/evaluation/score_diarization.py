"""Score hypothesis RTTMs against reference RTTMs with pyannote.metrics.

Two metric variants are computed per clip so the paper can report an
overlap breakdown:

    * ``DER_full``   - DiarizationErrorRate(collar, skip_overlap=False)
    * ``DER_noov``   - DiarizationErrorRate(collar, skip_overlap=True)
    * ``overlap_DER = DER_full - DER_noov``

`DER_full` decomposes into ``miss%``, ``false_alarm%``, ``confusion%``
(all normalized by total reference speech). ``confusion%`` is the
speaker-attribution error after Hungarian optimal speaker mapping - this
is what the paper draft calls the "speaker attribution error".

JER is reported alongside as a complementary, speaker-balanced score.

Both per-clip rows and corpus aggregates (macro = unweighted mean of
clips; micro = duration-weighted, taken from ``metric.report()``) are
returned and written as markdown + json.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


@dataclass
class ClipScore:
    clip_id: str
    der: float
    der_noov: float
    overlap_der: float
    jer: float
    miss: float
    false_alarm: float
    confusion: float
    total: float  # reference speech in seconds


@dataclass
class EvalReport:
    collar: float
    n_clips: int
    per_clip: List[ClipScore]
    macro: Dict[str, float]
    micro: Dict[str, float]
    skipped: List[str] = field(default_factory=list)


def _load_rttm(path: Path, uri: str):
    """Load a single-URI RTTM, robust to URI mismatches.

    pyannote.database.util.load_rttm keys its return dict by the RTTM's
    `<file-id>` column. Our writers force that column to `<clip_id>` but
    we still defensively pick the first annotation if the URI doesn't
    match.
    """
    from pyannote.core import Annotation
    from pyannote.database.util import load_rttm

    if not path.exists() or path.stat().st_size == 0:
        return Annotation(uri=uri)

    rttms = load_rttm(str(path))
    if not rttms:
        return Annotation(uri=uri)
    if uri in rttms:
        ann = rttms[uri]
    else:
        ann = next(iter(rttms.values()))
    ann.uri = uri
    return ann


def _ref_to_uem(ref, pad: float = 0.0):
    """Build a UEM Timeline from the union of ref segments.

    Anything in `hyp` outside this UEM is **ignored** (not counted as FA).
    Useful when scoring against a subset reference (e.g. on-screen-only):
    without a UEM, hyp emissions in the excluded off-screen regions would
    inflate FA artificially.

    `pad` widens each ref segment by `pad` seconds on both sides before
    union, so the UEM is at least as wide as the collar applied by
    pyannote's DER. Small pad avoids edge-effect FA right at boundaries.
    """
    from pyannote.core import Segment, Timeline
    segs = []
    for seg, _track, _label in ref.itertracks(yield_label=True):
        s = max(0.0, seg.start - pad)
        e = seg.end + pad
        if e > s:
            segs.append(Segment(s, e))
    return Timeline(segments=segs).support()


def score_clip(ref_path: Path, hyp_path: Path, clip_id: str, collar: float,
               use_ref_uem: bool = False) -> ClipScore:
    from pyannote.metrics.diarization import (
        DiarizationErrorRate,
        JaccardErrorRate,
    )

    ref = _load_rttm(ref_path, clip_id)
    hyp = _load_rttm(hyp_path, clip_id)

    der_full = DiarizationErrorRate(collar=collar, skip_overlap=False)
    der_noov = DiarizationErrorRate(collar=collar, skip_overlap=True)
    jer = JaccardErrorRate(collar=collar)

    uem_kwargs: dict = {}
    if use_ref_uem:
        uem_kwargs["uem"] = _ref_to_uem(ref, pad=collar)

    full = der_full(ref, hyp, detailed=True, **uem_kwargs)
    noov = der_noov(ref, hyp, detailed=True, **uem_kwargs)
    jer_val = float(jer(ref, hyp, **uem_kwargs))

    total = float(full.get("total", 0.0)) or 0.0
    def _pct(numerator_key: str) -> float:
        if total <= 0:
            return 0.0
        return 100.0 * float(full.get(numerator_key, 0.0)) / total

    der_val = 100.0 * float(full.get("diarization error rate", 0.0))
    der_noov_val = 100.0 * float(noov.get("diarization error rate", 0.0))

    return ClipScore(
        clip_id=clip_id,
        der=der_val,
        der_noov=der_noov_val,
        overlap_der=der_val - der_noov_val,
        jer=100.0 * jer_val,
        miss=_pct("missed detection"),
        false_alarm=_pct("false alarm"),
        confusion=_pct("confusion"),
        total=total,
    )


def _macro(per_clip: List[ClipScore]) -> Dict[str, float]:
    if not per_clip:
        return {k: 0.0 for k in ("der", "der_noov", "overlap_der", "jer", "miss", "false_alarm", "confusion")}
    return {
        "der":         mean(c.der for c in per_clip),
        "der_noov":    mean(c.der_noov for c in per_clip),
        "overlap_der": mean(c.overlap_der for c in per_clip),
        "jer":         mean(c.jer for c in per_clip),
        "miss":        mean(c.miss for c in per_clip),
        "false_alarm": mean(c.false_alarm for c in per_clip),
        "confusion":   mean(c.confusion for c in per_clip),
    }


def _micro(per_clip: List[ClipScore]) -> Dict[str, float]:
    """Duration-weighted aggregate over reference speech."""
    total = sum(c.total for c in per_clip)
    if total <= 0:
        return _macro(per_clip)  # fall back to unweighted
    def _w(attr: str) -> float:
        return sum(getattr(c, attr) * c.total for c in per_clip) / total
    return {
        "der":         _w("der"),
        "der_noov":    _w("der_noov"),
        "overlap_der": _w("overlap_der"),
        "jer":         _w("jer"),
        "miss":        _w("miss"),
        "false_alarm": _w("false_alarm"),
        "confusion":   _w("confusion"),
    }


def score_directory(
    ref_dir: str | os.PathLike,
    hyp_dir: str | os.PathLike,
    clip_ids: Optional[List[str]] = None,
    collar: float = 0.25,
    use_ref_uem: bool = False,
) -> EvalReport:
    """Score every clip with a reference RTTM in ``ref_dir``."""
    ref_dir = Path(ref_dir)
    hyp_dir = Path(hyp_dir)

    if clip_ids is None:
        clip_ids = sorted(p.stem for p in ref_dir.glob("*.rttm"))

    per_clip: List[ClipScore] = []
    skipped: List[str] = []
    for clip_id in clip_ids:
        ref_path = ref_dir / f"{clip_id}.rttm"
        hyp_path = hyp_dir / f"{clip_id}.rttm"
        if not ref_path.exists():
            skipped.append(f"{clip_id}:no-ref")
            continue
        try:
            per_clip.append(score_clip(ref_path, hyp_path, clip_id, collar,
                                        use_ref_uem=use_ref_uem))
        except Exception as e:
            skipped.append(f"{clip_id}:{type(e).__name__}:{e}")

    return EvalReport(
        collar=collar,
        n_clips=len(per_clip),
        per_clip=per_clip,
        macro=_macro(per_clip),
        micro=_micro(per_clip),
        skipped=skipped,
    )


# ------------------------------------------------------------------ output

_HEADER = (
    "| Video | DER (%) | JER (%) | Miss (%) | FA (%) | "
    "Confusion / Attr Err (%) | Overlap DER (%) | Ref dur (s) |"
)
_SEP = (
    "| ----- | ------- | ------- | -------- | ------ | "
    "----------------------- | --------------- | ----------- |"
)


def _row(label: str, c: ClipScore | Dict[str, float], total: float | None = None) -> str:
    if isinstance(c, ClipScore):
        return (
            f"| {c.clip_id} | {c.der:.2f} | {c.jer:.2f} | {c.miss:.2f} | "
            f"{c.false_alarm:.2f} | {c.confusion:.2f} | {c.overlap_der:.2f} | "
            f"{c.total:.1f} |"
        )
    dur = "" if total is None else f"{total:.1f}"
    return (
        f"| {label} | {c['der']:.2f} | {c['jer']:.2f} | {c['miss']:.2f} | "
        f"{c['false_alarm']:.2f} | {c['confusion']:.2f} | {c['overlap_der']:.2f} | "
        f"{dur} |"
    )


def render_markdown(report: EvalReport, title: str = "Diarization Evaluation") -> str:
    total = sum(c.total for c in report.per_clip)
    lines = [
        f"# {title}",
        "",
        f"Dataset: AVA-AVD test split | clips scored: {report.n_clips} | "
        f"collar: {report.collar:.2f}s",
        "",
        _HEADER,
        _SEP,
    ]
    for c in report.per_clip:
        lines.append(_row("", c))
    lines.append(_row("**Macro**", report.macro))
    lines.append(_row("**Micro**", report.micro, total=total))
    if report.skipped:
        lines += ["", "## Skipped clips", ""] + [f"- {s}" for s in report.skipped]
    lines += [
        "",
        "## Column definitions",
        "",
        "- **DER** - Diarization Error Rate (collar applied, overlap regions included).",
        "- **JER** - Jaccard Error Rate (per-speaker, unweighted across speakers).",
        "- **Miss** - reference speech not covered by any hypothesis (% of ref).",
        "- **FA** - hypothesis speech outside any reference segment (% of ref).",
        "- **Confusion / Attr Err** - reference time mapped to a wrong speaker label "
        "after Hungarian optimal mapping (% of ref). This is the speaker-attribution "
        "error reported in the paper.",
        "- **Overlap DER** - `DER - DER(skip_overlap=True)`. Approximates the share "
        "of error contributed by overlapped-speech regions.",
        "- **Macro** - unweighted mean across clips.",
        "- **Micro** - mean weighted by reference speech duration.",
    ]
    return "\n".join(lines) + "\n"


def write_report(
    report: EvalReport,
    out_dir: str | os.PathLike,
    title: str = "Diarization Evaluation",
    md_name: str = "diarization_results.md",
    json_name: str = "diarization_results.json",
) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / md_name
    json_path = out_dir / json_name

    md_path.write_text(render_markdown(report, title=title))
    json_path.write_text(json.dumps({
        "collar": report.collar,
        "n_clips": report.n_clips,
        "per_clip": [asdict(c) for c in report.per_clip],
        "macro": report.macro,
        "micro": report.micro,
        "skipped": report.skipped,
    }, indent=2))
    return md_path, json_path
