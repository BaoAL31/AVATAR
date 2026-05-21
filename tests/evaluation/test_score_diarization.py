"""Wiring tests for the diarization scorer.

These tests verify metric plumbing end-to-end against pyannote.metrics
using toy RTTMs. They intentionally do NOT depend on the AVA-AVD
download or the AVDiarizer - so they are cheap to run in CI.

Skips cleanly if pyannote.metrics is not installed (the eval harness is
an optional extra).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pyannote_metrics = pytest.importorskip("pyannote.metrics")

from src.evaluation.score_diarization import (  # noqa: E402
    render_markdown,
    score_clip,
    score_directory,
    write_report,
)


CLIP_ID = "toy_clip"


def _write_rttm(path: Path, segments: list[tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for start, end, speaker in segments:
            dur = end - start
            f.write(
                f"SPEAKER {CLIP_ID} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            )


@pytest.fixture()
def ref_rttm(tmp_path: Path) -> Path:
    # Two speakers, no overlap. 20s of total reference speech.
    path = tmp_path / "ref" / f"{CLIP_ID}.rttm"
    _write_rttm(path, [
        (0.0,  5.0,  "spkA"),
        (5.0, 10.0,  "spkB"),
        (10.0, 20.0, "spkA"),
    ])
    return path


def test_perfect_hypothesis_zero_der(tmp_path: Path, ref_rttm: Path) -> None:
    hyp = tmp_path / "hyp" / f"{CLIP_ID}.rttm"
    _write_rttm(hyp, [
        (0.0,  5.0,  "x0"),
        (5.0, 10.0,  "x1"),
        (10.0, 20.0, "x0"),
    ])
    score = score_clip(ref_rttm, hyp, CLIP_ID, collar=0.0)
    assert score.der == pytest.approx(0.0, abs=1e-6)
    assert score.miss == pytest.approx(0.0, abs=1e-6)
    assert score.false_alarm == pytest.approx(0.0, abs=1e-6)
    assert score.confusion == pytest.approx(0.0, abs=1e-6)
    assert score.jer == pytest.approx(0.0, abs=1e-6)
    assert score.total == pytest.approx(20.0, abs=1e-6)


def test_relabelled_hypothesis_zero_der_via_hungarian(tmp_path: Path, ref_rttm: Path) -> None:
    """Swap predicted labels -> optimal mapping recovers them -> DER stays 0."""
    hyp = tmp_path / "hyp" / f"{CLIP_ID}.rttm"
    _write_rttm(hyp, [
        (0.0,  5.0,  "swap0"),
        (5.0, 10.0,  "swap1"),
        (10.0, 20.0, "swap0"),
    ])
    score = score_clip(ref_rttm, hyp, CLIP_ID, collar=0.0)
    assert score.der == pytest.approx(0.0, abs=1e-6)
    assert score.confusion == pytest.approx(0.0, abs=1e-6)


def test_dropped_segment_yields_miss(tmp_path: Path, ref_rttm: Path) -> None:
    hyp = tmp_path / "hyp" / f"{CLIP_ID}.rttm"
    _write_rttm(hyp, [
        (0.0,  5.0,  "x0"),
        # 5-10s missed
        (10.0, 20.0, "x0"),
    ])
    score = score_clip(ref_rttm, hyp, CLIP_ID, collar=0.0)
    assert score.miss == pytest.approx(25.0, rel=1e-3)
    assert score.false_alarm == pytest.approx(0.0, abs=1e-6)


def test_inserted_segment_yields_false_alarm(tmp_path: Path, ref_rttm: Path) -> None:
    hyp = tmp_path / "hyp" / f"{CLIP_ID}.rttm"
    _write_rttm(hyp, [
        (0.0,  5.0,  "x0"),
        (5.0, 10.0,  "x1"),
        (10.0, 20.0, "x0"),
        (20.0, 25.0, "ghost"),  # FA outside reference
    ])
    score = score_clip(ref_rttm, hyp, CLIP_ID, collar=0.0)
    assert score.false_alarm == pytest.approx(25.0, rel=1e-3)
    assert score.miss == pytest.approx(0.0, abs=1e-6)


def test_empty_hyp_is_full_miss(tmp_path: Path, ref_rttm: Path) -> None:
    hyp = tmp_path / "hyp" / f"{CLIP_ID}.rttm"
    hyp.parent.mkdir(parents=True, exist_ok=True)
    hyp.write_text("")
    score = score_clip(ref_rttm, hyp, CLIP_ID, collar=0.0)
    assert score.miss == pytest.approx(100.0, rel=1e-3)
    assert score.false_alarm == pytest.approx(0.0, abs=1e-6)


def test_directory_scoring_macro_micro_and_report(tmp_path: Path) -> None:
    ref_dir = tmp_path / "ref"
    hyp_dir = tmp_path / "hyp"

    # clip A: perfect (20s of speech)
    _write_rttm(ref_dir / "clipA.rttm", [
        (0.0, 10.0, "spkA"),
        (10.0, 20.0, "spkB"),
    ])
    _write_rttm(hyp_dir / "clipA.rttm", [
        (0.0, 10.0, "x0"),
        (10.0, 20.0, "x1"),
    ])
    # clip B: missing second half (10s missed of 20s)
    _write_rttm(ref_dir / "clipB.rttm", [
        (0.0, 10.0, "spkA"),
        (10.0, 20.0, "spkB"),
    ])
    _write_rttm(hyp_dir / "clipB.rttm", [
        (0.0, 10.0, "x0"),
    ])

    report = score_directory(ref_dir, hyp_dir, collar=0.0)
    assert report.n_clips == 2
    # Macro = unweighted mean of clip DERs (0 + 50)/2 = 25
    assert report.macro["der"] == pytest.approx(25.0, abs=1e-3)
    # Micro = duration-weighted; both clips have equal ref duration so == macro
    assert report.micro["der"] == pytest.approx(25.0, abs=1e-3)

    md = render_markdown(report, title="Toy")
    assert "Macro" in md and "Micro" in md
    assert "clipA" in md and "clipB" in md

    md_path, json_path = write_report(report, tmp_path / "out", title="Toy")
    assert md_path.exists() and json_path.exists()
    import json
    payload = json.loads(json_path.read_text())
    assert payload["n_clips"] == 2
    assert {c["clip_id"] for c in payload["per_clip"]} == {"clipA", "clipB"}
