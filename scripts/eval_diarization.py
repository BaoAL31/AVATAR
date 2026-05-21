"""CLI entrypoint for the AVA-AVD diarization evaluation harness.

Usage:
    python scripts/eval_diarization.py \
        --config configs/eval/avaavd.yaml \
        --stage all          # or prep | infer | score

Each stage is re-runnable independently so scoring can iterate without
re-running diarization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import clip_avaavd, score_diarization  # noqa: E402

# `run_eval_pipeline` imports `src.pipeline` which transitively requires
# the heavy ML stack (cv2, torch, voxconverse). Imported lazily inside
# `stage_infer` so that prep/score work in lighter environments.


STAGES = ("prep", "infer", "infer_diar_only", "score", "all")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def stage_prep(cfg: dict) -> list[str]:
    avaavd_root = _resolve(cfg["avaavd_root"])
    out_root = _resolve(cfg["out_root"])
    split = cfg.get("split", "test")

    avaavd_repo = cfg.get("avaavd_repo")
    if not (avaavd_root / "split" / f"{split}.list").exists():
        clip_avaavd.download(avaavd_root, avaavd_repo=avaavd_repo)

    specs = clip_avaavd.prepare_clips(avaavd_root, out_root, split=split)
    return [s.clip_id for s in specs]


def stage_infer(cfg: dict, clip_ids: list[str] | None = None) -> None:
    from src.evaluation import run_eval_pipeline  # heavy import, lazy

    out_root = _resolve(cfg["out_root"])
    clips_dir = out_root / "clips"
    work_dir = _resolve(cfg["work_dir"])
    hyp_dir = _resolve(cfg["hyp_dir"])

    run_eval_pipeline.run_all(
        clips_dir=clips_dir,
        work_dir=work_dir,
        hyp_dir=hyp_dir,
        clip_ids=clip_ids,
        skip_existing=bool(cfg.get("skip_existing", True)),
        visualize=bool(cfg.get("visualize", False)),
    )


def stage_infer_diar_only(cfg: dict, clip_ids: list[str] | None = None) -> None:
    """Re-run only the diarizer (skip mouth-crop + USR) for fast iteration on
    VAD / clustering parameter tuning. Wipes per-clip work cache."""
    from src.evaluation import rerun_diarize_only  # heavy import, lazy

    out_root = _resolve(cfg["out_root"])
    clips_dir = out_root / "clips"
    work_dir = _resolve(cfg["work_dir"])
    hyp_dir = _resolve(cfg["hyp_dir"])

    rerun_diarize_only.run_all_diar_only(
        clips_dir=clips_dir,
        work_dir=work_dir,
        hyp_dir=hyp_dir,
        clip_ids=clip_ids,
        skip_existing=False,  # always re-run; tuning new params
        visualize=bool(cfg.get("visualize", False)),
        reset_cache=True,
    )


def stage_score(cfg: dict, clip_ids: list[str] | None = None) -> None:
    out_root = _resolve(cfg["out_root"])
    ref_dir = out_root / "rttms_clip"
    hyp_dir = _resolve(cfg["hyp_dir"])
    report_dir = _resolve(cfg["report_dir"])
    collar = float(cfg.get("collar", 0.25))
    title = cfg.get("report_title", "Diarization Evaluation")

    report = score_diarization.score_directory(
        ref_dir=ref_dir,
        hyp_dir=hyp_dir,
        clip_ids=clip_ids,
        collar=collar,
    )
    md_path, json_path = score_diarization.write_report(report, report_dir, title=title)
    print(f"[eval] wrote {md_path}")
    print(f"[eval] wrote {json_path}")
    print(f"[eval] macro DER={report.macro['der']:.2f}% JER={report.macro['jer']:.2f}% "
          f"micro DER={report.micro['der']:.2f}% JER={report.micro['jer']:.2f}% "
          f"(n={report.n_clips}, skipped={len(report.skipped)})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to YAML eval config")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--clip", action="append", default=None,
                   help="restrict to specific clip id (repeatable). Default: all in split.")
    args = p.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    clip_ids = args.clip
    if args.stage in ("prep", "all"):
        prepared = stage_prep(cfg)
        if clip_ids is None and args.stage == "all":
            clip_ids = prepared
    if args.stage in ("infer", "all"):
        stage_infer(cfg, clip_ids=clip_ids)
    if args.stage == "infer_diar_only":
        stage_infer_diar_only(cfg, clip_ids=clip_ids)
    if args.stage in ("score", "all"):
        stage_score(cfg, clip_ids=clip_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
