"""LRS2 Fusion eval: USR-per-clip vs pipeline WER comparison.

For each sample in ``data/lrs2_fusion_samples.jsonl`` (built by
``scripts/build_lrs2_fusion_samples.py``) and stitched by
``scripts/concat_lrs2_sample.py`` into ``data/lrs2_fusion/<sample_id>/``:

* **USR-per-clip** (oracle): run USR on each source clip independently,
  concat predictions in clip order, score vs concatenated reference.
* **Pipeline-on-concat** (unit under test): run the full pipeline
  (face detect → track → mouth crop → AV-diarization → USR per segment)
  on the stitched ``input.mp4``, concat segment predictions in start-time
  order, score vs concatenated reference.

Outputs one JSONL row per sample to ``--output`` and a console summary at the
end. Skips already-completed samples on resume unless ``--force``.

Smoke / regression guard: pass ``--abort-pipeline-regression-delta 0.35`` to
stop as soon as pipeline WER exceeds USR-per-clip WER by ≥35 pp while
per-clip WER ≤ ``--abort-pipeline-regression-max-per-clip`` (default 0.40).
Writes ``<sample>/pipeline_regression_abort_debug.json`` and exits with code 2.

Usage:
    LD_LIBRARY_PATH=/home/jembo/miniconda3/envs/usr_env/lib:/usr/lib/wsl/lib \\
    HF_TOKEN=... \\
    /home/jembo/miniconda3/envs/usr_env/bin/python \\
    scripts/eval_lrs2_fusion.py \\
        --samples data/lrs2_fusion_samples.jsonl \\
        --work-root data/lrs2_fusion \\
        --ckpt models/usr/checkpoints/baseplus_high_resource_lrs3vox2.pth \\
        --num-samples 10 \\
        --output data/eval_results.jsonl \\
        --quiet-pipeline \\
        --abort-pipeline-regression-delta 0.35 \\
        --abort-pipeline-regression-max-per-clip 0.40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from huggingface_hub import hf_hub_download
from torchvision.transforms import CenterCrop, Grayscale

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "models" / "usr"))

from data.transforms import NormalizeVideo  # noqa: E402
from espnet.asr.asr_utils import add_results_to_json  # noqa: E402
from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: E402
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E  # noqa: E402
from espnet.nets.scorers.length_bonus import LengthBonus  # noqa: E402
from utils.utils import UNIGRAM1000_LIST  # noqa: E402
from utils.au_npz import AU_FEATURE_DIM, load_au_from_npz  # noqa: E402

FPS = 25
SEAM_VIDEO_FRAMES = 5

_AVDIAR_DIR = _REPO_ROOT / "models" / "av-diarization"
if str(_AVDIAR_DIR) not in sys.path:
    sys.path.insert(0, str(_AVDIAR_DIR))

DEFAULT_SAMPLES = _REPO_ROOT / "data" / "lrs2_fusion_samples.jsonl"
DEFAULT_WORK_ROOT = _REPO_ROOT / "data" / "lrs2_fusion"
DEFAULT_HF_CACHE = _REPO_ROOT / "data" / "hf_cache"
DEFAULT_CKPT = _REPO_ROOT / "models" / "usr" / "checkpoints" / "baseplus_high_resource_lrs3vox2.pth"
DEFAULT_USR_CONF = _REPO_ROOT / "models" / "usr" / "conf"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "eval_results.jsonl"
DEFAULT_REPO_PREFIX = "HBaoAL/LRS2"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    ap.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                    help=f"USR checkpoint. Default: {DEFAULT_CKPT}")
    ap.add_argument("--repo-prefix", type=str, default=DEFAULT_REPO_PREFIX)
    ap.add_argument("--modality", type=str, default="av", choices=["a", "v", "av"])
    ap.add_argument("--num-samples", type=int, default=None,
                    help="Run only this many samples. Default: all.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--force", action="store_true", help="Re-run samples that already have results.")
    ap.add_argument("--quiet-pipeline", action="store_true",
                    help="Redirect pipeline subprocess stdout/stderr to a log file per sample.")
    ap.add_argument("--use-au", action="store_true",
                    help="Feed pre-computed AU features to USR. Required when --ckpt is the LoRA-AU model.")
    ap.add_argument("--au-features-dir", type=Path, default=None,
                    help="Directory holding LRS2 AU .npz files (flat: <dir>/<stem>.npz or nested: <dir>/<stem>/<stem>.npz).")
    ap.add_argument(
        "--abort-pipeline-regression-delta",
        type=float,
        default=None,
        metavar="WER",
        help="Smoke/regression guard: if set, stop immediately after a sample where pipeline WER exceeds "
             "USR-per-clip WER by at least this margin (e.g. 0.35 = 35 pp) and per-clip WER is "
             "≤ --abort-pipeline-regression-max-per-clip. Writes debug JSON beside the sample and exits 2.",
    )
    ap.add_argument(
        "--abort-pipeline-regression-max-per-clip",
        type=float,
        default=0.40,
        dest="abort_pipeline_regression_max_per_clip",
        help="With --abort-pipeline-regression-delta: only trigger when USR-per-clip WER is at most this "
             "(default 0.40), so we do not abort when oracle clip decoding is already poor.",
    )
    ap.add_argument(
        "--abort-pipeline-regression-max-concat",
        type=float,
        default=None,
        dest="abort_pipeline_regression_max_concat_deprecated",
        help=argparse.SUPPRESS,
    )
    return ap.parse_args()


def repo_from_tag(tag: str, prefix: str) -> str:
    m = re.match(r"^lrs2_(\d{2})$", tag)
    if m:
        return f"{prefix}_{m.group(1)}"
    if tag == "lrs2":
        return prefix
    raise ValueError(f"Unsupported tag: {tag}")


def load_samples(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if rows and rows[0].get("_meta"):
        return rows[0], rows[1:]
    return {}, rows


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def normalize_text(s: str) -> str:
    return " ".join(s.strip().upper().split())


def get_wer(pred: str, ref: str) -> float:
    pred_words = normalize_text(pred).split()
    ref_words = normalize_text(ref).split()
    n = len(ref_words)
    if n == 0:
        return 0.0 if not pred_words else 1.0
    m = len(pred_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_words[i - 1] == ref_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / n


def normalize_ckpt(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        return state
    out = {}
    for k, v in state.items():
        nk = k
        for p in ("_orig_mod.", "model.backbone.", "module."):
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def load_video_tensor(path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    x = torch.from_numpy(np.stack(frames)).permute((3, 0, 1, 2)).float() / 255.0
    x = CenterCrop(88)(x)
    x = x.transpose(0, 1)
    x = Grayscale()(x)
    x = x.transpose(0, 1)
    x = NormalizeVideo(mean=(0.421,), std=(0.165,))(x)
    return x


def load_audio_tensor(path: str) -> torch.Tensor:
    wav, _ = torchaudio.load(path, normalize=True)
    return wav


def build_model(ckpt_path: Path, device: torch.device, *, use_au: bool = False):
    """Load USR for in-process ``transcribe_usr`` (oracle rows).

    Subprocess ``run_usr.py`` picks Hydra config from checkpoint kind independently;
    when ``--use-au``, match it here via ``config_ft_lrs2_lora_au_base`` and key-by-key
    ``copy_`` load (same strategy as ``run_usr.Transcriber``) so merged LoRA-AU weights fit.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(DEFAULT_USR_CONF)):
        if use_au:
            cfg = compose(
                config_name="config_ft_lrs2_lora_au_base",
                overrides=["experiment_name=eval"],
            )
            model = E2E(1049, cfg.model.backbone)
            raw = torch.load(str(ckpt_path), map_location=device)
            state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
            if not isinstance(state_dict, dict):
                raise TypeError(f"Unexpected checkpoint type from {ckpt_path}")
            state_dict = {k.replace("_orig_mod.model.backbone.", ""): v for k, v in state_dict.items()}
            state_dict = normalize_ckpt(state_dict)
            model_sd = model.state_dict()
            for k, v in state_dict.items():
                if k in model_sd and v.shape == model_sd[k].shape:
                    model_sd[k].copy_(v)
            model = model.to(device).eval()
            return model, cfg

        config_name = "config" if (DEFAULT_USR_CONF / "config.yaml").exists() else "config_ft_lrs2_lora_base"
        cfg = compose(
            config_name=config_name,
            overrides=["experiment_name=eval", "model/backbone=resnet_transformer_baseplus"],
        )
    model = E2E(1049, cfg.model.backbone)
    state = torch.load(str(ckpt_path), map_location=device)
    state = normalize_ckpt(state)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, cfg


def build_beam(model, cfg) -> BatchBeamSearch:
    token_list = UNIGRAM1000_LIST
    odim = len(token_list)
    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(odim)
    weights = dict(
        decoder=1.0 - cfg.decode.ctc_weight,
        ctc=cfg.decode.ctc_weight,
        lm=cfg.decode.lm_weight,
        length_bonus=cfg.decode.penalty,
    )
    return BatchBeamSearch(
        beam_size=cfg.decode.beam_size,
        vocab_size=odim,
        weights=weights,
        scorers=scorers,
        sos=odim - 1,
        eos=odim - 1,
        token_list=token_list,
        pre_beam_score_key=None if cfg.decode.ctc_weight == 1.0 else "decoder",
    )


def detokenize(text: str) -> str:
    return (
        text.replace("<eos>", "")
        .replace(" ", "")
        .replace("\u2581", " ")
        .strip()
    )


def transcribe_usr(model, beam, cfg, video_path: str, audio_path: str,
                    modality: str, device: torch.device,
                    au: torch.Tensor | None = None) -> str:
    video = load_video_tensor(video_path).to(device)
    audio = load_audio_tensor(audio_path).to(device)
    au_in = None
    if au is not None:
        au_in = au.unsqueeze(0).to(device)  # (1, T, AU_DIM)
    with torch.no_grad():
        if modality == "v":
            feat, _, _ = model.encoder.forward_single(xs_v=video, au=au_in)
        elif modality == "a":
            feat, _, _ = model.encoder.forward_single(xs_a=audio.unsqueeze(0).transpose(1, 2))
        else:
            feat, _, _ = model.encoder.forward_single(
                xs_v=video,
                xs_a=audio.unsqueeze(0).transpose(1, 2),
                au=au_in,
            )
        nbest = beam(
            x=feat.squeeze(0),
            modality=modality,
            maxlenratio=cfg.decode.maxlenratio,
            minlenratio=cfg.decode.minlenratio,
        )
    raw = add_results_to_json([nbest[0].asdict()], UNIGRAM1000_LIST)
    return detokenize(raw)


def _patch_preprocessor_for_short_clips() -> None:
    import voxconverse.preprocessor as pp
    if getattr(pp.Preprocessor, "_avatar_fusion_patched", False):
        return
    old_init = pp.Preprocessor.__init__

    def patched_init(self, cache_dir, ckpt_dir=None, device="cpu", frame_rate=25,
                     crop_scale=0.40, min_track=1, num_failed_det=100,
                     min_face_size=40, facedet_scale=0.35):
        old_init(self, cache_dir, ckpt_dir, device, frame_rate, crop_scale,
                 min_track, num_failed_det, min_face_size, facedet_scale)

    pp.Preprocessor.__init__ = patched_init
    pp.Preprocessor._avatar_fusion_patched = True


def run_pipeline(stitched_mp4: str, work_dir: Path, *, ckpt_path: Path,
                 device: torch.device, log_path: Path | None,
                 au_concat_path: Path | None = None) -> tuple[str, dict]:
    """Returns (concat_text, info). info has n_segments, segments list, error str, etc."""
    _patch_preprocessor_for_short_clips()
    from src.pipeline import Pipeline

    info: dict = {"n_segments": 0, "tracker_failed": False, "error": None, "segments": []}
    try:
        p = Pipeline(
            video_path=stitched_mp4,
            output_dir=str(work_dir),
            device=device,
            ckpt_path=str(ckpt_path),
            au_concat_path=str(au_concat_path) if au_concat_path else None,
        )
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log_f, \
                 _redirect_stdio(log_f):
                results = p.run()
        else:
            results = p.run()
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["tracker_failed"] = True
        return "", info

    if not results:
        info["tracker_failed"] = True
        return "", info

    sorted_results = sorted(results, key=lambda r: float(r.get("start", 0.0)))
    info["n_segments"] = len(sorted_results)
    info["segments"] = [
        {"start": float(r.get("start", 0.0)), "end": float(r.get("end", 0.0)),
         "speaker": str(r.get("speaker", "")), "transcription": str(r.get("transcription", ""))}
        for r in sorted_results
    ]
    concat = " ".join(s["transcription"] for s in info["segments"] if s["transcription"]).strip()
    return concat, info


class _redirect_stdio:
    def __init__(self, target):
        self.target = target
        self._old_out = None
        self._old_err = None

    def __enter__(self):
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = self.target
        sys.stderr = self.target
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._old_out
        sys.stderr = self._old_err


def download_clip(rel: str, tag: str, *, repo_prefix: str, cache_dir: Path,
                  token: str | None) -> tuple[Path, Path, Path | None]:
    repo_id = repo_from_tag(tag, repo_prefix)
    folder = Path(rel).parent.as_posix()
    stem = Path(rel).stem
    mp4 = Path(hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=f"{folder}/{stem}.mp4",
        cache_dir=str(cache_dir), token=token,
    ))
    wav = Path(hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=f"{folder}/{stem}.wav",
        cache_dir=str(cache_dir), token=token,
    ))
    try:
        txt = Path(hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename=f"{folder}/{stem}.txt",
            cache_dir=str(cache_dir), token=token,
        ))
    except Exception:
        txt = None
    return mp4, wav, txt


def resolve_au_npz(au_dir: Path, stem: str) -> Path | None:
    """Find <au_dir>/<stem>.npz or <au_dir>/<stem>/<stem>.npz, else None."""
    flat = au_dir / f"{stem}.npz"
    if flat.is_file():
        return flat
    nested = au_dir / stem / f"{stem}.npz"
    if nested.is_file():
        return nested
    return None


def video_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return max(0, n)


def load_clip_au(au_dir: Path, stem: str, target_len: int) -> torch.Tensor:
    """Per-clip AU loader → (target_len, AU_FEATURE_DIM). Zeros if file missing."""
    p = resolve_au_npz(au_dir, stem)
    if p is None:
        return torch.zeros(target_len, AU_FEATURE_DIM, dtype=torch.float32)
    return load_au_from_npz(str(p), target_len)


def build_concat_au(clip_aus: list[torch.Tensor], total_len: int) -> torch.Tensor:
    """Concat per-clip AU tensors with SEAM_VIDEO_FRAMES zero rows between them.

    Final length is padded or truncated to total_len so it always matches the
    stitched video's frame count exactly (the concat seam plus any rounding
    can drift by ±1 frame relative to ffmpeg's actual output).
    """
    pieces: list[torch.Tensor] = []
    for i, au in enumerate(clip_aus):
        pieces.append(au)
        if i < len(clip_aus) - 1:
            pieces.append(torch.zeros(SEAM_VIDEO_FRAMES, AU_FEATURE_DIM, dtype=torch.float32))
    concat = torch.cat(pieces, dim=0) if pieces else torch.zeros(0, AU_FEATURE_DIM, dtype=torch.float32)
    if concat.size(0) < total_len:
        pad = torch.zeros(total_len - concat.size(0), AU_FEATURE_DIM, dtype=torch.float32)
        concat = torch.cat([concat, pad], dim=0)
    elif concat.size(0) > total_len:
        concat = concat[:total_len]
    return concat


def read_ground_truth(txt_path: Path | None) -> str:
    if txt_path is None or not txt_path.is_file():
        return ""
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.lower().startswith("text:"):
            return s.split(":", 1)[1].strip()
    return ""


def evaluate_sample(sample: dict, *, args: argparse.Namespace, model, beam, cfg,
                    device: torch.device, hf_token: str | None) -> dict:
    sample_id = sample["sample_id"]
    sample_work = args.work_root / sample_id
    stitched_mp4 = sample_work / "input.mp4"
    stitched_wav = sample_work / "input.wav"
    if not stitched_mp4.is_file() or not stitched_wav.is_file():
        raise RuntimeError(
            f"Stitched inputs missing for {sample_id}; expected {stitched_mp4} + {stitched_wav}. "
            f"Run scripts/concat_lrs2_sample.py first."
        )

    au_dir = args.au_features_dir if args.use_au else None
    if args.use_au and au_dir is None:
        raise SystemExit("--use-au requires --au-features-dir")

    clip_gts: list[str] = []
    clip_preds: list[str] = []
    clip_aus: list[torch.Tensor] = []
    for clip in sample["clips"]:
        mp4, wav, txt = download_clip(
            clip["rel"], clip["tag"],
            repo_prefix=args.repo_prefix,
            cache_dir=args.hf_cache,
            token=hf_token,
        )
        clip_gts.append(read_ground_truth(txt))
        clip_au = None
        if au_dir is not None:
            stem = Path(clip["rel"]).stem
            clip_au = load_clip_au(au_dir, stem, clip["frames"])
            clip_aus.append(clip_au)
        clip_preds.append(transcribe_usr(
            model, beam, cfg, str(mp4), str(wav), args.modality, device,
            au=clip_au,
        ))

    ref_concat = " ".join(g for g in clip_gts if g).strip()
    usr_per_clip_pred = " ".join(p for p in clip_preds if p).strip()
    wer_per_clip = get_wer(usr_per_clip_pred, ref_concat)

    au_concat_tensor: torch.Tensor | None = None
    au_concat_path: Path | None = None
    if au_dir is not None:
        stitched_frame_count = video_frame_count(str(stitched_mp4))
        au_concat_tensor = build_concat_au(clip_aus, stitched_frame_count)
        au_concat_path = sample_work / "au_concat.npy"
        np.save(au_concat_path, au_concat_tensor.numpy())

    pipeline_work = sample_work / "pipeline"
    pipeline_work.mkdir(parents=True, exist_ok=True)
    pipeline_log = pipeline_work / "pipeline.log" if args.quiet_pipeline else None
    pipeline_pred, pipe_info = run_pipeline(
        str(stitched_mp4), pipeline_work,
        ckpt_path=args.ckpt, device=device, log_path=pipeline_log,
        au_concat_path=au_concat_path,
    )
    wer_pipeline = get_wer(pipeline_pred, ref_concat) if not pipe_info["tracker_failed"] else None

    return {
        "sample_id": sample_id,
        "ckpt": str(args.ckpt),
        "modality": args.modality,
        "use_au": bool(args.use_au),
        "ref": ref_concat,
        "clip_refs": clip_gts,
        "clip_preds": clip_preds,
        "usr_per_clip": {"pred": usr_per_clip_pred, "wer": wer_per_clip},
        "pipeline_on_concat": {
            "pred": pipeline_pred,
            "wer": wer_pipeline,
            "n_segments": pipe_info["n_segments"],
            "tracker_failed": pipe_info["tracker_failed"],
            "error": pipe_info["error"],
        },
    }


def _rttm_rows_per_track(rttm_path: Path) -> dict[int, int] | None:
    if not rttm_path.is_file():
        return None
    try:
        rows = json.loads(rttm_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    counts: dict[int, int] = {}
    for e in rows:
        try:
            tix = int(e["track_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        counts[tix] = counts.get(tix, 0) + 1
    return counts


def dump_pipeline_regression_debug(
    sample_work: Path, row: dict, *, delta_pp: float, max_per_clip: float,
) -> Path:
    """Write a JSON bundle to help debug catastrophic pipeline WER vs USR-per-clip."""
    pipe_dir = sample_work / "pipeline"
    cache = pipe_dir / "cache"
    rttm_json = cache / "rttm_tracks.json"
    per_track = _rttm_rows_per_track(rttm_json)
    rttm_rows = None
    if rttm_json.is_file():
        try:
            rttm_rows = json.loads(rttm_json.read_text(encoding="utf-8"))
        except Exception:
            rttm_rows = None
    n_seg = row["pipeline_on_concat"].get("n_segments", 0)
    wc = row["usr_per_clip"]["wer"]
    dbg = {
        "reason": "pipeline_wer_regression_vs_usr_per_clip",
        "thresholds": {"min_delta_wer": delta_pp, "max_per_clip_wer": max_per_clip},
        "sample_id": row["sample_id"],
        "wer_usr_per_clip": wc,
        "wer_pipeline": row["pipeline_on_concat"]["wer"],
        "wer_delta_pipeline_minus_per_clip": (
            row["pipeline_on_concat"]["wer"] - wc
            if row["pipeline_on_concat"]["wer"] is not None else None
        ),
        "n_pipeline_segments_reported": n_seg,
        "ref_word_count": len(normalize_text(row["ref"]).split()),
        "usr_per_clip_pred_preview": row["usr_per_clip"]["pred"][:1200],
        "pipeline_pred_preview": (row["pipeline_on_concat"]["pred"] or "")[:1200],
        "rttm_tracks_path": str(rttm_json) if rttm_json.is_file() else None,
        "rttm_row_count": len(rttm_rows) if rttm_rows is not None else None,
        "rttm_rows_per_track_idx": per_track,
        "hint": (
            "If rttm_rows_per_track_idx shows counts > 1, multiple RTTM intervals map to the same mouth "
            "track; _get_attributed_transcript must merge them or you will repeat full-track transcriptions."
        ),
    }
    if per_track and n_seg and max(per_track.values(), default=0) > 1:
        dbg["likely_duplicate_transcription_bug"] = True
    out = sample_work / "pipeline_regression_abort_debug.json"
    out.write_text(json.dumps(dbg, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_done(output: Path) -> set[str]:
    if not output.is_file():
        return set()
    done = set()
    for ln in output.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception:
            continue
        sid = row.get("sample_id")
        if sid:
            done.add(sid)
    return done


def median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def paired_bootstrap_median_delta(
    paired: list[tuple[float, float]],
    *, n_resamples: int = 10_000, seed: int = 42, alpha: float = 0.05,
) -> dict | None:
    """Paired bootstrap 95% CI on median(a - b) over (a, b) pairs.

    Returns dict with point, ci_low, ci_high, n. None if not enough data.
    """
    pairs = [(a, b) for a, b in paired if a is not None and b is not None]
    if len(pairs) < 5:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    deltas = [a - b for a, b in pairs]
    point = statistics.median(deltas)
    resampled: list[float] = []
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        resampled.append(statistics.median(sample))
    resampled.sort()
    lo = resampled[int((alpha / 2) * n_resamples)]
    hi = resampled[int((1 - alpha / 2) * n_resamples)]
    return {"point": point, "ci_low": lo, "ci_high": hi, "n": n, "alpha": alpha}


def git_sha(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            sha = proc.stdout.strip()
            return sha or None
    except Exception:
        pass
    return None


def write_run_metadata(meta_path: Path, *, args: argparse.Namespace, samples_meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch as _t
        torch_ver = _t.__version__
    except Exception:
        torch_ver = None
    payload = {
        "command": " ".join([sys.argv[0]] + sys.argv[1:]),
        "git_sha": git_sha(_REPO_ROOT),
        "python": platform.python_version(),
        "torch": torch_ver,
        "ckpt": str(args.ckpt),
        "ckpt_sha256": sha256_file(args.ckpt) if args.ckpt.is_file() else None,
        "samples": str(args.samples),
        "samples_sha256": sha256_file(args.samples) if args.samples.is_file() else None,
        "samples_meta": samples_meta,
        "work_root": str(args.work_root),
        "output": str(args.output),
        "modality": args.modality,
        "use_au": bool(args.use_au),
        "au_features_dir": str(args.au_features_dir) if args.au_features_dir else None,
        "device": args.device,
        "num_samples": args.num_samples,
        "abort_pipeline_regression_delta": args.abort_pipeline_regression_delta,
        "abort_pipeline_regression_max_per_clip": args.abort_pipeline_regression_max_per_clip,
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    # Resolve to absolute paths so any subprocess (run_usr.py, ffmpeg) sees real paths
    # regardless of its own cwd.
    args.ckpt = args.ckpt.resolve()
    args.samples = args.samples.resolve()
    args.work_root = args.work_root.resolve()
    args.hf_cache = args.hf_cache.resolve()
    args.output = args.output.resolve()
    if args.au_features_dir is not None:
        args.au_features_dir = args.au_features_dir.resolve()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.ckpt}", flush=True)
    print(f"Modality: {args.modality}", flush=True)

    samples_meta, samples = load_samples(args.samples)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]
    print(f"Loaded {len(samples)} samples from {args.samples}", flush=True)

    done = set() if args.force else load_done(args.output)
    pending = [s for s in samples if s["sample_id"] not in done]
    print(f"{len(done)} already done, {len(pending)} pending.", flush=True)

    args.hf_cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    write_run_metadata(meta_path, args=args, samples_meta=samples_meta)
    print(f"Run metadata: {meta_path}", flush=True)

    print("Loading USR model ...", flush=True)
    model, cfg = build_model(args.ckpt, device, use_au=bool(args.use_au))
    beam = build_beam(model, cfg)

    hf_token = os.environ.get("HF_TOKEN")

    abort_delta = args.abort_pipeline_regression_delta
    abort_max_per_clip = args.abort_pipeline_regression_max_per_clip
    if getattr(args, "abort_pipeline_regression_max_concat_deprecated", None) is not None:
        abort_max_per_clip = args.abort_pipeline_regression_max_concat_deprecated
    regression_aborted = False
    regression_abort_info: dict | None = None

    t0 = time.time()
    with args.output.open("a", encoding="utf-8") as out_f:
        for i, sample in enumerate(pending, 1):
            sid = sample["sample_id"]
            print(f"\n[{i}/{len(pending)}] {sid}", flush=True)
            try:
                row = evaluate_sample(
                    sample, args=args,
                    model=model, beam=beam, cfg=cfg,
                    device=device, hf_token=hf_token,
                )
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  FAILED: {e}\n{tb}", flush=True)
                row = {
                    "sample_id": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": tb,
                }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            if "error" not in row:
                print(f"  USR/clip:     WER {row['usr_per_clip']['wer']:.2%}", flush=True)
                pipe = row["pipeline_on_concat"]
                if pipe["tracker_failed"]:
                    print(f"  Pipeline:     TRACKER FAILED ({pipe['error']})", flush=True)
                else:
                    print(f"  Pipeline:     WER {pipe['wer']:.2%}  (segments={pipe['n_segments']})", flush=True)

                if abort_delta is not None and not pipe["tracker_failed"] and pipe["wer"] is not None:
                    wc = row["usr_per_clip"]["wer"]
                    if wc <= abort_max_per_clip and (pipe["wer"] - wc) >= abort_delta:
                        sample_work = args.work_root / sid
                        dbg_path = dump_pipeline_regression_debug(
                            sample_work, row,
                            delta_pp=abort_delta, max_per_clip=abort_max_per_clip,
                        )
                        stamp = args.output.parent / f"{args.output.stem}.regression_abort.json"
                        regression_abort_info = {
                            "sample_id": sid,
                            "wer_usr_per_clip": wc,
                            "wer_pipeline": pipe["wer"],
                            "delta_pipeline_minus_per_clip": pipe["wer"] - wc,
                            "threshold_min_delta": abort_delta,
                            "threshold_max_per_clip": abort_max_per_clip,
                            "debug_json": str(dbg_path),
                            "elapsed_s": time.time() - t0,
                        }
                        stamp.write_text(
                            json.dumps(regression_abort_info, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                        print("\n" + "!" * 60, flush=True)
                        print(
                            "ABORT: Pipeline WER is far worse than USR-per-clip under the smoke guard "
                            f"(Δ ≥ {abort_delta:.2%} with /clip ≤ {abort_max_per_clip:.2%}).",
                            flush=True,
                        )
                        print(f"  Sample: {sid}", flush=True)
                        print(f"  /clip WER: {wc:.2%}  pipeline WER: {pipe['wer']:.2%}", flush=True)
                        print(f"  Debug:       {dbg_path}", flush=True)
                        print(f"  Abort stamp: {stamp}", flush=True)
                        print("!" * 60 + "\n", flush=True)
                        regression_aborted = True
                        break

    print(f"\nElapsed: {time.time() - t0:.1f}s", flush=True)

    if regression_aborted:
        print(
            "Run stopped early (exit 2). Fix pipeline concat / diarization–USR wiring, then re-run.\n"
            f"Details:\n{json.dumps(regression_abort_info, indent=2)}",
            flush=True,
        )
        return 2

    rows = [json.loads(ln) for ln in args.output.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [r for r in rows if "error" not in r]
    rows = [r for r in rows if r["sample_id"] in {s["sample_id"] for s in samples}]

    print("\n" + "=" * 60)
    print("SUMMARY (medians over non-failed samples)")
    print("=" * 60)
    n_total = len(rows)
    n_failed = sum(1 for r in rows if r["pipeline_on_concat"]["tracker_failed"])
    n_good = n_total - n_failed
    wer_per_clip = [r["usr_per_clip"]["wer"] for r in rows]
    wer_pipeline = [r["pipeline_on_concat"]["wer"] for r in rows if not r["pipeline_on_concat"]["tracker_failed"]]
    print(f"Samples scored:      {n_total}")
    print(f"Tracker-failed:      {n_failed} ({(n_failed / max(1, n_total)):.1%})")
    print(f"USR-per-clip   med:  {(median(wer_per_clip) or 0):.2%}")
    print(f"Pipeline       med:  {(median(wer_pipeline) or 0):.2%}  (n={n_good})")

    good_rows = [r for r in rows if not r["pipeline_on_concat"]["tracker_failed"]]
    paired_pipe_vs_clip = [
        (r["pipeline_on_concat"]["wer"], r["usr_per_clip"]["wer"]) for r in good_rows
    ]
    bs_pipe_vs_clip = paired_bootstrap_median_delta(paired_pipe_vs_clip)
    if bs_pipe_vs_clip:
        b = bs_pipe_vs_clip
        print(
            f"Δ med (pipe − /clip):     {b['point'] * 100:+.1f} pp  "
            f"[{b['ci_low'] * 100:+.1f}, {b['ci_high'] * 100:+.1f}]  (paired bootstrap 95% CI, n={b['n']})"
        )

    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary = {
        "n_total": n_total,
        "n_tracker_failed": n_failed,
        "tracker_failure_rate": (n_failed / n_total) if n_total else None,
        "median_wer_usr_per_clip": median(wer_per_clip),
        "median_wer_pipeline": median(wer_pipeline),
        "bootstrap_pipe_minus_per_clip": bs_pipe_vs_clip,
        "ckpt": str(args.ckpt),
        "modality": args.modality,
        "use_au": bool(args.use_au),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
