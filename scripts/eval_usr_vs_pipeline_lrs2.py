"""Compare USR-only vs e2e pipeline WER on LRS2 val samples.

Usage:
    LD_LIBRARY_PATH=/home/jembo/miniconda3/envs/usr_env/lib:/usr/lib/wsl/lib \\
    /home/jembo/miniconda3/envs/usr_env/bin/python \\
    scripts/eval_usr_vs_pipeline_lrs2.py \\
    --manifest data/val_manifest.csv \\
    --num-samples 10
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from huggingface_hub import hf_hub_download
from torchvision.transforms import CenterCrop, Grayscale

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "usr"))

from data.transforms import NormalizeVideo
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from utils.utils import UNIGRAM1000_LIST, ids_to_str

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CKPT = _REPO_ROOT / "models" / "usr" / "checkpoints" / "baseplus_high_resource_lrs3vox2.pth"
_DEFAULT_USR_CONF = _REPO_ROOT / "models" / "usr" / "conf"
_AVDIAR_DIR = _REPO_ROOT / "models" / "av-diarization"
if str(_AVDIAR_DIR) not in sys.path:
    sys.path.insert(0, str(_AVDIAR_DIR))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=_REPO_ROOT / "data" / "val_manifest.csv")
    ap.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    ap.add_argument("--repo-prefix", type=str, default="HBaoAL/LRS2")
    ap.add_argument("--ckpt", type=Path, default=_DEFAULT_CKPT)
    ap.add_argument("--modality", type=str, default="av", choices=["a", "v", "av"])
    return ap.parse_args()


def repo_from_tag(tag: str, prefix: str) -> str:
    m = re.match(r"^lrs2_(\d{2})$", tag)
    if m:
        return f"{prefix}_{m.group(1)}"
    if tag == "lrs2":
        return prefix
    raise ValueError(f"Unsupported tag: {tag}")


def load_manifest(path: Path, num: int):
    with path.open("r", encoding="utf-8") as f:
        rows = [ln.strip() for ln in f if ln.strip()]
    samples = []
    for i, row in enumerate(rows[:num]):
        parts = row.split(",", 3)
        if len(parts) != 4:
            continue
        tag = parts[0].strip()
        rel = parts[1].strip().replace("\\", "/")
        ids = [int(x) for x in parts[3].strip().split()] if parts[3].strip() else []
        samples.append((tag, rel, ids, parts[2].strip()))
    return samples


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


def load_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    x = torch.from_numpy(np.stack(frames)).permute((3, 0, 1, 2)).float() / 255.0
    x = CenterCrop(88)(x)
    x = x.transpose(0, 1)
    x = Grayscale()(x)
    x = x.transpose(0, 1)
    x = NormalizeVideo(mean=(0.421,), std=(0.165,))(x)
    return x


def load_audio(path: str):
    wav, _ = torchaudio.load(path, normalize=True)
    return wav


def build_model(ckpt_path: Path, device: torch.device):
    GlobalHydra.instance().clear()
    config_name = "config" if (_DEFAULT_USR_CONF / "config.yaml").exists() else "config_ft_lrs2_lora_base"
    with initialize_config_dir(config_dir=str(_DEFAULT_USR_CONF)):
        cfg = compose(config_name=config_name, overrides=["experiment_name=test", "model/backbone=resnet_transformer_baseplus"])
    model = E2E(1049, cfg.model.backbone)
    state = torch.load(str(ckpt_path), map_location=device)
    state = normalize_ckpt(state)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, cfg


def build_beam_search(model, cfg):
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


def transcribe_usr(model, beam, cfg, video_path: str, audio_path: str, modality: str, device: torch.device) -> str:
    video = load_video(video_path).to(device)
    audio = load_audio(audio_path).to(device)  # (1, T_raw)
    # Pad audio if too short for conv1d frontend (kernel_size=80)
    if audio.shape[1] < 16000:
        audio = torch.nn.functional.pad(audio, (0, 16000 - audio.shape[1]))
    with torch.no_grad():
        if modality == "v":
            feat, _, _ = model.encoder.forward_single(xs_v=video)
        elif modality == "a":
            feat, _, _ = model.encoder.forward_single(xs_a=audio.unsqueeze(0).transpose(1, 2))
        else:
            try:
                feat, _, _ = model.encoder.forward_single(xs_v=video, xs_a=audio.unsqueeze(0).transpose(1, 2))
            except RuntimeError:
                feat, _, _ = model.encoder.forward_single(xs_v=video)
        nbest = beam(x=feat.squeeze(0), modality=modality,
                     maxlenratio=cfg.decode.maxlenratio, minlenratio=cfg.decode.minlenratio)
    return add_results_to_json([nbest[0].asdict()], UNIGRAM1000_LIST).replace("<eos>", "").replace("▁", " ").strip()


def get_wer(pred: str, ref: str) -> float:
    """Levenshtein WER."""
    pred_words = pred.strip().split()
    ref_words = ref.strip().split()
    n = len(ref_words)
    if n == 0:
        return 0.0
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


def _patch_preprocessor():
    """Monkey-patch Preprocessor defaults for short LRS2 clips."""
    import voxconverse.preprocessor as pp
    old_init = pp.Preprocessor.__init__
    def patched_init(self, cache_dir, ckpt_dir=None, device="cpu", frame_rate=25,
                     crop_scale=0.40, min_track=1, num_failed_det=100,
                     min_face_size=40, facedet_scale=0.35):
        old_init(self, cache_dir, ckpt_dir, device, frame_rate, crop_scale,
                 min_track, num_failed_det, min_face_size, facedet_scale)
    pp.Preprocessor.__init__ = patched_init


def run_pipeline(mp4_path: str, output_dir: str) -> str:
    """Run e2e pipeline on a single MP4 and return the transcription text."""
    _patch_preprocessor()
    from src.pipeline import Pipeline
    p = Pipeline(video_path=mp4_path, output_dir=output_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results = p.run()
    if not results:
        return ""
    return results[0].get("transcription", "")


def download_sample(repo_id: str, folder: str, stem: str, tag: str, cache_dir: Path):
    """Download .mp4, .wav, .txt for a sample. Return local paths."""
    rel_mp4 = f"{folder}/{stem}.mp4"
    # For manifest .avi-based paths, map to .mp4
    rel_wav = f"{folder}/{stem}.wav"
    rel_txt = f"{folder}/{stem}.txt"
    token = os.environ.get("HF_TOKEN")
    # Some shards may not have .mp4; try .mp4 first, fall back to using .avi
    try:
        local_mp4 = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_mp4, cache_dir=str(cache_dir), token=token)
    except Exception:
        rel_avi = f"{folder}/{stem}.avi"
        local_mp4 = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_avi, cache_dir=str(cache_dir), token=token)
    local_wav = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_wav, cache_dir=str(cache_dir), token=token)
    local_txt = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_txt, cache_dir=str(cache_dir), token=token)
    return local_mp4, local_wav, local_txt


def read_ground_truth(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("text:"):
                return s.split(":", 1)[1].strip()
    return ""


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    samples = load_manifest(args.manifest, args.num_samples)
    print(f"Loaded {len(samples)} samples from {args.manifest}")
    print()

    # Build USR model once
    print("Loading USR model...")
    model, cfg = build_model(args.ckpt.resolve(), device)
    beam = build_beam_search(model, cfg)
    print("USR model loaded.")
    print()

    cache_dir = _REPO_ROOT / "data" / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for tag, rel_video, ids, n_frames in samples:
        stem = Path(rel_video).stem
        folder = Path(rel_video).parent.as_posix()
        repo_id = repo_from_tag(tag, args.repo_prefix)

        print(f"[{stem}] downloading from {repo_id}...")
        local_video, local_wav, local_txt = download_sample(repo_id, folder, stem, tag, cache_dir)
        gt_text = read_ground_truth(local_txt)
        if not gt_text:
            gt_text = ids_to_str(ids, UNIGRAM1000_LIST).replace("▁", " ").replace("<eos>", "").strip()

        # USR-only
        print(f"  USR-only ({args.modality})...")
        pred_usr = transcribe_usr(model, beam, cfg, local_video, local_wav, args.modality, device)
        wer_usr = get_wer(pred_usr, gt_text)
        print(f"    PRED: {pred_usr}")
        print(f"    WER:  {wer_usr:.2%}")

        # E2E pipeline
        print(f"  E2E pipeline...")
        work_dir = _REPO_ROOT / "tmp" / "eval_lrs2" / stem
        try:
            # If video is .avi (mouth crop), it has no audio. Skip muxing, directly use it.
            if local_video.endswith(".mp4"):
                # Mux .mp4 + .wav into a single file
                muxed = str(Path(work_dir) / "input.mp4")
                Path(work_dir).mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", local_video, "-i", local_wav,
                     "-c:v", "copy", "-c:a", "aac", "-shortest", muxed],
                    check=True, capture_output=True,
                )
                input_path = muxed
            else:
                # .avi (mouth crop) — no audio track, use .wav alongside
                input_path = local_video

            pred_pipe = run_pipeline(input_path, str(work_dir))
            wer_pipe = get_wer(pred_pipe, gt_text) if pred_pipe else 1.0
            print(f"    PRED: {pred_pipe}")
            print(f"    WER:  {wer_pipe:.2%}")
        except Exception as e:
            print(f"    FAILED: {e}")
            pred_pipe = ""
            wer_pipe = 1.0

        results.append({
            "stem": stem,
            "gt": gt_text,
            "usr": {"pred": pred_usr, "wer": wer_usr},
            "pipeline": {"pred": pred_pipe, "wer": wer_pipe},
        })
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    wer_usr_list = [r["usr"]["wer"] for r in results]
    wer_pipe_list = [r["pipeline"]["wer"] for r in results]
    print(f"{'Sample':<20} {'USR WER':<10} {'Pipeline WER':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['stem']:<20} {r['usr']['wer']:.2%}     {r['pipeline']['wer']:.2%}")
    print("-" * 45)
    print(f"{'AVG':<20} {sum(wer_usr_list)/len(wer_usr_list):.2%}     {sum(wer_pipe_list)/len(wer_pipe_list):.2%}")


if __name__ == "__main__":
    main()
