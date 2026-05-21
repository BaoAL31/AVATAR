import os
import sys
import json
import argparse
from pathlib import Path

# Resolve repo root + USR root from this file's location:
#   <repo>/src/transcription/run_usr.py  ->  <repo>/models/usr
_REPO_ROOT = Path(__file__).resolve().parents[2]
_USR_DIR = _REPO_ROOT / "models" / "usr"

# Ensure USR-local packages (data/, espnet/, utils/) are importable when the
# script is launched directly (e.g. from CI or a notebook) without first
# `cd`-ing into models/usr/. Idempotent if already on sys.path.
if str(_USR_DIR) not in sys.path:
    sys.path.insert(0, str(_USR_DIR))

import cv2
import numpy as np
import torch
import torchaudio
from torchvision.transforms import CenterCrop, Grayscale

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from data.transforms import NormalizeVideo
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from utils.utils import UNIGRAM1000_LIST
from utils.au_npz import AU_FEATURE_DIM

# Default checkpoint + Hydra config dir live under the repo's USR tree.
# Override either with environment variables for unusual layouts.
CKPT_PATH = os.environ.get(
    "AVATAR_USR_CKPT",
    str(_USR_DIR / "checkpoints" / "baseplus_high_resource_lrs3vox2.pth"),
)
USR_CONF = os.environ.get(
    "AVATAR_USR_CONF",
    str(_USR_DIR / "conf"),
)


def _detokenize_sentencepiece(text: str) -> str:
    """Convert espnet sentencepiece output to clean text.

    `add_results_to_json` returns tokens joined by spaces with the Unigram
    word-start marker `\u2581` prefixed on every word-initial subword:

        "\u2581HE 'S \u2581A \u2581FINALIST \u2581NAME <eos>"

    The previous implementation did `.replace("\u2581", " ").strip()` only,
    which left orphan subwords (e.g. `\u2581CHRISTIA NS \u2581HAVE` -> `
    CHRISTIA NS  HAVE`). Correct detokenization removes the intra-token
    spaces FIRST (joining subwords into words), then converts the word-start
    marker into a real space.
    """
    return (
        text.replace("<eos>", "")
            .replace(" ", "")
            .replace("\u2581", " ")
            .strip()
    )


def _safe_default_device():
    """Honor AVATAR_FORCE_CPU and survive partial-CUDA WSL setups."""
    if os.environ.get("AVATAR_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


class Transcriber:
    def __init__(self, video_name: str, output_dir: str, ckpt_path: str = CKPT_PATH,
                 device: "torch.device | None" = None,
                 au_concat_path: "str | None" = None):
        self.video_name = video_name
        self.output_dir = output_dir
        self.ckpt_path  = ckpt_path
        self.device     = device if device is not None else _safe_default_device()
        self.au_concat_path = au_concat_path
        self.au_concat = self._load_au_concat()
        self.paths      = self._get_paths()
        self.model, self.cfg = self._load_model()

    def _get_paths(self) -> dict:
        return {
            "mouth_crops": os.path.join(self.output_dir, "cache", "mouth_crops"),
            "pycrop_wav":  os.path.join(self.output_dir, "cache", "pycrop_wav"),
            "rttm_tracks": os.path.join(self.output_dir, "cache", "rttm_tracks.json"),
        }

    def _load_au_concat(self) -> "torch.Tensor | None":
        """Load the pre-computed sample-level AU tensor, shape (T_total, AU_FEATURE_DIM).

        Returned tensor is on CPU; ``_au_for_track`` slices and moves to device. None if
        no path was supplied.
        """
        if not self.au_concat_path:
            return None
        if not os.path.isfile(self.au_concat_path):
            print(f"[run_usr] WARNING: au_concat_path not found: {self.au_concat_path}", flush=True)
            return None
        arr = np.load(self.au_concat_path)
        if arr.ndim != 2 or arr.shape[1] != AU_FEATURE_DIM:
            print(f"[run_usr] WARNING: au_concat has unexpected shape {arr.shape}; expected (T,{AU_FEATURE_DIM})", flush=True)
            return None
        return torch.from_numpy(arr.astype(np.float32))

    def _au_for_track(self, track_idx: int, target_len: int) -> "torch.Tensor | None":
        """Slice the sample-level AU tensor for one track and resample to target_len.

        ``rttm_tracks.json`` stores **speech-segment** intervals, not visual-track
        intervals; a single ``track_idx`` may appear in multiple entries (one per
        speech turn). The mouth-crop AVI for the track spans the *visual* range
        (first to last face-detected frame), which is bounded below by the earliest
        speech turn's start and above by the latest turn's end. We take that union
        as the track-time approximation — it's a few frames loose vs. ``tracks.pkl``
        truth but avoids a heavy pickle import path, and the final
        ``F.interpolate`` to ``target_len`` absorbs the slack.

        Returns a CPU tensor of shape (target_len, AU_FEATURE_DIM), or None if
        AU is unavailable.
        """
        if self.au_concat is None:
            return None
        rttm_path = self.paths["rttm_tracks"]
        if not os.path.isfile(rttm_path):
            return None
        try:
            with open(rttm_path) as f:
                rttm_entries = json.load(f)
        except Exception:
            return None
        starts: list[float] = []
        ends: list[float] = []
        for e in rttm_entries:
            if int(e.get("track_idx", -1)) == int(track_idx):
                starts.append(float(e.get("start", 0.0)))
                ends.append(float(e.get("end", 0.0)))
        if not starts:
            return None
        start = min(starts)
        end = max(ends)
        fps = 25
        s_frame = max(0, int(round(start * fps)))
        e_frame = max(s_frame, int(round(end * fps)))
        e_frame = min(e_frame, self.au_concat.size(0))
        if e_frame <= s_frame:
            return None
        slc = self.au_concat[s_frame:e_frame]  # (T_slice, D)
        if slc.size(0) == target_len:
            return slc.contiguous()
        x = slc.T.unsqueeze(0)  # (1, D, T_slice)
        x = torch.nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.squeeze(0).T.contiguous()

    def _load_model(self):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=USR_CONF):
            # Use LoRA AU config if available (resnet_transformer_base + use_au=true),
            # else fall back to original logic (resnet_transformer_baseplus).
            au_config = os.path.join(USR_CONF, "config_ft_lrs2_lora_au_base.yaml")
            if os.path.exists(au_config):
                cfg = compose(
                    config_name="config_ft_lrs2_lora_au_base",
                    overrides=["experiment_name=test"],
                )
            else:
                config_name = "config" if os.path.exists(os.path.join(USR_CONF, "config.yaml")) else "config_ft_lrs2_lora_base"
                cfg = compose(
                    config_name=config_name,
                    overrides=[
                        'experiment_name=test',
                        'model/backbone=resnet_transformer_baseplus'
                    ]
                )
        state_dict = torch.load(self.ckpt_path, map_location=self.device)
        state_dict = {k.replace('_orig_mod.model.backbone.', ''): v for k, v in state_dict.items()}
        model = E2E(1049, cfg.model.backbone)
        # Direct key-by-key load to avoid PyTorch 2.0.1's strict size checks
        model_sd = model.state_dict()
        for k, v in state_dict.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                model_sd[k].copy_(v)
        model = model.to(self.device)
        model.eval()
        return model, cfg

    def _video_to_tensor(self, path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        frames = torch.from_numpy(np.stack(frames))
        return frames.permute((3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)

    def _audio_to_tensor(self, path: str) -> torch.Tensor:
        audio, _ = torchaudio.load(path, normalize=True)
        return audio

    def _transcribe_inputs(self, video_path: str, audio_path: str, modality: str = "av") -> str:
        video = self._video_to_tensor(video_path)
        video = video / 255.
        video = CenterCrop(88)(video)
        video = video.transpose(0, 1)
        video = Grayscale()(video)
        video = video.transpose(0, 1)
        video = NormalizeVideo(mean=(0.421,), std=(0.165,))(video)
        video = video.to(self.device)
        audio = self._audio_to_tensor(audio_path).to(self.device)
        beam_search = self._get_beam_search()

        with torch.no_grad():
            if modality == "v":
                feat, _, _ = self.model.encoder.forward_single(xs_v=video)
            elif modality == "a":
                feat, _, _ = self.model.encoder.forward_single(xs_a=audio.unsqueeze(0).transpose(1, 2))
            else:
                feat, _, _ = self.model.encoder.forward_single(
                    xs_v=video,
                    xs_a=audio.unsqueeze(0).transpose(1, 2)
                )
            nbest_hyps = beam_search(
                x=feat.squeeze(0),
                modality=modality,
                maxlenratio=self.cfg.decode.maxlenratio,
                minlenratio=self.cfg.decode.minlenratio
            )

        nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]
        transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
        return _detokenize_sentencepiece(transcription)

    def _load_and_preprocess_track(self, track_idx: int):
        mouth_avi = os.path.join(self.paths["mouth_crops"], "%05d.avi" % track_idx)
        audio_wav = os.path.join(self.paths["pycrop_wav"],  "%05d.wav" % track_idx)

        mouth_tensor = self._video_to_tensor(mouth_avi)
        mouth_tensor = mouth_tensor / 255.
        mouth_tensor = CenterCrop(88)(mouth_tensor)
        mouth_tensor = mouth_tensor.transpose(0, 1)   # (C, T, H, W) -> (T, C, H, W) for Grayscale
        mouth_tensor = Grayscale()(mouth_tensor)
        mouth_tensor = mouth_tensor.transpose(0, 1)   # (T, C, H, W) -> (C, T, H, W)
        mouth_tensor = NormalizeVideo(mean=(0.421,), std=(0.165,))(mouth_tensor)

        audio_tensor = self._audio_to_tensor(audio_wav)
        return mouth_tensor, audio_tensor

    def _get_beam_search(self) -> BatchBeamSearch:
        token_list = UNIGRAM1000_LIST
        odim = len(token_list)
        scorers = self.model.scorers()
        scorers["lm"] = None
        scorers["length_bonus"] = LengthBonus(odim)
        weights = dict(
            decoder=1.0 - self.cfg.decode.ctc_weight,
            ctc=self.cfg.decode.ctc_weight,
            lm=self.cfg.decode.lm_weight,
            length_bonus=self.cfg.decode.penalty,
        )
        return BatchBeamSearch(
            beam_size=self.cfg.decode.beam_size,
            vocab_size=odim,
            weights=weights,
            scorers=scorers,
            sos=odim - 1,
            eos=odim - 1,
            token_list=token_list,
            pre_beam_score_key=None if self.cfg.decode.ctc_weight == 1.0 else "decoder",
        )

    def _transcribe_track(self, track_idx: int, modality: str = "av") -> str:
        video, audio = self._load_and_preprocess_track(track_idx)
        video = video.to(self.device)
        audio = audio.to(self.device)
        au_in = None
        if self.au_concat is not None:
            au_track = self._au_for_track(track_idx, target_len=int(video.size(1)))
            if au_track is not None:
                au_in = au_track.unsqueeze(0).to(self.device)  # (1, T, D)

        beam_search = self._get_beam_search()

        with torch.no_grad():
            feat, _, _ = self.model.encoder.forward_single(
                xs_v=video,
                xs_a=audio.unsqueeze(0).transpose(1, 2),
                au=au_in,
            )
            nbest_hyps = beam_search(
                x=feat.squeeze(0),
                modality=modality,
                maxlenratio=self.cfg.decode.maxlenratio,
                minlenratio=self.cfg.decode.minlenratio
            )

        nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]
        transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
        return _detokenize_sentencepiece(transcription)

    def run(self) -> dict:
        tracks = sorted([f for f in os.listdir(self.paths["mouth_crops"]) if f.endswith(".avi")])
        results = {}
        if not tracks:
            # Fallback path: no per-track mouth crops available; run one-shot transcription
            # on preprocessed full video/audio so pipeline can still emit usable text output.
            full_video = os.path.join(self.output_dir, "cache", "pyavi", "video.avi")
            full_audio = os.path.join(self.output_dir, "cache", "pyavi", "audio.wav")
            if os.path.exists(full_video) and os.path.exists(full_audio):
                print("No mouth tracks found. Falling back to full-video transcription...")
                results["0"] = self._transcribe_inputs(full_video, full_audio, modality="v")
            return results
        for track_avi in tracks:
            track_idx = int(os.path.splitext(track_avi)[0])
            print(f"Transcribing track {track_idx}...")
            results[str(track_idx)] = self._transcribe_track(track_idx)
        return results

# Work around due to dependency issues with USR and the main pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_path",  type=str, default=CKPT_PATH)
    parser.add_argument("--tmp_output", type=str, required=True)
    parser.add_argument("--au_concat_path", type=str, default=None,
                        help="Optional sample-level AU tensor (T_total, AU_FEATURE_DIM). "
                             "Required when --ckpt_path is a LoRA-AU checkpoint.")
    args = parser.parse_args()

    transcriber = Transcriber(args.video_name, args.output_dir, args.ckpt_path,
                              au_concat_path=args.au_concat_path)
    results = transcriber.run()

    with open(args.tmp_output, "w") as f:
        json.dump(results, f)