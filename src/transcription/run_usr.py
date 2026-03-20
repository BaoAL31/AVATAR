import os
import sys
import subprocess
import tempfile
import cv2
import numpy as np
import argparse

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import torch
import torchaudio
from torchvision.transforms import CenterCrop, Compose, Grayscale, Lambda
from data.transforms import NormalizeVideo
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from utils.utils import UNIGRAM1000_LIST

DATA_DIR = "/home/jembo/AVATAR/data/processed"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_paths(video_name, data_dir=DATA_DIR):
    base = os.path.join(data_dir, f"{video_name}")
    return {
        "mouth_crops": os.path.join(base, "mouth_crops"),
        "pycrop":      os.path.join(base, "cache", "pycrop"),
        "pycrop_wav":  os.path.join(base, "cache", "pycrop_wav"),
        "rttm":        os.path.join(base, f"{video_name}.rttm"),
    }

def load_and_preprocess_track(video_name, track_idx):
    '''
    Input: video name and track index
    Load the corresponding mouth crop video and audio wav file, 
    preprocess them according to usr's format, and return as tensors.
    Output: a pair of (video tensor, audio tensor) for the given track
    '''
    paths = get_paths(video_name)
    mouth_avi = os.path.join(paths["mouth_crops"], "%05d.avi" % track_idx)
    audio_wav = os.path.join(paths["pycrop_wav"], "%05d.wav" % track_idx)
    
    mouth_tensor = video_to_tensor(mouth_avi)
    mouth_tensor = mouth_tensor / 255.
    mouth_tensor = CenterCrop(88)(mouth_tensor)
    mouth_tensor = mouth_tensor.transpose(0, 1)   # (C, T, H, W) -> (T, C, H, W) for Grayscale
    mouth_tensor = Grayscale()(mouth_tensor)
    mouth_tensor = mouth_tensor.transpose(0, 1)   # (T, C, H, W) -> (C, T, H, W)
    mouth_tensor = NormalizeVideo(mean=(0.421,), std=(0.165,))(mouth_tensor)
    
    audio_tensor = audio_to_tensor(audio_wav)
    
    return mouth_tensor, audio_tensor

def video_to_tensor(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = torch.from_numpy(np.stack(frames))
    return frames.permute((3, 0, 1, 2))  # cv2 gives frames in TxHxWxC, need to change to CxTxHxW to comply with torch

def audio_to_tensor(path):
    audio, sr = torchaudio.load(path, normalize=True)
    return audio

def load_model(ckpt_path: str, device: torch.device = DEVICE):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir='/home/jembo/AVATAR/models/usr/conf'):
        cfg = compose(config_name='config', overrides=[
            'experiment_name=test',
            'model/backbone=resnet_transformer_baseplus'
            ])

    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace('_orig_mod.model.backbone.', ''): v for k, v in state_dict.items()}
    model = E2E(1049, cfg.model.backbone)  
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, cfg

def transcribe(video_name: str, track_idx: int, model: E2E, cfg, device: torch.device = DEVICE, modality: str = "av") -> str:
    video, audio = load_and_preprocess_track(video_name, track_idx)
    video = video.to(device)
    audio = audio.to(device)

    beam_search = get_beam_search(cfg, model)

    with torch.no_grad():
        feat, _, _ = model.encoder.forward_single(
            xs_v=video, 
            xs_a=audio.unsqueeze(0).transpose(1, 2)
        )
        nbest_hyps = beam_search(
            x=feat.squeeze(0),
            modality=modality,
            maxlenratio=cfg.decode.maxlenratio,
            minlenratio=cfg.decode.minlenratio
        )

    nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]
    transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
    transcription = transcription.replace("<eos>", "").replace("▁", " ").strip()

    return transcription


def get_beam_search(cfg, model: E2E) -> BatchBeamSearch:
    token_list = UNIGRAM1000_LIST
    odim = len(token_list)
    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = dict(
        decoder=1.0 - cfg.decode.ctc_weight,
        ctc=cfg.decode.ctc_weight,
        lm=cfg.decode.lm_weight,
        length_bonus=cfg.decode.penalty,
    )
    return BatchBeamSearch(
        beam_size=cfg.decode.beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=odim - 1,
        eos=odim - 1,
        token_list=token_list,
        pre_beam_score_key=None if cfg.decode.ctc_weight == 1.0 else "decoder",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--track_idx", type=int, required=True)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt_path, device)
    result = transcribe(args.video_name, args.track_idx, model, cfg, device)

    with open(args.output, "w") as f:
        f.write(result)