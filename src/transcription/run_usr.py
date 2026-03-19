import os
import sys
import subprocess
import tempfile
import cv2
import numpy as np
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

def load_model(ckpt_path: str, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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