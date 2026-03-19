import os
import sys
import subprocess
import tempfile
import cv2
import numpy as np
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
    base = os.path.join(data_dir, f"{video_name}_av")
    return {
        "mouth_crops": os.path.join(base, "mouth_crops"),
        "pycrop":      os.path.join(base, "cache", "pycrop"),
        "pycrop_wav":  os.path.join(base, "cache", "pycrop_wav"),
        "rttm":        os.path.join(base, f"{video_name}.rttm"),
    }

def load_track(video_name, track_idx):
    paths = get_paths(video_name)
    mouth_avi = os.path.join(paths["mouth_crops"], "%05d.avi" % track_idx)
    audio_wav = os.path.join(paths["pycrop_wav"], "%05d.wav" % track_idx)
    
    video = load_video(mouth_avi)
    audio = load_audio(audio_wav)
    
    return video, audio

