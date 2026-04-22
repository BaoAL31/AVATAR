import os
import glob
import pickle
import argparse
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering
from insightface.app import FaceAnalysis


BASE_DIR = "/home/hoangbng/AVATAR/AVATAR/data/processed"

def get_paths(video_name, base_dir=BASE_DIR):
    base = os.path.join(base_dir, video_name)
    return {
        "pywork":   os.path.join(base, "pywork"),
        "pyframes": os.path.join(base, "pyframes"),
        "tracks":   os.path.join(base, "pywork", "tracks.pkl"),
        "scores":   os.path.join(base, "pywork", "scores.pkl"),
        "rttm":     os.path.join(base, "diarization.rttm"),
    }

def load_face_model():
    app = FaceAnalysis(
        name='buffalo_sc',
        allowed_modules=['recognition'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(128, 128))
    return app

def diarize(video_name:str):
    paths = get_paths(video_name)

    if not os.path.exists(paths["tracks"]):
        raise FileNotFoundError(f"tracks.pkl not found: {paths['tracks']}")
    if not os.path.exists(paths["scores"]):
        raise FileNotFoundError(f"scores.pkl not found: {paths['scores']}")
    if not os.path.exists(paths["pyframes"]):
        raise FileNotFoundError(f"pyframes not found: {paths['pyframes']}")

    with open(paths["tracks"], "rb") as f:
        tracks = pickle.load(f)
    with open(paths["scores"], "rb") as f:
        scores = pickle.load(f)


    return paths["rttm"]

if __name__ == "__main__":
    diarize("-FaXLcSFjUI_trimmed")