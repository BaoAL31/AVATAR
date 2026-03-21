import face_alignment
import numpy as np
import cv2
import os
import pickle
import json
from tqdm import tqdm

MOUTH_CROP_SIZE = (96, 96)
MOUTH_START = 48
MOUTH_END   = 68
MOUTH_PAD = 10
DATA_DIR = "./data/processed"
FPS = 25

def get_paths(video_name, base_folder=DATA_DIR):
    """Build all required input/output paths from video name."""
    base_folder = os.path.join(base_folder, f"{video_name}")
    return {
        "pycrop":      os.path.join(base_folder, "cache", "pycrop"),
        "tracks":      os.path.join(base_folder, "cache", "tracks.pkl"),
        "mouth_crops": os.path.join(base_folder, "mouth_crops"),
        "rttm_tracks": os.path.join(base_folder, "cache", "rttm_tracks.json"),
    }

def load_tracks(tracks_path):
    """Load face tracks from tracks.pkl."""
    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)
    return tracks

def read_face_clip(clip_path):
    """
    Read all frames from a face clip avi.
    Returns a list of RGB numpy arrays.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise IOError(f"Could not open face clip: {clip_path}")

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def get_mouth_crop(image, landmarks, pad=MOUTH_PAD, size=MOUTH_CROP_SIZE):
    """
    Extract and resize the mouth region from an image using facial landmarks.

    image:     numpy array (H, W, 3) in RGB, already cropped to face region
    landmarks: (68, 2) array of x,y coordinates relative to this image
    """
    mouth_pts = landmarks[MOUTH_START:MOUTH_END]

    # bounding box around mouth points
    x_min = int(np.min(mouth_pts[:, 0])) - pad
    x_max = int(np.max(mouth_pts[:, 0])) + pad
    y_min = int(np.min(mouth_pts[:, 1])) - pad
    y_max = int(np.max(mouth_pts[:, 1])) + pad

    # clamp to image bounds
    h, w = image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    mouth_crop = image[y_min:y_max, x_min:x_max]

    if mouth_crop.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    return cv2.resize(mouth_crop, size)

def process_video(video_name, base_folder=DATA_DIR, device="cuda"):
    paths = get_paths(video_name, base_folder)

    if not os.path.exists(paths["tracks"]):
        raise FileNotFoundError(f"tracks.pckl not found at {paths['tracks']}")
    if not os.path.exists(paths["pycrop"]):
        raise FileNotFoundError(f"pycrop directory not found at {paths['pycrop']}")

    os.makedirs(paths["mouth_crops"], exist_ok=True)

    # only process unique track indices from rttm_tracks
    with open(paths["rttm_tracks"]) as f:
        rttm_tracks = json.load(f)
    track_indices = list(set(rttm_track["track_idx"] for rttm_track in rttm_tracks))
    print(f"Processing {len(track_indices)} unique tracks")

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=device,
        flip_input=False,
    )

    skipped = 0
    total = 0

    for track_idx in tqdm(track_indices, desc="Processing tracks"):
        clip_path = os.path.join(paths["pycrop"], "%05d.avi" % track_idx)
        if not os.path.exists(clip_path):
            print(f"Warning: clip not found for track {track_idx}, skipping.")
            continue

        frames = read_face_clip(clip_path)
        out_path = os.path.join(paths["mouth_crops"], "%05d.avi" % track_idx)
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            FPS,
            MOUTH_CROP_SIZE
        )

        for frame in frames:
            total += 1
            landmarks_list = fa.get_landmarks(frame)
            if landmarks_list is None:
                skipped += 1
                mouth_crop = np.zeros((MOUTH_CROP_SIZE[1], MOUTH_CROP_SIZE[0], 3), dtype=np.uint8)
            else:
                landmarks = landmarks_list[0]
                mouth_crop = get_mouth_crop(frame, landmarks)
            writer.write(cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2BGR))

        writer.release()

    print(f"\nDone. Processed {total} frames, skipped {skipped}.")
    print(f"Mouth crops saved to: {paths['mouth_crops']}")