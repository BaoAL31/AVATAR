import face_alignment
import numpy as np
import cv2
import os
import pickle
import argparse
import glob
from tqdm import tqdm

MOUTH_CROP_SIZE = (96, 96)
MOUTH_START = 48
MOUTH_END   = 68
MOUTH_PAD = 10
BASE_FOLDER = "/home/jembo/AVATAR/data/processed"
FPS = 25

def get_paths(video_name, base_folder=BASE_FOLDER):
    base_path = os.path.join(base_folder, f"{video_name}_av", "cache")
    return {
        "pycrop":      os.path.join(base_path, "pycrop"),
        "tracks":      os.path.join(base_path, "tracks.pckl"),
        "mouth_crops": os.path.join(base_folder, f"{video_name}_av", "mouth_crops"),
    }

def process_video(video_name, base_folder=BASE_FOLDER, device="cuda"):
    paths = get_paths(video_name, base_folder)

    if not os.path.exists(paths["tracks"]):
        raise FileNotFoundError(f"tracks.pckl not found at {paths['tracks']}")
    if not os.path.exists(paths["pycrop"]):
        raise FileNotFoundError(f"pycrop directory not found at {paths['pycrop']}")

    os.makedirs(paths["mouth_crops"], exist_ok=True)

    tracks = load_tracks(paths["tracks"])
    print(f"Loaded {len(tracks)} face tracks for video: {video_name}")

    print(f"Initialising face_alignment on {device}...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=device,
        flip_input=False,
    )

    skipped = 0
    total   = 0

    for track_idx, track in enumerate(tqdm(tracks, desc="Processing tracks")):
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
                landmarks  = landmarks_list[0]
                mouth_crop = get_mouth_crop(frame, landmarks)

            writer.write(cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2BGR))

        writer.release()

    print(f"\nDone. Processed {total} frames, skipped {skipped}.")
    print(f"Mouth crops saved to: {paths['mouth_crops']}")