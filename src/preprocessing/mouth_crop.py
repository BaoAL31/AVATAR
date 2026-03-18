import face-alignment
import numpy as np
import cv2
import os
import pickle
import argparse
from tqdm import tqdm

MOUTH_CROP_SIZE = (96, 96)
MOUTH_START = 48
MOUTH_END   = 68
MOUTH_PAD = 10
BASE_FOLDER = "data/processed"

def get_paths(video_name, base_folder=BASE_FOLDER):
    """Build all required input/output paths from video name."""
    base_path = os.path.join(base_folder, video_name)
    return {
        "pyframes":   os.path.join(base_path, "pyframes"),
        "tracks":     os.path.join(base_path, "pywork", "tracks.pckl"),
        "mouth_crops": os.path.join(base_path, "mouth_crops"),
    }

def load_tracks(tracks_path):
    """Load face tracks from LR-ASD's tracks.pckl."""
    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)
    return tracks

def read_face_clip(clip_path):
    """
    Read all frames from an LR-ASD face clip avi.
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
        # fallback: return blank crop if bounding box is degenerate
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
 
    return cv2.resize(mouth_crop, size)

def process_video(video_name, base_folder=BASE_FOLDER, device="cuda"):
 """
    Main processing function. For each track from LR-ASD, runs face alignment
    on each frame and saves the mouth crop.
    """
    paths = get_paths(video_name, base_folder)
 
    # validate inputs exist
    if not os.path.exists(paths["tracks"]):
        raise FileNotFoundError(
            f"tracks.pckl not found at {paths['tracks']}. "
            f"Make sure LR-ASD has been run for this video first."
        )
    if not os.path.exists(paths["pycrop"]):
        raise FileNotFoundError(
            f"pycrop directory not found at {paths['pycrop']}. "
            f"Make sure LR-ASD has been run for this video first."
        )
 
    os.makedirs(paths["mouth_crops"], exist_ok=True)
 
    # load face tracks from LR-ASD
    tracks = load_tracks(paths["tracks"])
    print(f"Loaded {len(tracks)} face tracks for video: {video_name}")
 
    # initialise face alignment model once, reuse for all frames
    print(f"Initialising face_alignment on {device}...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=device,
        flip_input=False,
    )
 
    skipped = 0
    total   = 0
 
    for track_idx, track in enumerate(tqdm(tracks, desc="Processing tracks")):
        track_out_dir = os.path.join(paths["mouth_crops"], "%05d" % track_idx)
        os.makedirs(track_out_dir, exist_ok=True)

        # read face clip directly from LR-ASD pycrop output
        clip_path = os.path.join(paths["pycrop"], "%05d.avi" % track_idx)
        if not os.path.exists(clip_path):
            print(f"Warning: clip not found for track {track_idx}, skipping.")
            skipped += len(track["track"]["frame"])
            continue

        frames = read_face_clip(clip_path)
 
        for fidx, frame in enumerate(frames):
            total += 1
 
            # run face alignment on the 224x224 face crop
            # landmarks are relative to this cropped image, no offset needed
            landmarks_list = fa.get_landmarks(frame)
 
            if landmarks_list is None:
                # face_alignment found no face, save blank crop
                skipped += 1
                mouth_crop = np.zeros(
                    (MOUTH_CROP_SIZE[1], MOUTH_CROP_SIZE[0], 3), dtype=np.uint8
                )
            else:
                # take the first detected face
                landmarks  = landmarks_list[0]
                mouth_crop = get_mouth_crop(frame, landmarks)
 
            # save as jpg, named by local frame index within this track
            out_path = os.path.join(track_out_dir, "%06d.jpg" % fidx)
            cv2.imwrite(out_path, cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2BGR))
 
    print(f"\nDone. Processed {total} frames, skipped {skipped}.")
    print(f"Mouth crops saved to: {paths['mouth_crops']}")

