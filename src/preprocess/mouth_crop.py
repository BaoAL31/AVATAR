import face_alignment
import numpy as np
import cv2
import os
import pickle
import json
import torch
from tqdm import tqdm

MOUTH_CROP_SIZE = (96, 96)
MOUTH_START = 48
MOUTH_END   = 68
MOUTH_PAD   = 10
FPS         = 25


class MouthCropper:
    def __init__(self, video_name: str, output_dir: str, device: torch.device):
        self.video_name = video_name
        self.output_dir = output_dir
        self.device = device
        self.paths = self._get_paths()

    def _get_paths(self) -> dict:
        return {
            "pycrop":      os.path.join(self.output_dir, "cache", "pycrop"),
            "tracks":      os.path.join(self.output_dir, "cache", "tracks.pkl"),
            "mouth_crops": os.path.join(self.output_dir, "cache", "mouth_crops"),
            "rttm_tracks": os.path.join(self.output_dir, "cache", "rttm_tracks.json"),
        }

    def _read_face_clip(self, clip_path: str) -> list:
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

    def _get_mouth_crop(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        mouth_pts = landmarks[MOUTH_START:MOUTH_END]

        x_min = int(np.min(mouth_pts[:, 0])) - MOUTH_PAD
        x_max = int(np.max(mouth_pts[:, 0])) + MOUTH_PAD
        y_min = int(np.min(mouth_pts[:, 1])) - MOUTH_PAD
        y_max = int(np.max(mouth_pts[:, 1])) + MOUTH_PAD

        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        mouth_crop = image[y_min:y_max, x_min:x_max]
        if mouth_crop.size == 0:
            return np.zeros((MOUTH_CROP_SIZE[1], MOUTH_CROP_SIZE[0], 3), dtype=np.uint8)

        return cv2.resize(mouth_crop, MOUTH_CROP_SIZE)

    def run(self):
        if not os.path.exists(self.paths["tracks"]):
            raise FileNotFoundError(f"tracks.pkl not found at {self.paths['tracks']}")
        if not os.path.exists(self.paths["pycrop"]):
            raise FileNotFoundError(f"pycrop directory not found at {self.paths['pycrop']}")

        os.makedirs(self.paths["mouth_crops"], exist_ok=True)

        with open(self.paths["rttm_tracks"]) as f:
            rttm_tracks = json.load(f)
        track_indices = list(set(entry["track_idx"] for entry in rttm_tracks))
        print(f"Processing {len(track_indices)} unique tracks")

        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=self.device.type,
            flip_input=False,
        )

        skipped = 0
        total   = 0

        for track_idx in tqdm(track_indices, desc="Processing tracks"):
            clip_path = os.path.join(self.paths["pycrop"], "%05d.avi" % track_idx)
            if not os.path.exists(clip_path):
                print(f"Warning: clip not found for track {track_idx}, skipping.")
                continue

            frames = self._read_face_clip(clip_path)
            out_path = os.path.join(self.paths["mouth_crops"], "%05d.avi" % track_idx)
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
                    mouth_crop = self._get_mouth_crop(frame, landmarks_list[0])
                writer.write(cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2BGR))

            writer.release()

        print(f"\nDone. Processed {total} frames, skipped {skipped}.")
        print(f"Mouth crops saved to: {self.paths['mouth_crops']}")


def process_video(video_name: str, output_dir: str = None, device: torch.device = None):
    if output_dir is None:
        output_dir = f"/home/hoangbng/AVATAR/AVATAR/data/processed/{video_name}"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cropper = MouthCropper(video_name, output_dir, device)
    cropper.run()
