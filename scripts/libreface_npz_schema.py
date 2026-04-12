"""
Expected keys in NPZ files written by scripts/process_lrs2_libreface.py.

LibreFace `get_facial_attributes` (video, default joint model) returns a pandas
DataFrame; we save `np.savez_compressed(..., **{col: array per column})`.

Schema matches upstream LibreFace (ihp-lab/LibreFace) for:
- `get_frames_from_video_ffmpeg` → frame_idx, frame_time_in_ms (path_to_frame dropped before join)
- head pose json → pitch, yaw, roll
- MediaPipe landmarks → lm_mp_{i}_{x|y|z} for i in 0..477 (478 points × 3)
- AU detection → au_* per solver_inference_combine.au_detection_aus
- AU intensity → au_*_intensity per au_recognition_aus
- expression → facial_expression

If you upgrade libreface and keys change, update this module and re-verify.
"""

from __future__ import annotations

from typing import FrozenSet, Tuple

# --- LibreFace AU_Recognition/solver_inference_combine.py (joint model) ---
AU_DETECTION_AUS: Tuple[int, ...] = (1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24)
AU_RECOGNITION_AUS: Tuple[int, ...] = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)

# MediaPipe FaceMesh landmark count used by restructure_landmark_mediapipe (enumerate landmarks).
NUM_MEDIAPIPE_LANDMARKS: int = 478

FRAME_KEYS: Tuple[str, ...] = ("frame_idx", "frame_time_in_ms")
HEADPOSE_KEYS: Tuple[str, ...] = ("pitch", "yaw", "roll")
EXPR_KEYS: Tuple[str, ...] = ("facial_expression",)

AU_DETECTION_KEYS: Tuple[str, ...] = tuple(f"au_{k}" for k in AU_DETECTION_AUS)
AU_INTENSITY_KEYS: Tuple[str, ...] = tuple(f"au_{k}_intensity" for k in AU_RECOGNITION_AUS)


def landmark_column_keys(num_points: int = NUM_MEDIAPIPE_LANDMARKS) -> Tuple[str, ...]:
    keys = []
    for i in range(num_points):
        keys.extend((f"lm_mp_{i}_x", f"lm_mp_{i}_y", f"lm_mp_{i}_z"))
    return tuple(keys)


# Full column set saved by process_lrs2_libreface (default libreface video pipeline).
EXPECTED_NPZ_KEYS: Tuple[str, ...] = (
    FRAME_KEYS
    + HEADPOSE_KEYS
    + landmark_column_keys()
    + AU_DETECTION_KEYS
    + AU_INTENSITY_KEYS
    + EXPR_KEYS
)

EXPECTED_NPZ_KEY_SET: FrozenSet[str] = frozenset(EXPECTED_NPZ_KEYS)

# Subset checks (no full landmark list): timing + pose + AUs + expression.
CORE_NPZ_KEYS: Tuple[str, ...] = (
    FRAME_KEYS + HEADPOSE_KEYS + AU_DETECTION_KEYS + AU_INTENSITY_KEYS + EXPR_KEYS
)
