"""LibreFace NPZ → per-frame AU tensor (aligned with process_lrs2_libreface / libreface_npz_schema)."""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Same order as scripts/libreface_npz_schema.py (joint model)
AU_DETECTION_KEYS: Tuple[str, ...] = (
    "au_1",
    "au_2",
    "au_4",
    "au_6",
    "au_7",
    "au_10",
    "au_12",
    "au_14",
    "au_15",
    "au_17",
    "au_23",
    "au_24",
)
AU_INTENSITY_KEYS: Tuple[str, ...] = (
    "au_1_intensity",
    "au_2_intensity",
    "au_4_intensity",
    "au_5_intensity",
    "au_6_intensity",
    "au_9_intensity",
    "au_12_intensity",
    "au_15_intensity",
    "au_17_intensity",
    "au_20_intensity",
    "au_25_intensity",
    "au_26_intensity",
)

AU_FEATURE_KEYS: Tuple[str, ...] = AU_DETECTION_KEYS + AU_INTENSITY_KEYS
AU_FEATURE_DIM: int = len(AU_FEATURE_KEYS)


def load_au_from_npz(npz_path: str, target_len: int) -> torch.Tensor:
    """
    Load AU detection + intensity columns, resample linearly to target_len (video frames).
    Returns float tensor (target_len, AU_FEATURE_DIM). Zeros if file missing or invalid.
    """
    out = torch.zeros(target_len, AU_FEATURE_DIM, dtype=torch.float32)
    if not npz_path or not os.path.isfile(npz_path):
        return out

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            cols: List[np.ndarray] = []
            for k in AU_FEATURE_KEYS:
                if k not in data.files:
                    return out
                cols.append(np.asarray(data[k], dtype=np.float32).reshape(-1))
            t = int(cols[0].shape[0])
            if t == 0:
                return out
            for c in cols[1:]:
                if int(c.shape[0]) != t:
                    return out
            arr = np.stack(cols, axis=1)  # T_npz, D
    except Exception:
        return out

    t = torch.from_numpy(arr)
    if t.size(0) == target_len:
        return t
    # (T_npz, D) → interpolate time
    x = t.T.unsqueeze(0)  # 1, D, T_npz
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    x = x.squeeze(0).T.contiguous()
    return x.float()
