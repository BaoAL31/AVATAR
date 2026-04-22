"""Set Hugging Face cache env vars to the large /data quota before hub/datasets imports."""

from __future__ import annotations

import os
from typing import Any

# Large Hub/datasets blobs live here. We intentionally do NOT set HF_HOME to this path:
# the CLI stores your token at ~/.cache/huggingface/token (i.e. under default HF_HOME).
# If HF_HOME pointed at /data/..., the library would look for /data/.../token and get 401s.
_DEFAULT_HF_DATA_ROOT = "/data/hoangbng/.cache/huggingface"


def ensure_hf_env() -> None:
    """Point Hub/datasets (and other large HF caches) at /data; leave HF_HOME unset.

    ``HF_HOME`` stays default (~/.cache/huggingface) so ``hf auth login`` token at
    ``~/.cache/huggingface/token`` is found. All heavy downloads use the paths below.
    """
    root = os.path.expanduser(_DEFAULT_HF_DATA_ROOT)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(root, "datasets"))
    hub = os.path.join(root, "hub")
    os.environ.setdefault("HF_HUB_CACHE", hub)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub)
    assets = os.path.join(root, "assets")
    os.environ.setdefault("HF_ASSETS_CACHE", assets)
    os.environ.setdefault("HUGGINGFACE_ASSETS_CACHE", assets)
    os.environ.setdefault("HF_XET_CACHE", os.path.join(root, "xet"))
    # Default hub uses 10s read timeout; large / XET-backed files often need longer (see HF constants).
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    # After the Hub cache is fully populated, set HF_MEDIA_LOCAL_ONLY=1 so training resolves paths
    # from disk only (no per-file HTTP). First-time download must run with this unset/false.


def apply_hub_download_ui_env(cfg: Any) -> None:
    """Set ``HF_HUB_DISABLE_PROGRESS_BARS`` from training config before ``huggingface_hub`` first import.

    Call once at the start of ``main_ft.main`` (Hydra). See top-level ``hub_disable_progress_bars`` in YAML.
    """
    try:
        from omegaconf import OmegaConf

        v = OmegaConf.select(cfg, "hub_disable_progress_bars", default=True)
    except Exception:
        v = getattr(cfg, "hub_disable_progress_bars", True)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1" if v else "0"
