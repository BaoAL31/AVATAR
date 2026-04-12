"""Hugging Face Hub cache paths (aligned with ``huggingface_hub`` / ``hf_hub_download``)."""

from __future__ import annotations

import os


def default_hf_hub_cache_path() -> str:
    """
    Root directory where ``hf_hub_download`` stores files when ``cache_dir`` is not passed.
    Honors ``HF_HUB_CACHE``, then ``HF_HOME``/hub, then the usual default under ``~/.cache``.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return str(HF_HUB_CACHE)
    except ImportError:
        pass
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return os.path.expanduser(hub)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(os.path.expanduser(hf_home), "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def register_hf_hydra_resolvers() -> None:
    """Register ``${hf_hub_cache_path:}`` for use in YAML (call before @hydra.main)."""
    from omegaconf import OmegaConf

    if getattr(OmegaConf, "has_resolver", lambda *_: False)("hf_hub_cache_path"):
        return
    try:
        OmegaConf.register_new_resolver(
            "hf_hub_cache_path",
            lambda: default_hf_hub_cache_path(),
            use_cache=False,
        )
    except ValueError:
        pass
