"""Resolve dataset media paths via Hugging Face Hub (cached local files)."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Optional


def hub_local_path(
    repo_id: str,
    repo_filename: str,
    *,
    repo_type: str = "dataset",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Return a local path. Uses the Hugging Face hub cache unless ``cache_dir`` is set."""
    from huggingface_hub import hf_hub_download

    fn = PurePosixPath(str(repo_filename).replace("\\", "/")).as_posix()
    kwargs = {"repo_id": repo_id, "filename": fn, "repo_type": repo_type}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return hf_hub_download(**kwargs)
