"""Resolve dataset media paths via Hugging Face Hub (cached local files)."""

from __future__ import annotations

import os
import random
import time
from pathlib import PurePosixPath
from typing import Optional

from .hf_env import ensure_hf_env

ensure_hf_env()


def hub_local_path(
    repo_id: str,
    repo_filename: str,
    *,
    repo_type: str = "dataset",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_retries: int = 8,
) -> str:
    """Return a local path. Uses the Hugging Face hub cache unless ``cache_dir`` is set."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import ReadTimeout as RequestsReadTimeout

    fn = PurePosixPath(str(repo_filename).replace("\\", "/")).as_posix()
    kwargs = {"repo_id": repo_id, "filename": fn, "repo_type": repo_type}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    # Default False: huggingface_hub still does HTTP (ETag / resolve) even when blobs are cached.
    # Set HF_MEDIA_LOCAL_ONLY=1 once the cache is warm to skip Hub calls (fails if a file is missing locally).
    if os.environ.get("HF_MEDIA_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes", "on"):
        kwargs["local_files_only"] = True

    def _is_retryable_hub_error(err: BaseException) -> bool:
        """Retry temporary Hub-side/server-side failures and timeout-wrapped misses."""
        if isinstance(err, HfHubHTTPError):
            status_code = getattr(getattr(err, "response", None), "status_code", None)
            return status_code is None or int(status_code) >= 500

        if isinstance(err, LocalEntryNotFoundError):
            # This can wrap transient Hub metadata/HEAD failures when cache is cold.
            msg = str(err).lower()
            return (
                "taking longer than expected" in msg
                or "gateway timeout" in msg
                or "try again later" in msg
                or "connection" in msg
                or "timeout" in msg
            )
        return False

    force_download_next = False
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            call_kwargs = dict(kwargs)
            if force_download_next:
                call_kwargs["force_download"] = True
            return hf_hub_download(**call_kwargs)
        except (RequestsReadTimeout, RequestsConnectionError) as e:
            last_err = e
            # jittered exponential backoff (Hub / XET can throttle under parallel workers)
            delay = min(2.0**attempt, 120.0) + random.uniform(0, 1.0)
            time.sleep(delay)
        except (HfHubHTTPError, LocalEntryNotFoundError) as e:
            last_err = e
            if _is_retryable_hub_error(e):
                delay = min(2.0**attempt, 120.0) + random.uniform(0, 1.0)
                time.sleep(delay)
                continue
            raise
        except OSError as e:
            last_err = e
            # Recover from partial/corrupt cache entry (e.g., size 0 wav): retry forcing a fresh download.
            if "Consistency check failed" in str(e):
                force_download_next = True
                delay = min(2.0**attempt, 120.0) + random.uniform(0, 1.0)
                time.sleep(delay)
                continue
            raise
    assert last_err is not None
    raise last_err
