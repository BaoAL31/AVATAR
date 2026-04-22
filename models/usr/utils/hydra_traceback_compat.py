"""Patch traceback.print_exception for Hydra <1.2 on Python 3.10+.

Hydra 1.1.x calls ``print_exception(etype=None, value=..., tb=...)``; ``etype`` was
removed and is invalid. Upgrading to hydra-core>=1.2 is preferred; this avoids a
secondary exception while formatting the real one.
"""

from __future__ import annotations

import traceback


_orig = traceback.print_exception


def _print_exception_compat(*args, **kwargs):
    # Hydra 1.1.x: print_exception(etype=None, value=ex, tb=final_tb) — no positional args.
    # Python 3.10+: print_exception(exc, /, value=..., tb=...) — first arg required; may be None if value+tb set.
    kwargs.pop("etype", None)
    if not args and ("value" in kwargs or "tb" in kwargs):
        return _orig(None, **kwargs)
    return _orig(*args, **kwargs)


traceback.print_exception = _print_exception_compat  # type: ignore[assignment]
