#!/usr/bin/env python3
"""
Backward-compatible entry point: same CLI as before, forwarded to
``process_lrs2_libreface.py upload`` (default ``--npz-dir`` is ``/data/hoangbng/libreface_out``).

Prefer calling the combined script explicitly::

  python3 scripts/libreface/process_lrs2_libreface.py upload --dry-run --repo-prefix HBaoAL/LRS2
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _main() -> int:
    here = Path(__file__).resolve().parent
    target = here / "process_lrs2_libreface.py"
    spec = importlib.util.spec_from_file_location("_lrs2_libreface_combined", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.main(["upload", *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(_main())
