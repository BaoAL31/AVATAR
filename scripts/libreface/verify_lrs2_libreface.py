"""
Verify LibreFace *.npz outputs produced by ``process_lrs2_libreface.py process`` (export).

Checks the same column keys as scripts/libreface_npz_schema.py (LibreFace default
video + joint model), plus per-array sanity (consistent frame length, finite floats).

NPZ layout (auto-detected under --output-dir):
  - flat:   output_dir/*.npz
  - nested: output_dir/<stem>/<stem>.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/verify_lrs2_libreface.py` from repo root
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from typing import FrozenSet, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from libreface_npz_schema import CORE_NPZ_KEYS, EXPECTED_NPZ_KEY_SET

# Default directory: match process_lrs2_libreface.py --output-dir
_DEFAULT_LIBREFACE_OUT = Path("/data/hoangbng/libreface_out")

# Same as process_lrs2_libreface: ignore these when scanning nested layout
SKIP_SUBDIRS = frozenset({"libreface_out", "temp", "libreface_weights"})


def discover_npz_files(root: Path) -> tuple[list[Path], str]:
    """
    Find all NPZ paths under root.

    - Flat: root/*.npz
    - Nested: root/<stem>/<stem>.npz
    """
    flat = sorted(root.glob("*.npz"))
    if flat:
        return flat, "flat"

    nested: list[Path] = []
    if not root.is_dir():
        return [], "nested"
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith("."):
            continue
        if d.name in SKIP_SUBDIRS:
            continue
        candidate = d / f"{d.name}.npz"
        if candidate.is_file():
            nested.append(candidate)

    return sorted(nested), "nested"


def _array_first_dim_length(arr: np.ndarray) -> Optional[int]:
    if arr.ndim == 0:
        return None
    if arr.dtype == object:
        try:
            return int(arr.shape[0])
        except Exception:
            return None
    return int(arr.shape[0])


def _is_float_kind(dtype: np.dtype) -> bool:
    return dtype.kind in "fc"


def validate_npz_contents(
    data: np.lib.npyio.NpzFile,
    required_keys: FrozenSet[str],
    strict_no_extra_keys: bool,
) -> Tuple[Optional[str], List[str]]:
    """Return (error or None, warnings)."""
    names: Set[str] = set(data.files)
    warnings: List[str] = []

    if not names:
        return "Empty NPZ (no arrays)", warnings

    missing = required_keys - names
    if missing:
        sample = sorted(missing)[:12]
        return (
            f"Missing {len(missing)} required keys (showing up to 12): {sample}"
            + (" ..." if len(missing) > 12 else ""),
            warnings,
        )

    extra = names - required_keys
    if extra:
        msg = f"{len(extra)} unexpected extra keys (not in schema), e.g. {sorted(extra)[:8]}"
        if strict_no_extra_keys:
            return msg, warnings
        warnings.append(msg)

    lengths: Set[int] = set()
    for name in sorted(names):
        try:
            arr = data[name]
        except Exception as e:
            return f"Key {name!r}: cannot read array ({e})", warnings

        if not isinstance(arr, np.ndarray):
            return f"Key {name!r}: expected ndarray, got {type(arr).__name__}", warnings

        n = _array_first_dim_length(arr)
        if n is not None and n > 0:
            lengths.add(n)

        if _is_float_kind(arr.dtype) and arr.size > 0:
            flat = arr.ravel()
            if not np.all(np.isfinite(flat)):
                return f"Key {name!r}: non-finite float values (nan/inf)", warnings

    if len(lengths) > 1:
        return f"Inconsistent frame lengths across arrays: {sorted(lengths)}", warnings

    return None, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify LibreFace NPZ outputs against process_lrs2_libreface schema"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_LIBREFACE_OUT,
        help="Directory containing *.npz files",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only require CORE_NPZ_KEYS (frame, headpose, AUs, expression) — no MediaPipe landmark columns",
    )
    parser.add_argument(
        "--strict-keys",
        action="store_true",
        help="Fail if NPZ contains keys not listed in the schema for this mode",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Only verify the first N *.npz files (sorted by path). Default: all files.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit / -n must be >= 1")

    required: FrozenSet[str] = (
        frozenset(CORE_NPZ_KEYS) if args.core_only else EXPECTED_NPZ_KEY_SET
    )

    output_dir = args.output_dir.resolve()
    npz_paths, layout = discover_npz_files(output_dir)
    total_in_dir = len(npz_paths)
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]

    corrupted: List[Tuple[str, str]] = []
    valid = 0
    warn_extra_total = 0

    for npz_path in tqdm(npz_paths, desc="Verifying NPZ"):
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                err, warns = validate_npz_contents(
                    data, required, strict_no_extra_keys=args.strict_keys
                )
                if err:
                    corrupted.append((npz_path.name, err))
                else:
                    valid += 1
                    if warns:
                        warn_extra_total += 1
        except Exception as e:
            corrupted.append((npz_path.name, str(e)))

    print("\n--- Verification Summary ---")
    print(f"NPZ layout: {layout} (under {output_dir})")
    print(f"Schema: {'CORE_NPZ_KEYS only' if args.core_only else 'full EXPECTED_NPZ_KEYS (incl. landmarks)'}")
    if args.limit is not None:
        print(
            f"NPZ files checked: {len(npz_paths)} "
            f"(limit {args.limit}; {total_in_dir} *.npz discovered)"
        )
    else:
        print(f"NPZ files checked: {len(npz_paths)}")
    print(f"Valid: {valid}")
    print(f"Failed: {len(corrupted)}")
    if warn_extra_total and not args.strict_keys:
        print(
            f"Files with extra keys (warning only; use --strict-keys to fail): {warn_extra_total}"
        )

    if not corrupted:
        print("All NPZ files passed required-key and structure checks.")
    else:
        print("Verification completed with issues.")
        for name, msg in corrupted[:15]:
            print(f"  - {name}: {msg}")
        if len(corrupted) > 15:
            print(f"  ... and {len(corrupted) - 15} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
