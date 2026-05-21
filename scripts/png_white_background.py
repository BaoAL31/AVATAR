"""Composite RGBA PNGs onto a white background (in-place by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def flatten_to_white(path: Path, *, background: tuple[int, int, int] = (255, 255, 255)) -> bool:
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (*background, 255))
            out = Image.alpha_composite(bg, rgba).convert("RGB")
            out.save(path, format="PNG", optimize=True)
            return True
        if im.mode != "RGB":
            im.convert("RGB").save(path, format="PNG", optimize=True)
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="PNG files or directories (default: assets/)",
    )
    parser.add_argument(
        "--bg",
        default="255,255,255",
        help="Background RGB, e.g. 255,255,255",
    )
    args = parser.parse_args()
    bg = tuple(int(x) for x in args.bg.split(","))
    if len(bg) != 3:
        raise SystemExit("--bg must be three comma-separated integers")

    roots = args.paths or [Path("assets")]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.png")))
        elif root.suffix.lower() == ".png":
            files.append(root)
        else:
            raise SystemExit(f"Not a PNG file or directory: {root}")

    if not files:
        raise SystemExit("No PNG files found")

    changed = 0
    for path in files:
        if ".venv" in path.parts:
            continue
        if flatten_to_white(path, background=bg):  # type: ignore[arg-type]
            print(f"flattened: {path}")
            changed += 1
    print(f"done ({changed} file(s))")


if __name__ == "__main__":
    main()
