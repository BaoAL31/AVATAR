#!/usr/bin/env python3
"""Build the LRS2 fusion-eval sample list.

Reads ``data/val_manifest.csv`` and emits ``data/lrs2_fusion_samples.jsonl``
with N entries; each entry is K=3 LRS2 val clips drawn at random (seed=42)
whose total duration is at most ``--max-duration-s`` seconds. The manifest
does not carry speaker identity, so speaker distinctness across a triple is
probabilistic (see ``CONTEXT.md``).

A frame rate of 25 FPS is assumed (LRS2 / mouth-crop convention; FPS=25 in
``src/preprocess/mouth_crop.py``).

Output schema (one JSON object per line):

    {
      "sample_id": "fs_000",
      "total_frames": <int>,
      "total_duration_s": <float>,
      "clips": [
        {"tag": "lrs2_00", "rel": "0000218/0000218.avi", "frames": 33,
         "duration_s": 1.32, "ids": [582, 1005, ...]},
        ...
      ]
    }

Stdlib + pathlib only; no torch/cv2 dependencies (frame counts come from the
manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

FPS = 25.0
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = DEFAULT_REPO_ROOT / "data" / "val_manifest.csv"
DEFAULT_OUTPUT = DEFAULT_REPO_ROOT / "data" / "lrs2_fusion_samples.jsonl"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help=f"Input val manifest CSV. Default: {DEFAULT_MANIFEST}")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Output JSONL. Default: {DEFAULT_OUTPUT}")
    ap.add_argument("--num-samples", "-n", type=int, default=100,
                    help="Number of triples to draw. Default: 100.")
    ap.add_argument("--clips-per-sample", "-k", type=int, default=3,
                    help="Number of clips per sample (K). Default: 3.")
    ap.add_argument("--max-duration-s", type=float, default=25.0,
                    help="Reject samples whose total concat exceeds this. Default: 25.")
    ap.add_argument("--min-clip-frames", type=int, default=15,
                    help="Drop manifest rows shorter than this (avoids 0-frame stubs). Default: 15.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed. Default: 42.")
    ap.add_argument("--max-rejections", type=int, default=10_000,
                    help="Give up after this many duration-rejections in a row. Default: 10000.")
    return ap.parse_args()


def parse_manifest_row(line: str) -> tuple[str, str, int, list[int]] | None:
    parts = line.split(",", 3)
    if len(parts) != 4:
        return None
    tag = parts[0].strip()
    rel = parts[1].strip().replace("\\", "/")
    try:
        frames = int(parts[2].strip())
    except ValueError:
        return None
    raw_ids = parts[3].strip()
    ids = [int(x) for x in raw_ids.split()] if raw_ids else []
    if not rel or frames <= 0:
        return None
    return tag, rel, frames, ids


def load_manifest(path: Path, min_frames: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parsed = parse_manifest_row(s)
            if parsed is None:
                continue
            tag, rel, frames, ids = parsed
            if frames < min_frames:
                continue
            rows.append({
                "tag": tag,
                "rel": rel,
                "frames": frames,
                "duration_s": frames / FPS,
                "ids": ids,
            })
    return rows


def manifest_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_samples(
    rows: list[dict],
    *,
    n: int,
    k: int,
    max_duration_s: float,
    seed: int,
    max_rejections: int,
) -> list[list[dict]]:
    if len(rows) < k:
        raise SystemExit(f"manifest has only {len(rows)} usable rows; need at least {k}")
    rng = random.Random(seed)
    samples: list[list[dict]] = []
    rejections = 0
    while len(samples) < n:
        if rejections > max_rejections:
            raise SystemExit(
                f"Gave up after {rejections} rejections at {len(samples)}/{n} samples. "
                f"Lower --max-duration-s or relax --min-clip-frames."
            )
        idxs = rng.sample(range(len(rows)), k)
        triple = [rows[i] for i in idxs]
        total_s = sum(c["duration_s"] for c in triple)
        if total_s > max_duration_s:
            rejections += 1
            continue
        samples.append(triple)
        rejections = 0
    return samples


def write_jsonl(samples: list[list[dict]], output: Path, *, manifest_sha: str, args: argparse.Namespace) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "_meta": True,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "num_samples": len(samples),
        "clips_per_sample": args.clips_per_sample,
        "max_duration_s": args.max_duration_s,
        "min_clip_frames": args.min_clip_frames,
        "seed": args.seed,
        "fps_assumed": FPS,
    }
    with output.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, sort_keys=True) + "\n")
        for i, triple in enumerate(samples):
            total_frames = sum(c["frames"] for c in triple)
            entry = {
                "sample_id": f"fs_{i:03d}",
                "total_frames": total_frames,
                "total_duration_s": round(total_frames / FPS, 4),
                "clips": [
                    {
                        "tag": c["tag"],
                        "rel": c["rel"],
                        "frames": c["frames"],
                        "duration_s": round(c["duration_s"], 4),
                        "ids": c["ids"],
                    }
                    for c in triple
                ],
            }
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    rows = load_manifest(args.manifest, args.min_clip_frames)
    print(f"Loaded {len(rows)} usable rows from {args.manifest} "
          f"(filtered: frames>={args.min_clip_frames}).")
    if not rows:
        print("ERROR: manifest produced zero usable rows", file=sys.stderr)
        return 2

    samples = draw_samples(
        rows,
        n=args.num_samples,
        k=args.clips_per_sample,
        max_duration_s=args.max_duration_s,
        seed=args.seed,
        max_rejections=args.max_rejections,
    )
    durations = [sum(c["duration_s"] for c in triple) for triple in samples]
    print(f"Drew {len(samples)} samples "
          f"(duration s: min={min(durations):.2f}, "
          f"median={sorted(durations)[len(durations) // 2]:.2f}, "
          f"max={max(durations):.2f}).")

    sha = manifest_fingerprint(args.manifest)
    write_jsonl(samples, args.output, manifest_sha=sha, args=args)
    print(f"Wrote {args.output}")
    print(f"Manifest sha256: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
