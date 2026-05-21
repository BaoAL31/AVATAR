#!/usr/bin/env python3
"""Concatenate one (or all) LRS2 fusion samples into a stitched MP4 + WAV.

Reads ``data/lrs2_fusion_samples.jsonl``, downloads each clip's ``.mp4`` and
``.wav`` from HF (repo: ``HBaoAL/LRS2_<suffix>``), and stitches them via
ffmpeg with a 0.2 s silence + 5 black-frame seam between every pair of
clips. Output is re-encoded to 25 FPS, libx264 video, AAC mono 16 kHz audio,
so the downstream pipeline (face detect → track → mouth crop) sees a
uniform stream.

Per-sample outputs land under ``data/lrs2_fusion/<sample_id>/``:

    input.mp4   ← stitched A+V (this is what the pipeline ingests)
    input.wav   ← stitched audio only (this is what USR-only ingests)
    refs.txt    ← concatenated ground-truth transcript, one clip per line
    sources.txt ← which source clips were used (for debugging)

Stitched files are *not* regenerated if they already exist (idempotent).

Requirements: ``ffmpeg`` on PATH, ``hf_hub_download`` available, and an
``HF_TOKEN`` env var (or anonymous access if the LRS2 mirror is public for you).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = DEFAULT_REPO_ROOT / "data" / "lrs2_fusion_samples.jsonl"
DEFAULT_WORKROOT = DEFAULT_REPO_ROOT / "data" / "lrs2_fusion"
DEFAULT_HF_CACHE = DEFAULT_REPO_ROOT / "data" / "hf_cache"
DEFAULT_REPO_PREFIX = "HBaoAL/LRS2"

FPS = 25
AUDIO_SR = 16000
SEAM_VIDEO_FRAMES = 5
SEAM_AUDIO_S = 0.2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES,
                    help=f"JSONL produced by build_lrs2_fusion_samples.py. Default: {DEFAULT_SAMPLES}")
    ap.add_argument("--work-root", type=Path, default=DEFAULT_WORKROOT,
                    help=f"Per-sample output root. Default: {DEFAULT_WORKROOT}")
    ap.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE,
                    help=f"HF Hub cache dir. Default: {DEFAULT_HF_CACHE}")
    ap.add_argument("--repo-prefix", type=str, default=DEFAULT_REPO_PREFIX,
                    help=f"HF repo prefix (shards are <prefix>_NN). Default: {DEFAULT_REPO_PREFIX}")
    ap.add_argument("--sample-id", type=str, default=None,
                    help="Build only this sample_id (e.g. fs_000). If omitted, build all.")
    ap.add_argument("--force", action="store_true",
                    help="Re-stitch even if output already exists.")
    ap.add_argument("--ffmpeg", type=str, default="ffmpeg")
    ap.add_argument("--quiet", action="store_true", help="Suppress ffmpeg's stderr.")
    return ap.parse_args()


def repo_from_tag(tag: str, prefix: str) -> str:
    m = re.match(r"^lrs2_(\d{2})$", tag)
    if m:
        return f"{prefix}_{m.group(1)}"
    if tag == "lrs2":
        return prefix
    raise ValueError(f"Unsupported manifest tag: {tag}")


def load_samples(path: Path) -> tuple[dict, list[dict]]:
    if not path.is_file():
        raise SystemExit(f"samples file not found: {path}")
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        raise SystemExit(f"samples file is empty: {path}")
    header, samples = (rows[0], rows[1:]) if rows[0].get("_meta") else ({}, rows)
    return header, samples


def hf_download(repo_id: str, filename: str, cache_dir: Path, *, token: str | None) -> Path:
    return Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(cache_dir),
        token=token,
    ))


def fetch_clip(rel: str, tag: str, *, repo_prefix: str, cache_dir: Path,
               token: str | None) -> tuple[Path, Path, Path | None]:
    """Return (mp4_path, wav_path, txt_path_or_None). Raises on missing mp4 or wav."""
    repo_id = repo_from_tag(tag, repo_prefix)
    folder = Path(rel).parent.as_posix()
    stem = Path(rel).stem
    mp4_rel = f"{folder}/{stem}.mp4"
    wav_rel = f"{folder}/{stem}.wav"
    txt_rel = f"{folder}/{stem}.txt"

    mp4 = hf_download(repo_id, mp4_rel, cache_dir, token=token)
    wav = hf_download(repo_id, wav_rel, cache_dir, token=token)
    try:
        txt = hf_download(repo_id, txt_rel, cache_dir, token=token)
    except Exception:
        txt = None
    return mp4, wav, txt


def read_gt_text(txt_path: Path | None) -> str:
    if txt_path is None or not txt_path.is_file():
        return ""
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.lower().startswith("text:"):
            return s.split(":", 1)[1].strip()
    return ""


def probe_resolution(ffmpeg: str, video_path: Path) -> tuple[int, int]:
    """Use ffprobe-via-ffmpeg to extract first-frame WxH. Returns (w, h)."""
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path)],
        capture_output=True, text=True,
    )
    stderr = proc.stderr
    m = re.search(r"(\d{2,5})x(\d{2,5})", stderr)
    if not m:
        raise RuntimeError(f"Could not probe resolution for {video_path}:\n{stderr}")
    return int(m.group(1)), int(m.group(2))


def run_ffmpeg(cmd: list[str], *, quiet: bool) -> None:
    proc = subprocess.run(
        cmd,
        capture_output=quiet,
        text=True,
    )
    if proc.returncode != 0:
        if quiet:
            sys.stderr.write(proc.stderr or "")
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {' '.join(cmd)}")


def build_seam(ffmpeg: str, dest: Path, *, width: int, height: int, quiet: bool) -> None:
    """Generate a short black/silent clip used between source clips."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    seam_duration = max(SEAM_VIDEO_FRAMES / FPS, SEAM_AUDIO_S)
    cmd = [
        ffmpeg, "-y", "-hide_banner",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={FPS}:d={seam_duration:.3f}",
        "-f", "lavfi", "-i", f"anullsrc=channel_layout=mono:sample_rate={AUDIO_SR}",
        "-t", f"{seam_duration:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
        "-c:a", "aac", "-shortest",
        str(dest),
    ]
    run_ffmpeg(cmd, quiet=quiet)


def stitch(ffmpeg: str, *, clips: list[Path], wavs: list[Path], seam: Path,
           width: int, height: int, out_mp4: Path, out_wav: Path, quiet: bool) -> None:
    """Concatenate clips with seams via the concat filter, then split the audio.

    The seam input is referenced once per inter-clip gap, so we explicitly
    split / asplit its video and audio streams; ffmpeg disallows consuming the
    same filter_complex input twice without an explicit split.
    """
    inputs: list[str] = []
    for i, mp4 in enumerate(clips):
        inputs += ["-i", str(mp4)]
        inputs += ["-i", str(wavs[i])]
    n_clips = len(clips)
    n_inputs = 2 * n_clips
    seam_idx = n_inputs
    inputs += ["-i", str(seam)]
    n_seams = n_clips - 1

    parts: list[str] = []
    for i in range(n_clips):
        parts.append(f"[{2 * i}:v:0]scale={width}:{height},fps={FPS},format=yuv420p[v{i}]")
        parts.append(f"[{2 * i + 1}:a:0]aformat=channel_layouts=mono:sample_rates={AUDIO_SR}[a{i}]")

    if n_seams > 0:
        # Canonicalize seam video/audio first, THEN split into the n_seams uses, so
        # every clip and every seam segment lands on the concat filter with the same
        # SAR / pix_fmt / sample-rate / channel-layout. ffmpeg's concat filter fails
        # otherwise.
        parts.append(
            f"[{seam_idx}:v:0]scale={width}:{height},fps={FPS},format=yuv420p[seam_v_can]"
        )
        parts.append(
            f"[{seam_idx}:a:0]aformat=channel_layouts=mono:sample_rates={AUDIO_SR}[seam_a_can]"
        )
        seam_v_outs = "".join(f"[sv{j}]" for j in range(n_seams))
        seam_a_outs = "".join(f"[sa{j}]" for j in range(n_seams))
        parts.append(f"[seam_v_can]split={n_seams}{seam_v_outs}")
        parts.append(f"[seam_a_can]asplit={n_seams}{seam_a_outs}")

    concat_segments: list[str] = []
    for i in range(n_clips):
        concat_segments.append(f"[v{i}][a{i}]")
        if i < n_clips - 1:
            concat_segments.append(f"[sv{i}][sa{i}]")
    n_streams = n_clips + n_seams
    concat_filter = "".join(concat_segments) + f"concat=n={n_streams}:v=1:a=1[outv][outa]"
    parts.append(concat_filter)
    full_filter = ";".join(parts)

    cmd = [
        ffmpeg, "-y", "-hide_banner",
        *inputs,
        "-filter_complex", full_filter,
        "-map", "[outv]", "-map", "[outa]",
        "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
        "-c:a", "aac",
        str(out_mp4),
    ]
    run_ffmpeg(cmd, quiet=quiet)

    cmd_wav = [
        ffmpeg, "-y", "-hide_banner",
        "-i", str(out_mp4),
        "-ac", "1", "-ar", str(AUDIO_SR),
        "-vn",
        str(out_wav),
    ]
    run_ffmpeg(cmd_wav, quiet=quiet)


def process_sample(sample: dict, *, args: argparse.Namespace, token: str | None) -> dict:
    sample_id = sample["sample_id"]
    out_dir = args.work_root / sample_id
    out_mp4 = out_dir / "input.mp4"
    out_wav = out_dir / "input.wav"
    refs = out_dir / "refs.txt"
    sources = out_dir / "sources.txt"

    if not args.force and out_mp4.exists() and out_wav.exists() and refs.exists():
        return {"sample_id": sample_id, "status": "exists", "out_mp4": str(out_mp4)}

    out_dir.mkdir(parents=True, exist_ok=True)
    mp4s: list[Path] = []
    wavs: list[Path] = []
    gt_lines: list[str] = []
    source_lines: list[str] = []
    for clip in sample["clips"]:
        mp4, wav, txt = fetch_clip(
            clip["rel"], clip["tag"],
            repo_prefix=args.repo_prefix,
            cache_dir=args.hf_cache,
            token=token,
        )
        mp4s.append(mp4)
        wavs.append(wav)
        gt_lines.append(read_gt_text(txt))
        source_lines.append(f"{clip['tag']}\t{clip['rel']}\tmp4={mp4}\twav={wav}")

    width, height = probe_resolution(args.ffmpeg, mp4s[0])
    seam = args.work_root / ".seam" / f"seam_{width}x{height}.mp4"
    build_seam(args.ffmpeg, seam, width=width, height=height, quiet=args.quiet)

    stitch(
        args.ffmpeg,
        clips=mp4s, wavs=wavs, seam=seam,
        width=width, height=height,
        out_mp4=out_mp4, out_wav=out_wav,
        quiet=args.quiet,
    )
    refs.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    sources.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    return {
        "sample_id": sample_id,
        "status": "built",
        "out_mp4": str(out_mp4),
        "width": width,
        "height": height,
        "gt": gt_lines,
    }


def main() -> int:
    args = parse_args()
    _, samples = load_samples(args.samples)
    if args.sample_id is not None:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        if not samples:
            print(f"ERROR: sample_id {args.sample_id!r} not found in {args.samples}", file=sys.stderr)
            return 2
    if shutil.which(args.ffmpeg) is None:
        print(f"ERROR: ffmpeg not on PATH: {args.ffmpeg}", file=sys.stderr)
        return 2

    args.work_root.mkdir(parents=True, exist_ok=True)
    args.hf_cache.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    n_built = n_exists = n_failed = 0
    for s in samples:
        try:
            res = process_sample(s, args=args, token=token)
        except Exception as e:
            n_failed += 1
            print(f"[{s['sample_id']}] FAILED: {e}", file=sys.stderr)
            continue
        if res["status"] == "built":
            n_built += 1
            print(f"[{res['sample_id']}] built {res['out_mp4']} ({res['width']}x{res['height']})")
        else:
            n_exists += 1
    print()
    print(f"Done. built={n_built} exists={n_exists} failed={n_failed} total={len(samples)}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
