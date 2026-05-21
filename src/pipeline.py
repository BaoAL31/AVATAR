import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import torch

from src.diarization.run_av_diarization import Diarizer
from src.preprocess.mouth_crop import MouthCropper

DATA_DIR_DEFAULT = "./data"


def _safe_default_device() -> torch.device:
    """Probe CUDA without aborting the process.

    On WSL hosts with a partial CUDA install (driver visible to torch but
    ``libcuda.so`` not resolvable) ``torch.cuda.is_available()`` itself
    has been observed to crash with "free(): double free detected" in
    torch 2.0.x. Honor ``AVATAR_FORCE_CPU`` to skip the probe entirely.
    """
    if os.environ.get("AVATAR_FORCE_CPU"):
        return torch.device("cpu")
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    return torch.device("cuda" if cuda_ok else "cpu")


DEVICE_DEFAULT = _safe_default_device()

# Repo layout (this file lives at <repo>/src/pipeline.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_USR_DIR = _REPO_ROOT / "models" / "usr"


def _resolve_usr_python() -> str:
    """USR has its own Python environment (espnet/fairseq/hydra stack pinned to
    torch 2.0.x). Resolution order:
    1. ``$AVATAR_USR_PYTHON`` env var (full path to python binary).
    2. Conda env named ``usr_env`` under common conda roots on this host.
    3. ``sys.executable`` as a last-resort fallback (may fail at import time).
    """
    override = os.environ.get("AVATAR_USR_PYTHON")
    if override:
        return override
    candidates = [
        Path.home() / "miniconda3" / "envs" / "usr_env" / "bin" / "python",
        Path.home() / "anaconda3"  / "envs" / "usr_env" / "bin" / "python",
        Path("/opt/conda/envs/usr_env/bin/python"),
        Path("/opt/miniconda3/envs/usr_env/bin/python"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return sys.executable


USR_PYTHON = _resolve_usr_python()
RUN_USR_SCRIPT = str(_REPO_ROOT / "src" / "transcription" / "run_usr.py")


class Pipeline:
    def __init__(self, video_path: str, output_dir: str = None, device: torch.device = DEVICE_DEFAULT,
                 visualize: bool = False, ckpt_path: str | None = None,
                 au_concat_path: str | None = None):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = output_dir or os.path.join(os.path.dirname(video_path), "outputs", self.video_name)
        self.device = device
        self.visualize = visualize
        self.ckpt_path = ckpt_path
        self.au_concat_path = au_concat_path
        self.paths = self._get_paths()

    def _get_paths(self) -> dict:
        return {
            "raw_video_path": self.video_path,
            "cache":          os.path.join(self.output_dir, "cache"),
            "mouth_crops":    os.path.join(self.output_dir, "cache", "mouth_crops"),
            "pycrop":         os.path.join(self.output_dir, "cache", "pycrop"),
            "pycrop_wav":     os.path.join(self.output_dir, "cache", "pycrop_wav"),
            "rttm":           os.path.join(self.output_dir, "cache", "result.rttm"),
            "rttm_tracks":    os.path.join(self.output_dir, "cache", "rttm_tracks.json"),
            "output":         os.path.join(self.output_dir, "transcript.txt"),
        }

    def _run_usr(self) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        # USR's local imports (`from data.transforms import ...`,
        # `from espnet... import ...`, `from utils.utils import ...`) resolve
        # against packages that live inside `models/usr/`. Pass that as cwd
        # AND prepend it to PYTHONPATH so the subprocess can find them.
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_USR_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        usr_args = [
            USR_PYTHON, RUN_USR_SCRIPT,
            f"--video_name={self.video_name}",
            f"--output_dir={os.path.abspath(self.output_dir)}",
            f"--tmp_output={tmp_path}",
        ]
        if self.ckpt_path:
            usr_args.append(f"--ckpt_path={os.path.abspath(self.ckpt_path)}")
        if self.au_concat_path:
            usr_args.append(f"--au_concat_path={os.path.abspath(self.au_concat_path)}")
        rc = subprocess.call(
            usr_args,
            cwd=str(_USR_DIR),
            env=env,
        )
        if rc != 0:
            print(f"[pipeline] USR subprocess exited with code {rc}")
        with open(tmp_path) as f:
            results = json.load(f)
        os.remove(tmp_path)
        return results

    def _get_attributed_transcript(self, transcriptions: dict) -> list:
        with open(self.paths["rttm_tracks"]) as f:
            rttm_tracks = json.load(f)
        if not rttm_tracks and transcriptions:
            # Fallback when diarization produced no track mappings.
            # Keep final output non-empty by emitting one unknown speaker segment.
            end = 0.0
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                if fps > 0:
                    end = float(frames / fps)
            cap.release()
            first_text = next(iter(transcriptions.values()), "")
            return [{
                "speaker": "ID_unknown",
                "start": 0.0,
                "end": end,
                "transcription": first_text
            }]
        # ``run_usr`` produces one transcription per *visual mouth track*, not per
        # RTTM speech interval. ``_build_rttm_tracks`` often emits several rows per
        # ``track_idx`` (one per voiced segment); without merging, callers that
        # concatenate segment text would paste the **same full-track transcript**
        # once per RTTM shard (e.g. 3× repetition for fs_000), destroying WER.
        rows = sorted(
            rttm_tracks,
            key=lambda e: (float(e.get("start", 0.0)), int(e["track_idx"])),
        )
        transcript = []
        for entry in rows:
            tidx = int(entry["track_idx"])
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", start))
            text = transcriptions.get(str(tidx), "")
            speaker = entry.get("speaker", "")
            if transcript and transcript[-1].get("_track_idx") == tidx:
                transcript[-1]["end"] = max(transcript[-1]["end"], end)
                continue
            transcript.append({
                "_track_idx": tidx,
                "speaker":       speaker,
                "start":         start,
                "end":           end,
                "transcription": text,
            })
        for seg in transcript:
            seg.pop("_track_idx", None)
        return transcript

    def _format_transcript(self, results: list) -> str:
        '''
        [<start> - <end>] <speaker_id>: <transcription>
        '''
        output = []
        for entry in results:
            output.append(
                f"[{entry['start']:.2f}s - {entry['end']:.2f}s] "
                f"{entry['speaker']}: {entry['transcription']}"
            )
        return "\n".join(output)

    def save_transcript(self, results: list) -> str:
        transcript = self._format_transcript(results)
        with open(self.paths["output"], "w") as f:
            f.write(transcript)
        print(f"Transcript saved to {self.paths['output']}")
        return self.paths["output"]

    # Save as subtitle file for easier visualization in video players
    def save_srt(self, results: list) -> str:
        def to_srt_time(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        output_path = os.path.join(self.output_dir, "transcript.srt")
        with open(output_path, "w") as f:
            for i, entry in enumerate(results, start=1):
                f.write(f"{i}\n")
                f.write(f"{to_srt_time(entry['start'])} --> {to_srt_time(entry['end'])}\n")
                f.write(f"{entry['speaker']}: {entry['transcription']}\n")
                f.write("\n")

        print(f"SRT saved to {output_path}")
        return output_path

    def run(self) -> list:
        print("[1/3] Diarization...")
        diarizer = Diarizer(self.video_path, self.output_dir, self.device)
        diarizer.run(visualize=self.visualize)

        print("[2/3] Mouth crop extraction...")
        cropper = MouthCropper(self.video_name, self.output_dir, self.device)
        cropper.run()

        print("[3/3] Transcription...")
        transcriptions = self._run_usr()

        print("Assembling transcript...")
        attributed_transcript = self._get_attributed_transcript(transcriptions)

        self.save_transcript(attributed_transcript)
        self.save_srt(attributed_transcript)
        print("Pipeline complete.")
        return attributed_transcript