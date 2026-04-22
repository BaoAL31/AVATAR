import os
import json
import subprocess
import tempfile
import torch

from src.diarization.run_av_diarization import Diarizer
from src.preprocess.mouth_crop import MouthCropper
DATA_DIR_DEFAULT = "./data"
DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USR_PYTHON = "/home/hoangbng/miniconda3/envs/usr_env/bin/python"
RUN_USR_SCRIPT = "/home/hoangbng/AVATAR/AVATAR/src/transcription/run_usr.py"


class Pipeline:
    def __init__(self, video_path: str, output_dir: str = None, device: torch.device = DEVICE_DEFAULT, visualize: bool = False):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = output_dir or os.path.join(os.path.dirname(video_path), "outputs", self.video_name)
        self.device = device
        self.visualize = visualize
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
        subprocess.call([
            USR_PYTHON, RUN_USR_SCRIPT,
            f"--video_name={self.video_name}",
            f"--output_dir={self.output_dir}",
            f"--tmp_output={tmp_path}"
        ])
        with open(tmp_path) as f:
            results = json.load(f)
        os.remove(tmp_path)
        return results

    def _get_attributed_transcript(self, transcriptions: dict) -> list:
        with open(self.paths["rttm_tracks"]) as f:
            rttm_tracks = json.load(f)
        transcript = []
        for entry in rttm_tracks:
            track_idx = str(entry["track_idx"])
            transcript.append({
                "speaker":       entry["speaker"],
                "start":         entry["start"],
                "end":           entry["end"],
                "transcription": transcriptions.get(track_idx, "")
            })
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