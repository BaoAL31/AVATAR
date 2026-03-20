import subprocess

from src.diarization.run_diarization import diarize
from src.preprocess.mouth_crop import process_video

USR_PYTHON = "/home/jembo/miniconda3/envs/usr_env/bin/python"
RUN_USR_SCRIPT = "/home/jembo/AVATAR/src/transcription/run_usr.py"

def run_usr(video_name, track_idx, output_path):
    subprocess.call([
        USR_PYTHON, RUN_USR_SCRIPT,
        "--video_name", video_name,
        "--track_idx", str(track_idx),
        "--output", output_path
    ])

def run_pipeline(video_name):