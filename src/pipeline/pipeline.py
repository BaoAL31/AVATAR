import subprocess

from src.diarization.run_diarization import diarize
from src.preprocess.mouth_crop import process_video

DATA_DIR = "./data/processed"
USR_PYTHON = "/home/jembo/miniconda3/envs/usr_env/bin/python"
RUN_USR_SCRIPT = "/home/jembo/AVATAR/src/transcription/run_usr.py"

def get_paths(video_name, data_dir=DATA_DIR):
    base = os.path.join(data_dir, f"{video_name}")
    return {
        "mouth_crops": os.path.join(base, "mouth_crops"),
        "pycrop":      os.path.join(base, "cache", "pycrop"),
        "pycrop_wav":  os.path.join(base, "cache", "pycrop_wav"),
        "rttm":        os.path.join(base, f"{video_name}.rttm"),
    }

def run_usr(video_name, track_idx):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.call([
        USR_PYTHON, RUN_USR_SCRIPT,
        "--video_name", video_name,
        "--track_idx", str(track_idx),
        "--tmp_output", tmp_path
    ])

    with open(tmp_path) as f:
        result = json.load(f)["transcription"]

    os.remove(tmp_path)
    return result

def run_pipeline(video_path):
    video_name = os.path.basename(video_path).split(".")[0]
    paths = get_paths(video_name)

    # Step 1: Diarization
    diarize(video_name)

    # Step 2: Mouth Crop
    process_video(video_name)

    # Step 3: USR Transcription
    