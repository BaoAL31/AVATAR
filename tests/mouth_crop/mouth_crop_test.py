import modal
from pathlib import Path
import os

app = modal.App("mouth-crop-test")

REMOTE_DATA_DIR = "/root/data"

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("opencv-python", "torch", "numpy", "face-alignment", "tqdm")
    .add_local_dir(
        Path(__file__).resolve().parent.parent.parent / "src",
        "/root/src"
    )
)

video_name = "-FaXLcSFjUI_trimmed"

data_volume = modal.Volume.from_name("avatar-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={REMOTE_DATA_DIR: data_volume},
    timeout=7200,
    gpu="T4",
)
def run_process_video(video_name: str):
    output_dir = os.path.join(REMOTE_DATA_DIR, "processed", video_name)
    from src.preprocess.mouth_crop import process_video
    process_video(video_name, output_dir=output_dir)

@app.local_entrypoint()
def main():
    run_process_video.remote(video_name)