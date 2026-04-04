from src.diarization.run_av_diarization import Diarizer
import torch
import os

DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    video_name = "-FaXLcSFjUI_trimmed"
    video_path = f"/home/jembo/AVATAR/data/raw/{video_name}.mp4"
    output_dir = f"/home/jembo/AVATAR/data/processed/{video_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    diarizer = Diarizer(video_path, output_dir, device=DEVICE_DEFAULT)
    diarizer.run(visualize=True)
