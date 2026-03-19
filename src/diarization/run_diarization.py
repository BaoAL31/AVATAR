from voxconverse.avdiarizer import AVDiarizer
import os
import argparse
import torch

DATA_DIR = "/home/jembo/AVATAR/data/"

def get_paths(video_name, data_dir=DATA_DIR):
    base = os.path.join(data_dir, "processed", f"{video_name}_av")
    return {
        "input":     os.path.join(data_dir, "raw", f"{video_name}.mp4"),
        "output":    base,
        "cache":     os.path.join(base, "cache"),
        "pycrop":    os.path.join(base, "cache", "pycrop"),
        "pyframes":  os.path.join(base, "cache", "pyframes"),
    }

def diarize(video_name: str, visualize: bool = False):
    paths = get_paths(video_name)

    os.makedirs(paths["output"], exist_ok=True)
    os.makedirs(paths["cache"], exist_ok=True)

    args = argparse.Namespace(
        input=paths["input"],
        out_dir=paths["output"],
        cache_dir=paths["cache"],
        ckpt_dir=None,
        visualize=visualize,
        vad='silero',
        speaker_model='ecapa-tdnn'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diarizer = AVDiarizer(args)
    diarizer.run(paths["input"], paths["output"], device, paths["cache"], args.visualize)
