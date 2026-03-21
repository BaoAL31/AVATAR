from voxconverse.avdiarizer import AVDiarizer
import os
import argparse
import torch
import pickle
import json

DATA_DIR = "/home/jembo/AVATAR/data/"
FPS = 25

def get_paths(video_name, data_dir=DATA_DIR):
    base = os.path.join(data_dir, "processed", video_name)
    return {
        "input":       os.path.join(data_dir, "raw", f"{video_name}.mp4"),
        "output":      base,
        "cache":       os.path.join(base, "cache"),
        "pycrop":      os.path.join(base, "cache", "pycrop"),
        "pyframes":    os.path.join(base, "cache", "pyframes"),
        "tracks":      os.path.join(base, "cache", "tracks.pkl"),
        "rttm":        os.path.join(base, "result.rttm"),
        "rttm_tracks": os.path.join(base, "cache", "rttm_tracks.json"),
    }


def parse_rttm(rttm_path: str) -> list:
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append((start, start + duration, speaker))
    return sorted(segments, key=lambda x: x[0])


def find_best_track(segment_start, segment_end, tracks, fps=FPS):
    best_track = None
    best_overlap = 0

    for tidx, track in enumerate(tracks):
        track_start = track['track']['frame'][0] / fps
        track_end = track['track']['frame'][-1] / fps

        overlap = min(segment_end, track_end) - max(segment_start, track_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_track = tidx

    return best_track


def build_rttm_tracks(rttm_path: str, tracks: list, faceidx: list, rttm_tracks_path: str) -> list:
    segments = parse_rttm(rttm_path)
    result = []

    for start, end, speaker in segments:
        if speaker == "unknown":
            continue

        # handle merged speakers like "ID_0/1"
        try:
            face_cluster_ids = [int(x) for x in speaker.replace("ID_", "").split("/")]
        except ValueError:
            continue

        for face_cluster_id in face_cluster_ids:
            candidate_tracks = [tidx for tidx, fid in enumerate(faceidx) if fid == face_cluster_id]

            best_track = None
            best_overlap = 0
            for tidx in candidate_tracks:
                track_start = tracks[tidx]['track']['frame'][0] / FPS
                track_end = tracks[tidx]['track']['frame'][-1] / FPS
                overlap = min(end, track_end) - max(start, track_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_track = tidx

            if best_track is not None:
                result.append({
                    "track_idx": best_track,
                    "start": start,
                    "end": end,
                    "speaker": f"ID_{face_cluster_id}"
                })

    with open(rttm_tracks_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(result)} RTTM track mappings to {rttm_tracks_path}")
    return result


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

    with open(paths["tracks"], "rb") as f:
        tracks = pickle.load(f)

    with open(os.path.join(paths["cache"], "faceidx.pkl"), "rb") as f:
        faceidx = pickle.load(f)

    build_rttm_tracks(paths["rttm"], tracks, faceidx, paths["rttm_tracks"])