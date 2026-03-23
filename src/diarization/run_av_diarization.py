from voxconverse.avdiarizer import AVDiarizer
import os
import argparse
import torch
import pickle
import json


class Diarizer:
    def __init__(self, video_path: str, output_dir: str, device: torch.device):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = output_dir
        self.device = device
        self.fps = 25
        self.paths = self._get_paths()

    def _get_paths(self) -> dict:
        return {
            "input":       self.video_path,
            "output":      self.output_dir,
            "cache":       os.path.join(self.output_dir, "cache"),
            "pycrop":      os.path.join(self.output_dir, "cache", "pycrop"),
            "pyframes":    os.path.join(self.output_dir, "cache", "pyframes"),
            "tracks":      os.path.join(self.output_dir, "cache", "tracks.pkl"),
            "faceidx":     os.path.join(self.output_dir, "cache", "faceidx.pkl"),
            "rttm":        os.path.join(self.output_dir, "cache", "result.rttm"),
            "rttm_tracks": os.path.join(self.output_dir, "cache", "rttm_tracks.json"),
        }

    def _parse_rttm(self) -> list:
        segments = []
        with open(self.paths["rttm"]) as f:
            for line in f:
                parts = line.strip().split()
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append((start, start + duration, speaker))
        return sorted(segments, key=lambda x: x[0])

    def _find_best_track(self, segment_start: float, segment_end: float, tracks: list) -> int:
        best_track = None
        best_overlap = 0
        for tidx, track in enumerate(tracks):
            track_start = track['track']['frame'][0] / self.fps
            track_end = track['track']['frame'][-1] / self.fps
            overlap = min(segment_end, track_end) - max(segment_start, track_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_track = tidx
        return best_track

    def _build_rttm_tracks(self, tracks: list, faceidx: list) -> list:
        segments = self._parse_rttm()
        result = []

        for start, end, speaker in segments:
            if speaker == "unknown":
                continue
            try:
                face_cluster_ids = [int(x) for x in speaker.replace("ID_", "").split("/")]
            except ValueError:
                continue

            for face_cluster_id in face_cluster_ids:
                candidate_tracks = [tidx for tidx, fid in enumerate(faceidx) if fid == face_cluster_id]

                best_track = None
                best_overlap = 0
                for tidx in candidate_tracks:
                    track_start = tracks[tidx]['track']['frame'][0] / self.fps
                    track_end = tracks[tidx]['track']['frame'][-1] / self.fps
                    overlap = min(end, track_end) - max(start, track_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_track = tidx

                if best_track is not None:
                    result.append({
                        "track_idx": best_track,
                        "start":     start,
                        "end":       end,
                        "speaker":   f"ID_{face_cluster_id}"
                    })

        with open(self.paths["rttm_tracks"], "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved {len(result)} RTTM track mappings to {self.paths['rttm_tracks']}")
        return result

    def run(self, visualize: bool = False) -> list:
        os.makedirs(self.paths["output"], exist_ok=True)
        os.makedirs(self.paths["cache"], exist_ok=True)

        args = argparse.Namespace(
            input=self.paths["input"],
            out_dir=self.paths["cache"],
            cache_dir=self.paths["cache"],
            ckpt_dir=None,
            visualize=visualize,
            vad='silero',
            speaker_model='ecapa-tdnn'
        )

        device = self.device
        diarizer = AVDiarizer(args)
        diarizer.run(self.paths["input"], self.paths["cache"], device,
                     self.paths["cache"], args.visualize)

        with open(self.paths["tracks"], "rb") as f:
            tracks = pickle.load(f)
        with open(self.paths["faceidx"], "rb") as f:
            faceidx = pickle.load(f)

        return self._build_rttm_tracks(tracks, faceidx)