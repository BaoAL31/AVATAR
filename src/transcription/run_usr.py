import os
import json
import argparse
import cv2
import numpy as np
import torch
import torchaudio
from torchvision.transforms import CenterCrop, Grayscale

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from data.transforms import NormalizeVideo
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from utils.utils import UNIGRAM1000_LIST

CKPT_PATH = "/home/hoangbng/AVATAR/AVATAR/models/usr/checkpoints/baseplus_high_resource_lrs3vox2.pth"
USR_CONF  = "/home/hoangbng/AVATAR/AVATAR/models/usr/conf"


class Transcriber:
    def __init__(self, video_name: str, output_dir: str, ckpt_path: str = CKPT_PATH):
        self.video_name = video_name
        self.output_dir = output_dir
        self.ckpt_path  = ckpt_path
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths      = self._get_paths()
        self.model, self.cfg = self._load_model()

    def _get_paths(self) -> dict:
        return {
            "mouth_crops": os.path.join(self.output_dir, "cache", "mouth_crops"),
            "pycrop_wav":  os.path.join(self.output_dir, "cache", "pycrop_wav"),
            "rttm_tracks": os.path.join(self.output_dir, "cache", "rttm_tracks.json"),
        }

    def _load_model(self):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=USR_CONF):
            cfg = compose(config_name='config', overrides=[
                'experiment_name=test',
                'model/backbone=resnet_transformer_baseplus'
            ])
        state_dict = torch.load(self.ckpt_path, map_location=self.device)
        state_dict = {k.replace('_orig_mod.model.backbone.', ''): v for k, v in state_dict.items()}
        model = E2E(1049, cfg.model.backbone)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        return model, cfg

    def _video_to_tensor(self, path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        frames = torch.from_numpy(np.stack(frames))
        return frames.permute((3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)

    def _audio_to_tensor(self, path: str) -> torch.Tensor:
        audio, _ = torchaudio.load(path, normalize=True)
        return audio

    def _load_and_preprocess_track(self, track_idx: int):
        mouth_avi = os.path.join(self.paths["mouth_crops"], "%05d.avi" % track_idx)
        audio_wav = os.path.join(self.paths["pycrop_wav"],  "%05d.wav" % track_idx)

        mouth_tensor = self._video_to_tensor(mouth_avi)
        mouth_tensor = mouth_tensor / 255.
        mouth_tensor = CenterCrop(88)(mouth_tensor)
        mouth_tensor = mouth_tensor.transpose(0, 1)   # (C, T, H, W) -> (T, C, H, W) for Grayscale
        mouth_tensor = Grayscale()(mouth_tensor)
        mouth_tensor = mouth_tensor.transpose(0, 1)   # (T, C, H, W) -> (C, T, H, W)
        mouth_tensor = NormalizeVideo(mean=(0.421,), std=(0.165,))(mouth_tensor)

        audio_tensor = self._audio_to_tensor(audio_wav)
        return mouth_tensor, audio_tensor

    def _get_beam_search(self) -> BatchBeamSearch:
        token_list = UNIGRAM1000_LIST
        odim = len(token_list)
        scorers = self.model.scorers()
        scorers["lm"] = None
        scorers["length_bonus"] = LengthBonus(odim)
        weights = dict(
            decoder=1.0 - self.cfg.decode.ctc_weight,
            ctc=self.cfg.decode.ctc_weight,
            lm=self.cfg.decode.lm_weight,
            length_bonus=self.cfg.decode.penalty,
        )
        return BatchBeamSearch(
            beam_size=self.cfg.decode.beam_size,
            vocab_size=odim,
            weights=weights,
            scorers=scorers,
            sos=odim - 1,
            eos=odim - 1,
            token_list=token_list,
            pre_beam_score_key=None if self.cfg.decode.ctc_weight == 1.0 else "decoder",
        )

    def _transcribe_track(self, track_idx: int, modality: str = "av") -> str:
        video, audio = self._load_and_preprocess_track(track_idx)
        video = video.to(self.device)
        audio = audio.to(self.device)

        beam_search = self._get_beam_search()

        with torch.no_grad():
            feat, _, _ = self.model.encoder.forward_single(
                xs_v=video,
                xs_a=audio.unsqueeze(0).transpose(1, 2)
            )
            nbest_hyps = beam_search(
                x=feat.squeeze(0),
                modality=modality,
                maxlenratio=self.cfg.decode.maxlenratio,
                minlenratio=self.cfg.decode.minlenratio
            )

        nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]
        transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
        return transcription.replace("<eos>", "").replace("▁", " ").strip()

    def run(self) -> dict:
        tracks = sorted([f for f in os.listdir(self.paths["mouth_crops"]) if f.endswith(".avi")])
        results = {}
        for track_avi in tracks:
            track_idx = int(os.path.splitext(track_avi)[0])
            print(f"Transcribing track {track_idx}...")
            results[str(track_idx)] = self._transcribe_track(track_idx)
        return results

# Work around due to dependency issues with USR and the main pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_path",  type=str, default=CKPT_PATH)
    parser.add_argument("--tmp_output", type=str, required=True)
    args = parser.parse_args()

    transcriber = Transcriber(args.video_name, args.output_dir, args.ckpt_path)
    results = transcriber.run()

    with open(args.tmp_output, "w") as f:
        json.dump(results, f)