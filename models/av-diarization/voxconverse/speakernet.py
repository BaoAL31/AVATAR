#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional

try:
    # speechbrain >= 1.0
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    # speechbrain < 1.0 (legacy path; required for torch 2.0.x compatibility
    # because speechbrain 1.x has a double-free crash with torch <2.1).
    from speechbrain.pretrained import EncoderClassifier
import torch
import torch.nn as nn
import torchaudio

from .models.resnetse34v2 import ResNetSE34V2
from .utils import load_checkpoint


class SpeakerNet(nn.Module):
    def __init__(self, 
                 cache_dir: str, 
                 ckpt_dir : Optional[str] = None, 
                 model_type: str = 'ecapa-tdnn', 
                 device: torch.device = torch.device('cpu'),
                 max_frames: int = 200):
        super(SpeakerNet, self).__init__()
        self.cache_dir = cache_dir
        self.ckpt_dir = ckpt_dir
        self.work_dir = os.path.join(cache_dir, 'pywork')
        self.avi_dir = os.path.join(cache_dir, 'pyavi')
        self.device = device
        self.model_type = model_type

        assert model_type in ['ecapa-tdnn', 'resnetse34'], \
            f'Model type {model_type} not supported'
        if self.model_type == 'ecapa-tdnn':
            hyperparameters = {'device': self.device.type}
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir=self.ckpt_dir,
                run_opts=hyperparameters
            )
        else:
            self.model = ResNetSE34V2(
                nOut=512, encoder_type="ASP", n_mels=64, log_input=True
            )
            self.model.load_state_dict(
                load_checkpoint("speakernet", download_root = self.ckpt_dir, device = device)
            )
            self.model.to(self.device)
        self.model.eval()
        self.max_frames = max_frames
    
    def _get_chunk_starts(self, num_samples: int) -> list:
        """Return chunk start indices; always yields at least one chunk."""
        window = self.max_frames * 160
        if num_samples <= window:
            return [0]
        return list(range(0, num_samples - window, 3200))
    
    @torch.no_grad()
    def run_resnetse34(self, fname: str) -> torch.Tensor:
        inp1, fs = torchaudio.load(fname)
        feats = []
        window = self.max_frames * 160
        for ii in self._get_chunk_starts(inp1.size()[-1]):
            chunk = inp1[:, ii:ii + window]
            if chunk.size(-1) < window:
                chunk = nn.functional.pad(chunk, (0, window - chunk.size(-1)))
            feats.append(
                self.model.forward(
                    chunk.to(self.device)
                ).detach().cpu()
            )
        return feats

    @torch.no_grad()
    def run_ecapa(self, fname: str) -> torch.Tensor:
        inp1, fs = torchaudio.load(fname)
        feats = []
        window = self.max_frames * 160
        for ii in self._get_chunk_starts(inp1.size()[-1]):
            chunk = inp1[:, ii:ii + window]
            if chunk.size(-1) < window:
                chunk = nn.functional.pad(chunk, (0, window - chunk.size(-1)))
            x = self.model.encode_batch(
                chunk.to(self.device)
            ).detach().cpu()
            feats.append(torch.squeeze(x, dim=1))
        return feats

    @torch.no_grad()
    def run(self) -> torch.Tensor:
        """
        Run speaker embedding extraction

        Returns:
            torch.Tensor, speaker features
        """
        logging.info("Running speaker embedding extraction...")
        filename = os.path.join(self.avi_dir, 'audio.wav')

        if self.model_type == 'ecapa-tdnn':
            logging.info("Running ECAPA-TDNN...")
            feats = self.run_ecapa(filename)
        else:
            logging.info("Running ResNetSE34...")
            feats = self.run_resnetse34(filename)
        if not feats:
            raise RuntimeError(f"No speaker features extracted from {filename}")
        feats = torch.cat(feats, dim=0)
        return feats


if __name__ == '__main__':
    cache_dir = '/users/jaesung/voxconverse_method/temp'
    model_type = 'resnetse34'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    speakernet = SpeakerNet(cache_dir=cache_dir, model_type=model_type, device=device)
    feats = speakernet.run()
    torch.save(feats, os.path.join(cache_dir, 'pywork', 'resnet.pt'))