#!/usr/bin/env python3
"""Smoke-test finetuning configs: compose Hydra, build SSLLearner, assert encoder adim vs checkpoint."""
from __future__ import annotations

import os
import sys

# Run from models/usr (same as main_ft.py).
USR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != USR_ROOT:
    os.chdir(USR_ROOT)
sys.path.insert(0, USR_ROOT)

import torch
from hydra import compose, initialize_config_dir
from learner_ft import SSLLearner
from utils.hf_env import ensure_hf_env
from utils.hf_paths import register_hf_hydra_resolvers


def _ckpt_adim(path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    k = next(x for x in ckpt if x.endswith("encoder.encoders.0.gamma_ff"))
    return int(ckpt[k].shape[0])


def main() -> None:
    ensure_hf_env()
    register_hf_hydra_resolvers()
    with initialize_config_dir(config_dir=os.path.join(USR_ROOT, "conf")):
        cases = [
            (
                "config_ft_lrs2_lora_base_high_lrs3",
                512,
                "/home/hoangbng/AVATAR/AVATAR/models/usr/checkpoints/base_high_resource_lrs3.pth",
            ),
            (
                "config_ft_lrs2_lora_baseplus_high_lrs3vox2",
                768,
                "/home/hoangbng/AVATAR/AVATAR/models/usr/checkpoints/baseplus_high_resource_lrs3vox2.pth",
            ),
        ]
        for name, want_adim, ckpt_path in cases:
            cfg = compose(config_name=name)
            got_adim = int(cfg.model.backbone.adim)
            ck_adim = _ckpt_adim(ckpt_path)
            print(f"\n=== {name} ===")
            print(f"  cfg.model.backbone.adim     = {got_adim}")
            print(f"  checkpoint encoder adim     = {ck_adim}")
            assert got_adim == want_adim, f"expected backbone adim {want_adim}, got {got_adim}"
            assert ck_adim == want_adim, f"checkpoint geometry {ck_adim} != expected {want_adim}"

            learner = SSLLearner(cfg)
            g = learner.model.encoder.encoders[0].gamma_ff
            assert g.shape[0] == want_adim, f"model gamma_ff {g.shape} vs want {want_adim}"
            print(f"  SSLLearner built OK, gamma_ff shape = {tuple(g.shape)}")
            del learner

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
