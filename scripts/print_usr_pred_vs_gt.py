#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from huggingface_hub import hf_hub_download
from torchvision.transforms import CenterCrop, Grayscale

from data.transforms import NormalizeVideo
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from utils.utils import UNIGRAM1000_LIST, ids_to_str


DEFAULT_MANIFEST = Path("/home/hoangbng/AVATAR/AVATAR/data/val_manifest.csv")
DEFAULT_CKPT = Path("/home/hoangbng/AVATAR/AVATAR/models/usr/checkpoints/baseplus_high_resource_lrs3vox2.pth")
DEFAULT_USR_CONF = Path("/home/hoangbng/AVATAR/AVATAR/models/usr/conf")


def parse_args():
    ap = argparse.ArgumentParser(description="Print USR prediction vs GT for one manifest row.")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--row", type=int, default=1, help="1-based row number in manifest")
    ap.add_argument("--repo-prefix", type=str, default="HBaoAL/LRS2")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--modality", type=str, default="av", choices=["a", "v", "av"])
    return ap.parse_args()


def repo_from_tag(tag: str, repo_prefix: str) -> str:
    m = re.match(r"^lrs2_(\d{2})$", tag)
    if m:
        return f"{repo_prefix}_{m.group(1)}"
    if tag == "lrs2":
        return repo_prefix
    raise ValueError("Unsupported tag: %s" % tag)


def load_manifest_row(path: Path, row_num_1based: int):
    if row_num_1based < 1:
        raise ValueError("--row must be >= 1")
    with path.open("r", encoding="utf-8") as f:
        rows = [ln.strip() for ln in f if ln.strip()]
    if row_num_1based > len(rows):
        raise ValueError("row out of range: %d > %d" % (row_num_1based, len(rows)))
    parts = rows[row_num_1based - 1].split(",", 3)
    if len(parts) != 4:
        raise ValueError("malformed row: %s" % rows[row_num_1based - 1])
    tag = parts[0].strip()
    rel = parts[1].strip().replace("\\", "/")
    ids = [int(x) for x in parts[3].strip().split()] if parts[3].strip() else []
    return tag, rel, ids


def normalize_ckpt(state):
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return state
    out = {}
    for k, v in state.items():
        nk = k
        if isinstance(nk, str):
            for p in ("_orig_mod.", "model.backbone.", "module."):
                if nk.startswith(p):
                    nk = nk[len(p):]
        out[nk] = v
    return out


def load_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("No frames in video: %s" % path)
    x = torch.from_numpy(np.stack(frames)).permute((3, 0, 1, 2)).float() / 255.0
    x = CenterCrop(88)(x)
    x = x.transpose(0, 1)
    x = Grayscale()(x)
    x = x.transpose(0, 1)
    x = NormalizeVideo(mean=(0.421,), std=(0.165,))(x)
    return x


def load_audio(path: str):
    wav, _ = torchaudio.load(path, normalize=True)
    return wav


def build_model(ckpt_path: Path, device: torch.device):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(DEFAULT_USR_CONF)):
        cfg = compose(config_name="config", overrides=["experiment_name=test", "model/backbone=resnet_transformer_baseplus"])
    model = E2E(1049, cfg.model.backbone)
    state = torch.load(str(ckpt_path), map_location=device)
    state = normalize_ckpt(state)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, cfg


def build_beam_search(model, cfg):
    token_list = UNIGRAM1000_LIST
    odim = len(token_list)
    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(odim)
    weights = dict(
        decoder=1.0 - cfg.decode.ctc_weight,
        ctc=cfg.decode.ctc_weight,
        lm=cfg.decode.lm_weight,
        length_bonus=cfg.decode.penalty,
    )
    return BatchBeamSearch(
        beam_size=cfg.decode.beam_size,
        vocab_size=odim,
        weights=weights,
        scorers=scorers,
        sos=odim - 1,
        eos=odim - 1,
        token_list=token_list,
        pre_beam_score_key=None if cfg.decode.ctc_weight == 1.0 else "decoder",
    )


def main():
    args = parse_args()
    tag, rel_video, ids = load_manifest_row(args.manifest.resolve(), args.row)
    stem = Path(rel_video).stem
    folder = Path(rel_video).parent.as_posix()
    repo_id = repo_from_tag(tag, args.repo_prefix)
    rel_audio = f"{folder}/{stem}.wav"
    rel_txt = f"{folder}/{stem}.txt"

    local_video = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_video)
    local_audio = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_audio)
    local_txt = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_txt)

    gt_from_ids = ids_to_str(ids, UNIGRAM1000_LIST).replace("▁", " ").replace("<eos>", "").strip()
    txt_text = ""
    with open(local_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("text:"):
                txt_text = s.split(":", 1)[1].strip()
                break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(args.ckpt.resolve(), device)
    beam = build_beam_search(model, cfg)

    video = load_video(local_video).to(device)
    audio = load_audio(local_audio).to(device)

    with torch.no_grad():
        if args.modality == "v":
            feat, _, _ = model.encoder.forward_single(xs_v=video)
        elif args.modality == "a":
            feat, _, _ = model.encoder.forward_single(xs_a=audio.unsqueeze(0).transpose(1, 2))
        else:
            feat, _, _ = model.encoder.forward_single(xs_v=video, xs_a=audio.unsqueeze(0).transpose(1, 2))
        nbest = beam(
            x=feat.squeeze(0),
            modality=args.modality,
            maxlenratio=cfg.decode.maxlenratio,
            minlenratio=cfg.decode.minlenratio,
        )
    pred = add_results_to_json([nbest[0].asdict()], UNIGRAM1000_LIST).replace("<eos>", "").replace("▁", " ").strip()

    print("repo:", repo_id)
    print("row:", args.row)
    print("video:", rel_video)
    print("audio:", rel_audio)
    print("modality:", args.modality)
    print("GT(ids):", gt_from_ids)
    print("GT(txt):", txt_text)
    print("PRED:", pred)


if __name__ == "__main__":
    main()

