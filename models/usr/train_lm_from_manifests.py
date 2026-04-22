#!/usr/bin/env python3
"""
Train a USR-compatible TransformerLM from manifest label-id sequences.

Inputs are CSV manifests whose 4th column is token ids, e.g.:
  tag,relpath.mp4,num_frames,"610 262 955 537 ... 1"

This script trains next-token prediction on those ids and writes a checkpoint
that can be loaded by espnet.asr.asr_utils.torch_load(..., TransformerLM).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.pytorch_backend.lm.transformer import TransformerLM
from utils.utils import UNIGRAM1000_LIST


def read_sequences(path: Path, eos_id: int) -> list[torch.Tensor]:
    seqs: list[torch.Tensor] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 3)
            if len(parts) < 4:
                continue
            labels = parts[3].strip()
            if not labels:
                continue
            try:
                ids = [int(x) for x in labels.split()]
            except ValueError:
                continue
            if not ids:
                continue
            if ids[-1] != eos_id:
                ids.append(eos_id)
            if len(ids) < 2:
                continue
            seqs.append(torch.tensor(ids, dtype=torch.long))
    return seqs


def make_batch(seqs: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = [s[:-1] for s in seqs]
    t = [s[1:] for s in seqs]
    x_pad = pad_sequence(x, batch_first=True, padding_value=0).to(device)
    t_pad = pad_sequence(t, batch_first=True, padding_value=0).to(device)
    return x_pad, t_pad


def iterate_minibatches(
    seqs: list[torch.Tensor], batch_size: int, shuffle: bool
) -> list[list[torch.Tensor]]:
    idx = list(range(len(seqs)))
    if shuffle:
        random.shuffle(idx)
    return [[seqs[i] for i in idx[s : s + batch_size]] for s in range(0, len(idx), batch_size)]


def build_lm(args: argparse.Namespace, vocab_size: int, device: torch.device) -> TransformerLM:
    lm_cfg = SimpleNamespace(
        layer=args.layer,
        unit=args.unit,
        att_unit=args.att_unit,
        embed_unit=args.embed_unit,
        head=args.head,
        dropout_rate=args.dropout_rate,
        att_dropout_rate=args.att_dropout_rate,
        emb_dropout_rate=args.emb_dropout_rate,
        tie_weights=args.tie_weights,
        pos_enc=args.pos_enc,
    )
    lm = TransformerLM(vocab_size, lm_cfg).to(device)
    return lm


def eval_epoch(
    lm: TransformerLM, seqs: list[torch.Tensor], batch_size: int, device: torch.device
) -> float:
    lm.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in iterate_minibatches(seqs, batch_size, shuffle=False):
            x_pad, t_pad = make_batch(batch, device)
            loss = lm_loss(lm, x_pad, t_pad)
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def lm_loss(lm: TransformerLM, x_pad: torch.Tensor, t_pad: torch.Tensor) -> torch.Tensor:
    """Compute LM CE loss robustly across local ESPnet return signatures."""
    xm = x_pad != 0
    emb = lm.embed(x_pad)
    if lm.embed_drop is not None:
        emb = lm.embed_drop(emb)

    enc_out = lm.encoder(emb, lm._target_mask(x_pad))
    if isinstance(enc_out, tuple):
        h = enc_out[0]
    else:
        h = enc_out
    y = lm.decoder(h)
    token_loss = F.cross_entropy(
        y.view(-1, y.shape[-1]), t_pad.view(-1), reduction="none"
    )
    mask = xm.to(dtype=token_loss.dtype).view(-1)
    denom = mask.sum().clamp_min(1.0)
    return (token_loss * mask).sum() / denom


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TransformerLM for USR decoding.")
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--val-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="Output .pth path (state_dict)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0, help="Reserved for future use.")
    ap.add_argument("--device", type=str, default="cuda")
    # LM architecture defaults aligned with conf/model/language_model/default.yaml
    ap.add_argument("--pos-enc", type=str, default="none", choices=["none", "sinusoidal"])
    ap.add_argument("--embed-unit", type=int, default=128)
    ap.add_argument("--att-unit", type=int, default=512)
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--unit", type=int, default=2048)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--dropout-rate", type=float, default=0.0)
    ap.add_argument("--att-dropout-rate", type=float, default=0.0)
    ap.add_argument("--emb-dropout-rate", type=float, default=0.0)
    ap.add_argument("--tie-weights", action="store_true")
    args = ap.parse_args()

    del args.num_workers  # keep CLI compatibility if you script around this arg

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available. Use --device cpu.")
    device = torch.device(args.device)

    vocab_size = len(UNIGRAM1000_LIST)
    eos_id = vocab_size - 1

    train = read_sequences(args.train_manifest, eos_id)
    val = read_sequences(args.val_manifest, eos_id)
    if not train:
        raise SystemExit(f"No train sequences loaded from {args.train_manifest}")
    if not val:
        raise SystemExit(f"No val sequences loaded from {args.val_manifest}")

    print(f"Loaded sequences | train={len(train)} val={len(val)} vocab={vocab_size}")
    lm = build_lm(args, vocab_size, device)
    optim = torch.optim.AdamW(lm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        lm.train()
        running = 0.0
        steps = 0
        for batch in iterate_minibatches(train, args.batch_size, shuffle=True):
            x_pad, t_pad = make_batch(batch, device)
            loss = lm_loss(lm, x_pad, t_pad)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 5.0)
            optim.step()
            running += float(loss.item())
            steps += 1

        train_loss = running / max(steps, 1)
        val_loss = eval_epoch(lm, val, args.batch_size, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(lm.state_dict(), args.output)
            print(f"  Saved best LM checkpoint -> {args.output}")

    print(f"Done. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
