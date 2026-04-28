from fairseq_manual.data_utils import compute_mask_indices
from hydra.utils import instantiate
import torch
from torch.optim import AdamW
from pytorch_lightning import LightningModule
import random

from schedulers.warmup_cosine import WarmupCosineScheduler

from espnet.asr.asr_utils import add_results_to_json, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.lm.transformer import TransformerLM
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.scorers.length_bonus import LengthBonus
from metrics import WER
from utils.au_npz import AU_FEATURE_DIM
from utils.lora import inject_lora
from utils.utils import ids_to_str, set_requires_grad, UNIGRAM1000_LIST, get_param_groups_ft


def normalize_pretrained_ckpt(ckpt):
    """Normalize wrapper/prefix variants to keys expected by SSLLearner/E2E."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        return ckpt

    prefixes = (
        "model._orig_mod.model.backbone.",
        "_orig_mod.model.backbone.",
        "model.backbone.",
        "model.",
        "_orig_mod.",
        "module.",
    )
    out = {}
    for k, v in ckpt.items():
        nk = k
        if isinstance(nk, str):
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                        changed = True
        out[nk] = v
    return out


def filter_e2e_compatible_ckpt(ckpt):
    """Drop self-supervised-only weights not present on finetuning E2E models."""
    if not isinstance(ckpt, dict):
        return ckpt
    drop_prefixes = ("target_backbone.", "out_layer_unlabelled_", "layer_norm.")
    return {k: v for k, v in ckpt.items() if not k.startswith(drop_prefixes)}


class SSLLearner(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = E2E(1049, cfg.model.backbone)

        if cfg.model.lora.enabled:
            self._enable_lora()

        if cfg.model.pretrained_model_path:
            print("Load pretrained model weights")
            ckpt = torch.load(cfg.model.pretrained_model_path, map_location=lambda storage, loc: storage)
            ckpt = normalize_pretrained_ckpt(ckpt)
            ckpt = filter_e2e_compatible_ckpt(ckpt)

            if cfg.model.transfer_only_encoder:
                ckpt = {k: v for k, v in ckpt.items() if k.startswith("encoder")}
                ckpt = {k[len("encoder.") :]: v for k, v in ckpt.items()}
                ckpt = {k: v for k, v in ckpt.items() if not k.startswith("after_norm")}
                self.model.encoder.load_state_dict(ckpt, strict=False)
            else:
                incompatible = self.model.load_state_dict(ckpt, strict=False)
                if incompatible.missing_keys:
                    print(
                        f"Non-strict checkpoint load: {len(incompatible.missing_keys)} missing keys "
                        f"(e.g. {incompatible.missing_keys[:6]})"
                    )
                if incompatible.unexpected_keys:
                    print(
                        f"Non-strict checkpoint load: {len(incompatible.unexpected_keys)} unexpected keys "
                        f"(e.g. {incompatible.unexpected_keys[:6]})"
                    )

        if getattr(cfg.model.backbone, "use_au", False):
            _au = getattr(cfg.data, "au", None)
            if _au is None or not _au.get("enabled", False):
                print(
                    "Warning: model.backbone.use_au=True but data.au.enabled is not True — "
                    "batches without 'au' will use zeros for AU fusion."
                )

        if cfg.compile_model:
            self.model = torch.compile(self.model)

        if cfg.debug.log_gradients:
            self.logger.experiment.watch(self.model, log="gradients")
        
        self.ignore_id = -1
        self.beam_search_video = self.get_beam_search(self.model)
        self.beam_search_audio = self.get_beam_search(self.model)
        self.beam_search_av = self.get_beam_search(self.model)
        self.wer_video = WER()
        self.wer_audio = WER()
        self.wer_av = WER()
        self.wer_video_val = WER()
        self.wer_audio_val = WER()
        self.wer_av_val = WER()

    def _enable_lora(self):
        lora_cfg = self.cfg.model.lora
        if lora_cfg.freeze_base_model:
            set_requires_grad(self.model, False)

        replaced = inject_lora(
            self.model,
            rank=lora_cfg.rank,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            linear_patterns=lora_cfg.linear_target_patterns,
            conv_patterns=lora_cfg.conv_target_patterns,
        )

        print(
            "LoRA enabled with rank="
            f"{lora_cfg.rank}, alpha={lora_cfg.alpha}, dropout={lora_cfg.dropout}. "
            f"Replaced modules: {replaced}"
        )
        if getattr(self.cfg.model.backbone, "use_au", False) and hasattr(
            self.model.encoder, "au_fusion"
        ):
            for p in self.model.encoder.au_fusion.parameters():
                p.requires_grad = True

    def get_beam_search(self, model):
        token_list = UNIGRAM1000_LIST

        odim = len(token_list)
        self.token_list = token_list

        scorers = model.scorers()

        if self.cfg.decode.lm_weight and self.cfg.model.pretrained_lm_path:
            lm = TransformerLM(len(token_list), self.cfg.model.language_model)
            set_requires_grad(lm, False)
            print("Load pretrained language model weights")
            torch_load(self.cfg.model.pretrained_lm_path, lm)
        else:
            lm = None

        scorers["lm"] = lm
        scorers["length_bonus"] = LengthBonus(len(token_list))

        weights = dict(
            decoder=1.0 - self.cfg.decode.ctc_weight,
            ctc=self.cfg.decode.ctc_weight,
            lm=self.cfg.decode.lm_weight,
            length_bonus=self.cfg.decode.penalty,
        )
        beam_search = BatchBeamSearch(
            beam_size=self.cfg.decode.beam_size,
            vocab_size=len(token_list),
            weights=weights,
            scorers=scorers,
            sos=odim - 1,
            eos=odim - 1,
            token_list=token_list,
            pre_beam_score_key=None if self.cfg.decode.ctc_weight == 1.0 else "decoder",
        )

        return beam_search

    def _au_encoder_kwargs(self, data, video_in):
        """Optional AU tensor (B, T, AU_FEATURE_DIM) for encoder fusion; zeros if missing."""
        if not getattr(self.cfg.model.backbone, "use_au", False):
            return {}
        au = data.get("au")
        if au is None:
            if video_in.dim() == 5:
                b, t = video_in.shape[0], video_in.shape[2]
            else:
                b, t = video_in.shape[0], video_in.shape[1]
            au = video_in.new_zeros(b, t, AU_FEATURE_DIM)
        else:
            au = au.to(video_in.device)
        return {"au": au}

    def get_mask(self, data, padding_mask, mask_prob, mask_length):
        B, C, T, H, W = data["video"].shape
        mask = ~compute_mask_indices(
            (B, T),
            ~padding_mask,
            mask_prob,
            mask_length,
            min_masks=1
        )
        return torch.from_numpy(mask).to(data["video"].device)
            
    def training_step(self, data, batch_idx):
        label = data["label"].squeeze(1)

        video = data["video"].squeeze(1)
        audio = data["audio"].transpose(1, 2)

        padding_mask = make_non_pad_mask(data["video_lengths"]).to(data["video"].device)

        enc_kw = self._au_encoder_kwargs(data, video)
        x_v, x_a, x_av, _, _ = self.model.encoder(
            video, audio, padding_mask.unsqueeze(-2), **enc_kw
        )
        loss_ctc_v = self.model.ctc_v(x_v, padding_mask.sum(-1).squeeze(-1), label)
        loss_ctc_a = self.model.ctc_a(x_a, padding_mask.sum(-1).squeeze(-1), label)
        loss_ctc_av = self.model.ctc_av(x_av, padding_mask.sum(-1).squeeze(-1), label)
        loss_att_v, loss_att_a, loss_att_av, acc_v, acc_a, acc_av = self.model.forward_labelled(
            x_v, x_a, x_av, padding_mask.unsqueeze(-2), label
        )

        self.log("loss_att_v_l", loss_att_v, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_att_a_l", loss_att_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_att_av_l", loss_att_av, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_ctc_v_l", loss_ctc_v, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_ctc_a_l", loss_ctc_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_ctc_av_l", loss_ctc_av, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("acc_v_l", acc_v, on_step=False, on_epoch=True, batch_size=len(label), sync_dist=True)
        self.log("acc_a_l", acc_a, on_step=False, on_epoch=True, batch_size=len(label), sync_dist=True)
        self.log("acc_av_l", acc_av, on_step=False, on_epoch=True, batch_size=len(label), sync_dist=True)

        loss = (1-self.cfg.model.ctc_rel_weight)*self.cfg.model.v_rel_weight*loss_att_v
        loss += (1-self.cfg.model.ctc_rel_weight)*(1-self.cfg.model.v_rel_weight)*loss_att_a
        loss += (1-self.cfg.model.ctc_rel_weight)*(1-self.cfg.model.v_rel_weight)*loss_att_av
        loss += self.cfg.model.ctc_rel_weight*self.cfg.model.v_rel_weight*loss_ctc_v 
        loss += self.cfg.model.ctc_rel_weight*(1-self.cfg.model.v_rel_weight)*loss_ctc_a
        loss += self.cfg.model.ctc_rel_weight*(1-self.cfg.model.v_rel_weight)*loss_ctc_av

        self.log("monitoring_step", float(self.trainer.global_step))  # last-k checkpoints; PL wants float metrics

        return loss     
    
    def shared_val_test_step(self, data):
        video, audio, label = data["video"], data["audio"], data["label"]
        padding_mask_v = make_non_pad_mask(data["video_lengths"]).to(data["video"].device).unsqueeze(-2)

        v_in = video.squeeze(1)
        enc_kw = self._au_encoder_kwargs(data, v_in)
        features_v, features_a, features_av, _, _ = self.model.encoder(
            v_in, audio.transpose(1, 2), padding_mask_v, **enc_kw
        )
        loss_ctc_v = self.model.ctc_v(
            features_v, torch.tensor(data["video_lengths"], device=features_v.device), data["label"].squeeze(1)
        )
        loss_ctc_a = self.model.ctc_a(
            features_a, torch.tensor(data["video_lengths"], device=features_a.device), data["label"].squeeze(1)
        )
        loss_ctc_av = self.model.ctc_av(
            features_av, torch.tensor(data["video_lengths"], device=features_a.device), data["label"].squeeze(1)
        )
        acc_video, acc_audio, acc_av = self.model.forward_labelled(
            features_v, features_a, features_av, padding_mask_v, label
        )[-3:]

        self.log("loss_ctc_v_val", loss_ctc_v, batch_size=len(label), sync_dist=True)
        self.log("loss_ctc_a_val", loss_ctc_a, batch_size=len(label), sync_dist=True)
        self.log("loss_ctc_av_val", loss_ctc_av, batch_size=len(label), sync_dist=True)
        self.log("acc_video_val", acc_video, batch_size=len(label), sync_dist=True)
        self.log("acc_audio_val", acc_audio, batch_size=len(label), sync_dist=True)
        self.log("acc_av_val", acc_av, batch_size=len(label), sync_dist=True)

    def validation_step(self, data, batch_idx):
        self.shared_val_test_step(data)
        lengths = torch.tensor(data["video_lengths"], device=data["video"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device)
        self.calculate_wer(
            data["video"].squeeze(1),
            data["audio"].transpose(1, 2),
            padding_mask,
            data["label"],
            texts=data.get("text"),
            au=data.get("au"),
            metrics=(self.wer_video_val, self.wer_audio_val, self.wer_av_val),
        )

    def calculate_wer(self, video, audio, padding_mask, labels, texts=None, au=None, metrics=None):
        if metrics is None:
            metrics = (self.wer_video, self.wer_audio, self.wer_av)
        wer_video_metric, wer_audio_metric, wer_av_metric = metrics
        labels = labels.squeeze(1)
        for i, (vid, aud, label, mask) in enumerate(zip(video, audio, labels, padding_mask)):
            enc_kw = {}
            if getattr(self.cfg.model.backbone, "use_au", False):
                if au is not None:
                    enc_kw["au"] = au[i : i + 1].to(vid.device)
                else:
                    t = vid.shape[1]
                    enc_kw["au"] = vid.new_zeros(1, t, AU_FEATURE_DIM)
            feat_v, feat_a, feat_av, _, _ = self.model.encoder(
                vid.unsqueeze(0), aud.unsqueeze(0), mask.unsqueeze(0).unsqueeze(-2), **enc_kw
            )
            
            nbest_hyps_v = self.beam_search_video(
                    x=feat_v.squeeze(0),
                    modality="v",
                    maxlenratio=self.cfg.decode.maxlenratio,
                    minlenratio=self.cfg.decode.minlenratio
                )
            nbest_hyps_a = self.beam_search_audio(
                    x=feat_a.squeeze(0),
                    modality="a",
                    maxlenratio=self.cfg.decode.maxlenratio,
                    minlenratio=self.cfg.decode.minlenratio
                )
            nbest_hyps_av = self.beam_search_av(
                    x=feat_av.squeeze(0),
                    modality="av",
                    maxlenratio=self.cfg.decode.maxlenratio,
                    minlenratio=self.cfg.decode.minlenratio
                )
            
            nbest_hyps_v = [
                h.asdict() for h in nbest_hyps_v[: min(len(nbest_hyps_v), 1)]
            ]
            nbest_hyps_a = [
                h.asdict() for h in nbest_hyps_a[: min(len(nbest_hyps_a), 1)]
            ]
            nbest_hyps_av = [
                h.asdict() for h in nbest_hyps_av[: min(len(nbest_hyps_av), 1)]
            ]

            transcription_v = add_results_to_json(nbest_hyps_v, self.token_list)
            transcription_v = transcription_v.replace("<eos>", "")
            transcription_a = add_results_to_json(nbest_hyps_a, self.token_list)
            transcription_a = transcription_a.replace("<eos>", "")
            transcription_av = add_results_to_json(nbest_hyps_av, self.token_list)
            transcription_av = transcription_av.replace("<eos>", "")

            groundtruth = None
            if texts is not None and i < len(texts):
                t = texts[i]
                if isinstance(t, str) and t.strip():
                    groundtruth = t.strip()
            if groundtruth is None:
                label = label[label != self.ignore_id]
                groundtruth = ids_to_str(label, self.token_list)

            groundtruth = groundtruth.replace("▁", " ").strip()
            transcription_v = transcription_v.replace("▁", " ").strip()
            transcription_a = transcription_a.replace("▁", " ").strip()
            transcription_av = transcription_av.replace("▁", " ").strip()

            wer_video_metric.update(transcription_v, groundtruth)
            wer_audio_metric.update(transcription_a, groundtruth)
            wer_av_metric.update(transcription_av, groundtruth)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        wer_video = self.wer_video_val.compute()
        wer_audio = self.wer_audio_val.compute()
        wer_av = self.wer_av_val.compute()
        self.log("wer_video_val", wer_video, sync_dist=True)
        self.log("wer_audio_val", wer_audio, sync_dist=True)
        self.log("wer_av_val", wer_av, sync_dist=True)
        self.wer_video_val.reset()
        self.wer_audio_val.reset()
        self.wer_av_val.reset()

    def test_step(self, data, batch_idx):
        lengths = torch.tensor(data["video_lengths"], device=data["video"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device)
        self.calculate_wer(
            data["video"].squeeze(1), 
            data["audio"].transpose(1, 2),
            padding_mask, 
            data["label"],
            texts=data.get("text"),
            au=data.get("au"),
        )

        print(self.wer_video.compute())
        print(self.wer_audio.compute())
        print(self.wer_av.compute())

    def on_test_epoch_end(self):
        wer_video = self.wer_video.compute()
        wer_audio = self.wer_audio.compute()
        wer_av = self.wer_av.compute()
        print(wer_video)
        print(wer_audio)
        print(wer_av)
        self.log("wer_video", wer_video)
        self.log("wer_audio", wer_audio)
        self.log("wer_av", wer_av)
        self.wer_video.reset()
        self.wer_audio.reset()
        self.wer_av.reset()
    
    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    # potentially want different schedulers for predictors and rest of model
    def configure_optimizers(self):
        param_groups = get_param_groups_ft(
            self.model, 
            self.cfg.model.backbone.elayers, 
            self.cfg.optimizer.base_lr, 
            self.cfg.optimizer.base_lr_other, 
            self.cfg.optimizer.lr_decay_rate,
            min_lr=self.cfg.optimizer.min_lr,
        )

        optimizer = AdamW(
            param_groups, weight_decay=self.cfg.optimizer.weight_decay, betas=self.cfg.optimizer.betas
        )

        warmup_epochs = self.cfg.optimizer.warmup_epochs
        train_len = len(self.trainer.datamodule.train_dataloader())
        scheduler = {
            'scheduler': WarmupCosineScheduler(
                optimizer,
                warmup_epochs,
                self.cfg.trainer.max_epochs,
                train_len,
                self.cfg.optimizer.cosine_decay,
                excluded_group=None,
            ),
            'interval': 'step',
            'frequency': 1
        }

        self.momentum_scheduler = instantiate(
            self.cfg.model.momentum_scheduler,
            iter_per_epoch=train_len,
        )

        return [optimizer], [scheduler]

