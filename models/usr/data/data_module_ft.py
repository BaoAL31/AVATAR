from copy import deepcopy
import os

from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from .dataset_ft import AVDataset
from .samplers import ByFrameCountSampler, DistributedSamplerWrapper, RandomSamplerWrapper
from .transforms import AdaptiveLengthTimeMask, NormalizeVideo


def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) < 3:
        collated_batch = collated_batch.unsqueeze(1)
    else:
        collated_batch = collated_batch.permute((0, 4, 1, 2, 3)) # [B, T, H, W, C] -> [B, C, T, H, W]
    return collated_batch, lengths


def pad_time_feature(samples, pad_val=0.0):
    """Batch (T, D) tensors to [B, T_max, D] without video-style channel permute."""
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in ('video', 'audio', 'label'):
        pad_val = -1 if data_type == 'label' else 0.0
        c_batch, sample_lengths = pad([s[data_type] for s in batch if s[data_type] is not None], pad_val)
        batch_out[data_type] = c_batch
        batch_out[data_type + '_lengths'] = sample_lengths

    batch_out["text"] = [s.get("text") for s in batch if s.get("video") is not None]

    if batch[0].get("au") is not None:
        samples = [s["au"] for s in batch if s.get("video") is not None and s.get("au") is not None]
        if samples:
            c_batch, sample_lengths = pad_time_feature(samples, 0.0)
            batch_out["au"] = c_batch
            batch_out["au_lengths"] = sample_lengths

    return batch_out


class DataModule(LightningDataModule):

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        print('total gpus:', self.total_gpus)

    def _au_dataset_kwargs(self):
        au_cfg = getattr(self.cfg.data, "au", None)
        if not au_cfg or not au_cfg.get("enabled"):
            return {"au_enabled": False, "au_npz_dirs": {}}
        return {
            "au_enabled": True,
            "au_npz_dirs": {
                "lrs2": au_cfg.get("lrs2_npz_dir") or "",
                "lrs3": au_cfg.get("lrs3_npz_dir") or "",
                "vox2": au_cfg.get("vox2_npz_dir") or "",
            },
        }

    def _hub_dataset_kwargs(self):
        h = getattr(self.cfg.data, "hub", None)
        if not h:
            return {
                "hub_repo_ids": {},
                "hub_repo_prefixes": {},
                "hub_repo_type": "dataset",
                "hub_revision": None,
                "hub_cache_dir": None,
            }
        rev = h.get("revision")
        if rev is not None and str(rev).strip() == "":
            rev = None
        ids = {
            "lrs2": (h.get("lrs2_repo_id") or "").strip(),
            "lrs3": (h.get("lrs3_repo_id") or "").strip(),
            "vox2": (h.get("vox2_repo_id") or "").strip(),
        }
        prefixes = {
            "lrs2": (h.get("lrs2_repo_prefix") or "").strip(),
            "lrs3": (h.get("lrs3_repo_prefix") or "").strip(),
            "vox2": (h.get("vox2_repo_prefix") or "").strip(),
        }
        use_hub = any(ids.values()) or any(prefixes.values())
        hub_cache_dir = None
        if use_hub:
            cd = (h.get("cache_dir") or "").strip()
            if cd:
                os.makedirs(cd, exist_ok=True)
                hub_cache_dir = cd
        return {
            "hub_repo_ids": ids,
            "hub_repo_prefixes": prefixes,
            "hub_repo_type": h.get("repo_type") or "dataset",
            "hub_revision": rev,
            "hub_cache_dir": hub_cache_dir,
        }

    def _manifest_sample_kw(self, split: str) -> dict:
        """Optional cap on CSV rows: train / val / test (see data.max_*_samples)."""
        d = getattr(self.cfg.data, "max_train_samples", None)
        v = getattr(self.cfg.data, "max_val_samples", None)
        t = getattr(self.cfg.data, "max_test_samples", None)
        if split == "train":
            n = d
        elif split == "val":
            n = v
        else:
            n = t if t is not None else v
        if n is None:
            return {}
        return {"max_manifest_samples": int(n)}

    def _make_train_dataset(self):
        ds_args = self.cfg.data.dataset
        transform_video = self._video_transform(mode="train")
        transform_audio = self._raw_audio_transform(mode="train")
        return AVDataset(
            data_path=ds_args.train_csv,
            video_path_prefix_lrs2=self.cfg.data.lrs2_video_dir,
            audio_path_prefix_lrs2=self.cfg.data.lrs2_audio_dir,
            video_path_prefix_lrs3=self.cfg.data.lrs3_video_dir,
            audio_path_prefix_lrs3=self.cfg.data.lrs3_audio_dir,
            video_path_prefix_vox2=self.cfg.data.vox2_video_dir,
            audio_path_prefix_vox2=self.cfg.data.vox2_audio_dir,
            transforms={"video": transform_video, "audio": transform_audio},
            max_frames_per_sample=self.cfg.data.frames_per_gpu,
            **self._manifest_sample_kw("train"),
            **self._au_dataset_kwargs(),
            **self._hub_dataset_kwargs(),
        )

    def prepare_data(self):
        """Optional parallel prefetch of all train-manifest Hub files (rank 0 only in Lightning)."""
        h = getattr(self.cfg.data, "hub", None)
        if not h or not h.get("prefetch_manifest", False):
            return
        w = int(h.get("prefetch_max_workers") or 16)
        self._make_train_dataset().prefetch_hub_cache(max_workers=w)

    def _video_transform(self, mode):
        args = self.cfg.data
        transform = [
            Lambda(lambda x: x / 255.),
        ] + (
            [
                RandomCrop(args.crop_type.random_crop_dim),
                Resize(args.crop_type.resize_dim, antialias=True),
                RandomHorizontalFlip(args.horizontal_flip_prob)
            ]
            if mode == "train" else [CenterCrop(args.crop_type.random_crop_dim), Resize(args.crop_type.resize_dim, antialias=True)]
        )
        if self.cfg.data.channel.in_video_channels == 1:
            transform.extend([Lambda(lambda x: x.transpose(0, 1)), Grayscale(), Lambda(lambda x: x.transpose(0, 1))])
        transform.append(NormalizeVideo(args.channel.obj.mean, args.channel.obj.std))

        if mode == "train":
            transform.append(
                AdaptiveLengthTimeMask(
                    window=int(args.timemask_window * 25),
                    stride=int(args.timemask_stride * 25),
                    replace_with_zero=True
                )
            )

        return Compose(transform)

    def _raw_audio_transform(self, mode):
        args = self.cfg.data
        transform = [Lambda(lambda x: x)]
        if mode == "train":
            transform.append(
                AdaptiveLengthTimeMask(
                    window=int(args.timemask_window_audio * 16_000),
                    stride=int(args.timemask_stride_audio * 16_000),
                    replace_with_zero=True
                )
            )

        return Compose(transform)

    def _dataloader(self, ds, sampler, collate_fn):
        return DataLoader(
            ds,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        train_ds = self._make_train_dataset()
        sampler = ByFrameCountSampler(train_ds, self.cfg.data.frames_per_gpu)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = RandomSamplerWrapper(sampler)
        return self._dataloader(train_ds, sampler, collate_pad)

    def val_dataloader(self):
        ds_args = self.cfg.data.dataset

        transform_video = self._video_transform(mode='val')
        transform_audio = self._raw_audio_transform(mode='val')

        val_ds = AVDataset(
            data_path=ds_args.val_csv,
            video_path_prefix_lrs2=self.cfg.data.lrs2_video_dir,
            audio_path_prefix_lrs2=self.cfg.data.lrs2_audio_dir,
            video_path_prefix_lrs3=self.cfg.data.lrs3_video_dir,
            audio_path_prefix_lrs3=self.cfg.data.lrs3_audio_dir,
            video_path_prefix_vox2=self.cfg.data.vox2_video_dir,
            audio_path_prefix_vox2=self.cfg.data.vox2_audio_dir,
            transforms={'video': transform_video, 'audio': transform_audio},
            max_frames_per_sample=self.cfg.data.frames_per_gpu_val,
            **self._manifest_sample_kw("val"),
            **self._au_dataset_kwargs(),
            **self._hub_dataset_kwargs(),
        )
        sampler = ByFrameCountSampler(val_ds, self.cfg.data.frames_per_gpu_val, shuffle=False)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(val_ds, sampler, collate_pad)

    def test_dataloader(self):
        ds_args = self.cfg.data.dataset

        transform_video = self._video_transform(mode='val')
        transform_audio = self._raw_audio_transform(mode='val')

        test_ds = AVDataset(
            data_path=ds_args.test_csv,
            video_path_prefix_lrs2=self.cfg.data.lrs2_video_dir,
            audio_path_prefix_lrs2=self.cfg.data.lrs2_audio_dir,
            video_path_prefix_lrs3=self.cfg.data.lrs3_video_dir,
            audio_path_prefix_lrs3=self.cfg.data.lrs3_audio_dir,
            video_path_prefix_vox2=self.cfg.data.vox2_video_dir,
            audio_path_prefix_vox2=self.cfg.data.vox2_audio_dir,
            transforms={'video': transform_video, 'audio': transform_audio},
            max_frames_per_sample=self.cfg.data.frames_per_gpu_val,
            **self._manifest_sample_kw("test"),
            **self._au_dataset_kwargs(),
            **self._hub_dataset_kwargs(),
        )
        sampler = ByFrameCountSampler(test_ds, self.cfg.data.frames_per_gpu_val, shuffle=False)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(test_ds, sampler, collate_pad)
