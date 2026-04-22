import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from utils.au_npz import AU_FEATURE_DIM, load_au_from_npz
from utils.hf_media import hub_local_path


def cut_or_pad(data, size, dim=0):
    # Pad with zeros on the right if data is too short
    # assert abs(data.size(dim) - size) < 2000 
    if data.size(dim) < size:
        # assert False
        padding = size - data.size(dim)
        data = torch.from_numpy(np.pad(data, (0, padding), "constant"))
    # Cut from the right if data is too long
    elif data.size(dim) > size:
        data = data[:size]
    # Keep if data is exactly right
    assert data.size(dim) == size
    return data


class AVDataset(Dataset):
    def __init__(
            self, 
            data_path,
            video_path_prefix_lrs2,
            audio_path_prefix_lrs2,
            video_path_prefix_lrs3, 
            audio_path_prefix_lrs3, 
            video_path_prefix_vox2=None, 
            audio_path_prefix_vox2=None, 
            transforms=None,
            skip_fails=True,
            au_enabled=False,
            au_npz_dirs=None,
            hub_repo_ids=None,
            hub_repo_prefixes=None,
            hub_repo_type="dataset",
            hub_revision=None,
            hub_cache_dir=None,
            max_frames_per_sample=None,
            max_manifest_samples=None,
    ):

        self.data_path = data_path
        self.video_path_prefix_lrs3 = video_path_prefix_lrs3
        self.audio_path_prefix_lrs3 = audio_path_prefix_lrs3
        self.video_path_prefix_vox2 = video_path_prefix_vox2
        self.audio_path_prefix_vox2 = audio_path_prefix_vox2
        self.video_path_prefix_lrs2 = video_path_prefix_lrs2
        self.audio_path_prefix_lrs2 = audio_path_prefix_lrs2
        self.transforms = transforms
        # Fairseq batch_by_size requires every length <= max_tokens (see frames_per_gpu in samplers).
        self.max_frames_per_sample = max_frames_per_sample
        self.max_manifest_samples = max_manifest_samples

        self.paths_counts_labels = self.configure_files()
        self.num_fails = 0

        self.skip_fails = skip_fails
        self.au_enabled = au_enabled
        self.au_npz_dirs = au_npz_dirs or {}
        self.hub_repo_ids = hub_repo_ids or {}
        self.hub_repo_prefixes = hub_repo_prefixes or {}
        self.hub_repo_type = hub_repo_type or "dataset"
        rev = hub_revision
        if rev is not None and str(rev).strip() == "":
            rev = None
        self.hub_revision = rev
        self.hub_cache_dir = hub_cache_dir

    @staticmethod
    def _parse_tag(tag: str):
        """Split e.g. 'lrs2_03' into ('lrs2', '03') or plain 'lrs2' into ('lrs2', None)."""
        import re
        m = re.match(r"^(.+?)_(\d{2})$", tag)
        if m:
            return m.group(1), m.group(2)
        return tag, None

    def _hub_repo(self, tag: str):
        r = (self.hub_repo_ids.get(tag) or "").strip()
        if r:
            return r
        base, shard = self._parse_tag(tag)
        if shard:
            prefix = (self.hub_repo_prefixes.get(base) or "").strip()
            if prefix:
                return f"{prefix}_{shard}"
        prefix = (self.hub_repo_prefixes.get(tag) or "").strip()
        if prefix:
            return prefix
        return None

    def _posix_stem_ext(self, file_path: str, ext: str) -> str:
        base, _ = os.path.splitext(file_path)
        return (base + ext).replace("\\", "/")

    def _local_prefix(self, kind: str, tag: str) -> str:
        """Resolve local path prefix, falling back from e.g. lrs2_03 to lrs2."""
        attr = f"{kind}_path_prefix_{tag}"
        val = getattr(self, attr, None)
        if val:
            return val
        base, shard = self._parse_tag(tag)
        if shard:
            attr_base = f"{kind}_path_prefix_{base}"
            return getattr(self, attr_base, "") or ""
        return ""

    def _resolve_video_path(self, tag: str, file_path: str) -> str:
        repo = self._hub_repo(tag)
        if repo:
            return hub_local_path(
                repo,
                file_path.replace("\\", "/"),
                repo_type=self.hub_repo_type,
                revision=self.hub_revision,
                cache_dir=self.hub_cache_dir,
            )
        return os.path.join(self._local_prefix("video", tag), file_path)

    def _resolve_audio_path(self, tag: str, file_path: str) -> str:
        rel_wav = self._posix_stem_ext(file_path, ".wav")
        repo = self._hub_repo(tag)
        if repo:
            return hub_local_path(
                repo,
                rel_wav,
                repo_type=self.hub_repo_type,
                revision=self.hub_revision,
                cache_dir=self.hub_cache_dir,
            )
        return os.path.join(self._local_prefix("audio", tag), rel_wav)

    def _resolve_au_npz_path(self, tag: str, file_path: str) -> str:
        repo = self._hub_repo(tag)
        if repo:
            rel_npz = self._posix_stem_ext(file_path, ".npz")
            return hub_local_path(
                repo,
                rel_npz,
                repo_type=self.hub_repo_type,
                revision=self.hub_revision,
                cache_dir=self.hub_cache_dir,
            )
        return self._au_npz_path(tag, file_path) or ""

    def _au_npz_path(self, tag: str, file_path: str):
        if not self.au_enabled:
            return None
        base_tag, _ = self._parse_tag(tag)
        npz_base = self.au_npz_dirs.get(tag) or self.au_npz_dirs.get(base_tag) or ""
        if not npz_base:
            return None
        stem = os.path.splitext(os.path.basename(file_path))[0]
        return os.path.join(npz_base, f"{stem}.npz")

    def _resolve_media_paths(self, tag: str, file_path: str):
        """Local paths for video, audio, and optional AU npz. Hub downloads run in parallel per clip."""
        if not self._hub_repo(tag):
            v = self._resolve_video_path(tag, file_path)
            a = self._resolve_audio_path(tag, file_path)
            u = self._resolve_au_npz_path(tag, file_path) if self.au_enabled else None
            return v, a, u
        n_workers = 3 if self.au_enabled else 2
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            fut_v = pool.submit(self._resolve_video_path, tag, file_path)
            fut_a = pool.submit(self._resolve_audio_path, tag, file_path)
            if self.au_enabled:
                fut_u = pool.submit(self._resolve_au_npz_path, tag, file_path)
                return fut_v.result(), fut_a.result(), fut_u.result()
            return fut_v.result(), fut_a.result(), None

    def prefetch_hub_cache(self, max_workers: int = 16):
        """Download all Hub files referenced by this manifest in parallel (warms cache; rank-0 only recommended)."""
        tasks = []
        seen = set()
        for tag, file_path, _c, _l in self.paths_counts_labels:
            repo = self._hub_repo(tag)
            if not repo:
                continue
            fp = file_path.replace("\\", "/")
            for rel in (fp, self._posix_stem_ext(fp, ".wav")):
                key = (repo, rel)
                if key not in seen:
                    seen.add(key)
                    tasks.append(key)
            if self.au_enabled:
                rel = self._posix_stem_ext(fp, ".npz")
                key = (repo, rel)
                if key not in seen:
                    seen.add(key)
                    tasks.append(key)
        if not tasks:
            return

        def _download(item):
            repo_id, filename = item
            hub_local_path(
                repo_id,
                filename,
                repo_type=self.hub_repo_type,
                revision=self.hub_revision,
                cache_dir=self.hub_cache_dir,
            )

        print(f"Prefetching {len(tasks)} Hugging Face Hub files (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_download, tasks))
        print("Hub prefetch finished.")
    
    def configure_files(self):
        # from https://github.com/facebookresearch/pytorchvideo/blob/874d27cb55b9d7e9df6cd0881e2d7fe9f262532b/pytorchvideo/data/labeled_video_paths.py#L37
        paths_counts_labels = []
        limit = self.max_manifest_samples
        with open(self.data_path, "r") as f:
            for path_count_label in f:
                path_count_label = path_count_label.strip()
                if not path_count_label:
                    continue
                tag, file_path, count, label = path_count_label.split(",")
                tag, file_path, count, label = (
                    tag.strip(),
                    file_path.strip(),
                    count.strip(),
                    label.strip(),
                )
                c = int(count)
                if self.max_frames_per_sample is not None:
                    c = min(c, int(self.max_frames_per_sample))
                paths_counts_labels.append((tag, file_path, c, [int(lab) for lab in label.split()]))
                if limit is not None and len(paths_counts_labels) >= int(limit):
                    break
        return paths_counts_labels

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()
        if not frames:
            print(path)
            return None
        frames = torch.from_numpy(np.stack(frames))
        frames = frames.permute((3, 0, 1, 2))  # TxHxWxC -> # CxTxHxW
        return frames
    
    def load_audio(self, path):
        try:
            audio, sr = torchaudio.load(path, normalize=True)
        except (RuntimeError, OSError) as e:
            print(f"load_audio failed ({path}): {e}")
            return None
        # assert sr == 16_000
        return audio
        
    def __len__(self):
        return len(self.paths_counts_labels)

    def __getitem__(self, index):
        tag, file_path, count, label = self.paths_counts_labels[index]

        video_path, audio_path, npz_path = self._resolve_media_paths(tag, file_path)
        video = self.load_video(video_path)
        if video is None:
            self.num_fails += 1
            if self.num_fails == 300:
                raise ValueError("Too many file errors.")
            # if count > 450:
            # return self.__getitem__(index + 1)
            if self.skip_fails or int(count) < 350:
                out = {'video': None, 'audio': None, 'label': None}
                if self.au_enabled:
                    out['au'] = None
                return out
            else:
                return self.__getitem__(index + 1)
        # Match capped manifest length (fairseq: each sample must be <= frames_per_gpu / frames_per_gpu_val).
        if video.size(1) > count:
            video = video[:, :count].contiguous()
        audio = self.load_audio(audio_path)
        if audio is None:
            self.num_fails += 1
            if self.num_fails == 300:
                raise ValueError("Too many file errors.")
            if self.skip_fails or int(count) < 350:
                out = {"video": None, "audio": None, "label": None}
                if self.au_enabled:
                    out["au"] = None
                return out
            return self.__getitem__((index + 1) % len(self))

        audio = cut_or_pad(audio.squeeze(0), video.size(1) * 640)

        video_clean = self.transforms['video'](video).permute((1, 2, 3, 0))
        audio_clean = self.transforms['audio'](audio.unsqueeze(0)).squeeze(0)

        out = {
            'video': video_clean, 'audio': audio_clean, 'label': torch.tensor(label)
        }
        if self.au_enabled:
            t_frames = video_clean.size(0)
            out['au'] = load_au_from_npz(npz_path, t_frames)
            assert out['au'].shape == (t_frames, AU_FEATURE_DIM)
        return out
