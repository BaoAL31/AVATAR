import os

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
        ):

        self.data_path = data_path
        self.video_path_prefix_lrs3 = video_path_prefix_lrs3
        self.audio_path_prefix_lrs3 = audio_path_prefix_lrs3
        self.video_path_prefix_vox2 = video_path_prefix_vox2
        self.audio_path_prefix_vox2 = audio_path_prefix_vox2
        self.video_path_prefix_lrs2 = video_path_prefix_lrs2
        self.audio_path_prefix_lrs2 = audio_path_prefix_lrs2
        self.transforms = transforms

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
    
    def configure_files(self):
        # from https://github.com/facebookresearch/pytorchvideo/blob/874d27cb55b9d7e9df6cd0881e2d7fe9f262532b/pytorchvideo/data/labeled_video_paths.py#L37
        paths_counts_labels = []
        with open(self.data_path, "r") as f:
            for path_count_label in f.read().splitlines():
                tag, file_path, count, label = path_count_label.split(",")
                paths_counts_labels.append((tag, file_path, int(count), [int(lab) for lab in label.split()]))
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
        audio, sr = torchaudio.load(path, normalize=True)
        # assert sr == 16_000
        return audio
        
    def __len__(self):
        return len(self.paths_counts_labels)

    def __getitem__(self, index):
        tag, file_path, count, label = self.paths_counts_labels[index]

        video = self.load_video(self._resolve_video_path(tag, file_path))
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
        audio = self.load_audio(self._resolve_audio_path(tag, file_path))

        audio = cut_or_pad(audio.squeeze(0), video.size(1) * 640)

        video_clean = self.transforms['video'](video).permute((1, 2, 3, 0))
        audio_clean = self.transforms['audio'](audio.unsqueeze(0)).squeeze(0)

        out = {
            'video': video_clean, 'audio': audio_clean, 'label': torch.tensor(label)
        }
        if self.au_enabled:
            t_frames = video_clean.size(0)
            npz_path = self._resolve_au_npz_path(tag, file_path)
            out['au'] = load_au_from_npz(npz_path, t_frames)
            assert out['au'].shape == (t_frames, AU_FEATURE_DIM)
        return out
