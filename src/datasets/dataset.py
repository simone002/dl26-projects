import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict

FPS = 24


def ms_to_frame(ms: int) -> int:
    return max(1, int(ms / 1000 * FPS))


def load_action_labels(csv_path: str) -> dict:
    cls_path = csv_path.replace("action_labels.csv", "cls_label_index.csv")
    label_to_id = {}
    with open(cls_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 2:
                try:
                    label_to_id[parts[1].strip()] = int(parts[0])
                except ValueError:
                    continue

    annotations = defaultdict(list)
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 6:
                continue
            try:
                video_session = parts[2].strip()
                start_ms      = int(parts[3])
                end_ms        = int(parts[4])
                action_label  = parts[5].strip()
                action_id     = label_to_id.get(action_label, -1)
                if action_id == -1:
                    continue
                annotations[video_session].append(
                    (ms_to_frame(start_ms), ms_to_frame(end_ms), action_id)
                )
            except (ValueError, IndexError):
                continue

    return dict(annotations)


def load_split(split_file: str) -> list:
    clips = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            clip_prefix   = parts[0]
            tokens        = clip_prefix.split("-")
            frame_start   = int(tokens[-2][1:])
            frame_end     = int(tokens[-1][1:])
            video_session = "-".join(tokens[:-4])
            clips.append({
                "clip_prefix":   clip_prefix,
                "video_session": video_session,
                "frame_start":   frame_start,
                "frame_end":     frame_end,
                "action_id":     int(parts[1]) - 1,
                "verb_id":       int(parts[2]) - 1,
                "noun_id":       int(parts[3]) - 1,
            })
    return clips


class NpyReader:
    def __init__(self, features_dir: str, feat_dim: int = 768):
        self.features_dir = Path(features_dir)
        self.feat_dim = feat_dim

    def get_clip(self, video_session: str,
                 frame_start: int, frame_end: int) -> np.ndarray:
        path = self.features_dir / f"{video_session}.npy"
        T = frame_end - frame_start + 1

        if not path.exists():
            return np.zeros((T, self.feat_dim), dtype=np.float32)

        features = np.load(str(path), mmap_mode="r")
        T_total  = len(features)
        s = max(0, frame_start - 1)
        e = min(frame_end, T_total)
        chunk = features[s:e].astype(np.float32)

        if len(chunk) < T:
            pad   = T - len(chunk)
            chunk = np.concatenate(
                [chunk, np.zeros((pad, self.feat_dim), dtype=np.float32)]
            )
        return chunk


class EGTEADataset(Dataset):
    def __init__(
        self,
        features_dir: str,
        annotation_dir: str,
        split_file: str,
        clips: list[dict] | None = None,
        seq_len: int   = 128,
        feat_dim: int  = 768,
        background_id: int   = 0,
        sliding_window: bool = False,
        stride: int          = 64,
        temporal_aug_range: int   = 0,
        temporal_aug_prob: float  = 0.5,
        feat_noise_std: float     = 0.0,
        feat_drop_prob: float     = 0.0,
    ):
        self.seq_len        = seq_len
        self.feat_dim       = feat_dim
        self.background_id  = background_id
        self.sliding_window = sliding_window
        self.stride         = stride
        self.temporal_aug_range = temporal_aug_range
        self.temporal_aug_prob  = temporal_aug_prob
        self.feat_noise_std = feat_noise_std
        self.feat_drop_prob = feat_drop_prob

        ann_dir = Path(annotation_dir)
        self.clips = clips if clips is not None else load_split(
            str(ann_dir / split_file)
        )

        raw_ann = ann_dir / "raw_annotations" / "action_labels.csv"
        self.dense_annotations = load_action_labels(str(raw_ann))

        self.reader = NpyReader(features_dir, feat_dim=feat_dim)

        self._samples = self._build_sliding_samples() if sliding_window else None

    def _build_sliding_samples(self) -> list[tuple[int, int]]:
        samples = []
        for i, clip in enumerate(self.clips):
            T = clip["frame_end"] - clip["frame_start"] + 1
            if T <= self.seq_len:
                samples.append((i, 0))
            else:
                for start in range(0, T - self.seq_len + 1, self.stride):
                    samples.append((i, start))
                last_start = T - self.seq_len
                if (last_start % self.stride) != 0:
                    samples.append((i, last_start))
        return samples

    def __len__(self) -> int:
        return len(self._samples) if self.sliding_window else len(self.clips)

    def _build_dense_labels(self, video_session, frame_start, frame_end):
        T = frame_end - frame_start + 1
        labels = np.full(T, self.background_id, dtype=np.int64)
        for (ann_start, ann_end, action_id) in \
                self.dense_annotations.get(video_session, []):
            o_start = max(ann_start, frame_start)
            o_end   = min(ann_end,   frame_end)
            if o_start <= o_end:
                labels[o_start - frame_start : o_end - frame_start + 1] = action_id
        return labels

    def _temporal_shift_labels(self, labels: np.ndarray) -> np.ndarray:
        if self.temporal_aug_range == 0 or np.random.rand() > self.temporal_aug_prob:
            return labels
        shift = np.random.randint(
            -self.temporal_aug_range, self.temporal_aug_range + 1
        )
        if shift == 0:
            return labels
        shifted = np.full_like(labels, self.background_id)
        if shift > 0:
            shifted[shift:] = labels[:-shift]
        else:
            shifted[:shift] = labels[-shift:]
        return shifted

    def _augment_features(self, feat: np.ndarray) -> np.ndarray:
        if self.feat_noise_std > 0:
            feat = feat + np.random.randn(*feat.shape).astype(np.float32) \
                   * self.feat_noise_std
        if self.feat_drop_prob > 0:
            mask = np.random.rand(feat.shape[1]).astype(np.float32)
            feat = feat * (mask > self.feat_drop_prob).astype(np.float32)
        return feat

    def _pad_or_crop(self, feat, labels, window_start=None):
        T = feat.shape[0]
        if T <= self.seq_len:
            pad    = self.seq_len - T
            feat   = np.concatenate(
                [feat, np.zeros((pad, feat.shape[1]), dtype=np.float32)]
            )
            labels = np.concatenate(
                [labels, np.full(pad, self.background_id, dtype=np.int64)]
            )
        elif window_start is not None:
            feat   = feat[window_start : window_start + self.seq_len]
            labels = labels[window_start : window_start + self.seq_len]
        else:
            start  = np.random.randint(0, T - self.seq_len + 1)
            feat   = feat[start : start + self.seq_len]
            labels = labels[start : start + self.seq_len]
            labels = self._temporal_shift_labels(labels)
            feat   = self._augment_features(feat)
        return feat, labels

    def __getitem__(self, idx: int):
        if self.sliding_window:
            clip_idx, window_start = self._samples[idx]
            clip = self.clips[clip_idx]
        else:
            clip = self.clips[idx]
            window_start = None

        feat = self.reader.get_clip(
            clip["video_session"], clip["frame_start"], clip["frame_end"]
        )
        labels = self._build_dense_labels(
            clip["video_session"], clip["frame_start"], clip["frame_end"]
        )
        feat, labels = self._pad_or_crop(feat, labels, window_start)

        return (
            torch.from_numpy(feat).float(),
            torch.from_numpy(labels).long(),
        )
