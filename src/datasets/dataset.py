import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict

FPS = 24  # EGTEA Gaze+ è girato a 24fps


def ms_to_frame(ms: int) -> int:
    return max(1, int(ms / 1000 * FPS))


def load_action_labels(csv_path: str) -> dict:
    """
    action_labels.csv + cls_label_index.csv
    → {video_session: [(frame_start, frame_end, action_id), ...]}
    """
    cls_path   = csv_path.replace("action_labels.csv", "cls_label_index.csv")
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


_LMDB_ENVS: dict = {}


class LMDBReader:
    def __init__(self, path: str, feat_dim: int = 1024):
        self.path     = path
        self.feat_dim = feat_dim

    def _get_env(self):
        if self.path not in _LMDB_ENVS:
            _LMDB_ENVS[self.path] = lmdb.open(
                self.path, readonly=True, lock=False,
                readahead=False, meminit=False,
            )
        return _LMDB_ENVS[self.path]

    def get_frame(self, video_session: str, frame_idx: int):
        key = f"{video_session}_frame_{frame_idx:010d}.jpg"
        with self._get_env().begin(write=False) as txn:
            data = txn.get(key.encode("utf-8"))
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32).copy()

    def get_clip(self, video_session: str, frame_start: int, frame_end: int):
        frames = []
        for idx in range(frame_start, frame_end + 1):
            feat = self.get_frame(video_session, idx)
            if feat is None:
                feat = np.zeros(self.feat_dim, dtype=np.float32)
            frames.append(feat)
        return np.stack(frames, axis=0)


class EGTEADataset(Dataset):
    """
    Dataset con label dense frame-per-frame e sliding window.

    Modalità:
      - sliding_window=False (default): crop casuale a seq_len (training)
      - sliding_window=True:  genera una finestra per ogni posizione
                              con stride=stride (validation/test più accurato)
    """

    def __init__(
        self,
        lmdb_rgb_path: str,
        annotation_dir: str,
        split_file: str,
        clips: list[dict] | None = None,
        seq_len: int        = 128,
        feat_dim: int       = 1024,
        use_flow: bool      = False,
        lmdb_flow_path      = None,
        background_id: int  = 0,
        sliding_window: bool = False,
        stride: int         = 64,   # overlap 50% con seq_len=128
        temporal_aug_range: int = 0,  # max frame shift (training only)
        temporal_aug_prob: float = 0.5,  # probability of applying shift
        feat_noise_std: float = 0.0,  # gaussian noise std on features (training only)
        feat_drop_prob: float = 0.0,  # probability of zeroing each feature dim
    ):
        self.seq_len        = seq_len
        self.feat_dim       = feat_dim
        self.use_flow       = use_flow
        self.background_id  = background_id
        self.sliding_window = sliding_window
        self.stride         = stride
        self.temporal_aug_range = temporal_aug_range
        self.temporal_aug_prob  = temporal_aug_prob
        self.feat_noise_std = feat_noise_std
        self.feat_drop_prob = feat_drop_prob

        ann_dir = Path(annotation_dir)
        self.clips = clips if clips is not None else load_split(str(ann_dir / split_file))

        raw_ann = ann_dir / "raw_annotations" / "action_labels.csv"
        self.dense_annotations = load_action_labels(str(raw_ann))

        self.rgb_reader  = LMDBReader(lmdb_rgb_path,  feat_dim=feat_dim)
        self.flow_reader = LMDBReader(lmdb_flow_path, feat_dim=feat_dim) \
                           if use_flow and lmdb_flow_path else None

        # Con sliding window pre-calcola tutti i sample (clip_idx, window_start)
        if sliding_window:
            self._samples = self._build_sliding_samples()
        else:
            self._samples = None

    def _build_sliding_samples(self) -> list[tuple[int, int]]:
        """
        Per ogni clip genera tutte le finestre di seq_len con passo stride.
        Ritorna lista di (clip_idx, window_start_local).
        """
        samples = []
        for i, clip in enumerate(self.clips):
            T = clip["frame_end"] - clip["frame_start"] + 1
            if T <= self.seq_len:
                # Clip più corto di seq_len → una sola finestra (con padding)
                samples.append((i, 0))
            else:
                # Finestre sovrapposte
                for start in range(0, T - self.seq_len + 1, self.stride):
                    samples.append((i, start))
                # Assicura che l'ultima finestra copra la fine del clip
                last_start = T - self.seq_len
                if (last_start % self.stride) != 0:
                    samples.append((i, last_start))
        return samples

    def __len__(self) -> int:
        if self.sliding_window:
            return len(self._samples)
        return len(self.clips)

    def _build_dense_labels(self, video_session, frame_start, frame_end):
        T      = frame_end - frame_start + 1
        labels = np.full(T, self.background_id, dtype=np.int64)
        for (ann_start, ann_end, action_id) in \
                self.dense_annotations.get(video_session, []):
            o_start = max(ann_start, frame_start)
            o_end   = min(ann_end,   frame_end)
            if o_start <= o_end:
                labels[o_start - frame_start : o_end - frame_start + 1] = action_id
        return labels

    def _get_window(self, clip_idx: int, window_start: int):
        """Carica feature e label per una finestra specifica."""
        clip = self.clips[clip_idx]

        feat = self.rgb_reader.get_clip(
            clip["video_session"], clip["frame_start"], clip["frame_end"]
        )
        if self.use_flow and self.flow_reader is not None:
            flow = self.flow_reader.get_clip(
                clip["video_session"], clip["frame_start"], clip["frame_end"]
            )
            feat = np.concatenate([feat, flow], axis=-1)

        labels = self._build_dense_labels(
            clip["video_session"], clip["frame_start"], clip["frame_end"]
        )

        T = feat.shape[0]
        if T <= self.seq_len:
            # Padding
            pad = self.seq_len - T
            feat   = np.concatenate([feat,
                np.zeros((pad, feat.shape[1]), dtype=np.float32)])
            labels = np.concatenate([labels,
                np.full(pad, self.background_id, dtype=np.int64)])
        else:
            # Estrai finestra
            feat   = feat[window_start : window_start + self.seq_len]
            labels = labels[window_start : window_start + self.seq_len]

        return feat, labels

    def __getitem__(self, idx: int):
        if self.sliding_window:
            clip_idx, window_start = self._samples[idx]
            feat, labels = self._get_window(clip_idx, window_start)
        else:
            # Crop casuale (training)
            clip   = self.clips[idx]
            feat   = self.rgb_reader.get_clip(
                clip["video_session"], clip["frame_start"], clip["frame_end"]
            )
            if self.use_flow and self.flow_reader is not None:
                flow = self.flow_reader.get_clip(
                    clip["video_session"], clip["frame_start"], clip["frame_end"]
                )
                feat = np.concatenate([feat, flow], axis=-1)

            labels = self._build_dense_labels(
                clip["video_session"], clip["frame_start"], clip["frame_end"]
            )
            feat, labels = self._random_crop(feat, labels)

        return (
            torch.from_numpy(feat).float(),
            torch.from_numpy(labels).long(),
        )

    def _temporal_shift_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Applica shift temporale casuale alle label per data augmentation.
        Aiuta il modello a gestire offset temporali e riduce bias di anticipazione.
        shift > 0: label vengono shiftate in avanti (azione accade dopo nella sequenza)
        shift < 0: label vengono shiftate indietro (azione accade prima)
        """
        if self.temporal_aug_range == 0 or np.random.rand() > self.temporal_aug_prob:
            return labels
        
        shift = np.random.randint(-self.temporal_aug_range, self.temporal_aug_range + 1)
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
            feat = feat + np.random.randn(*feat.shape).astype(np.float32) * self.feat_noise_std
        if self.feat_drop_prob > 0:
            mask = np.random.rand(feat.shape[1]).astype(np.float32)
            feat = feat * (mask > self.feat_drop_prob).astype(np.float32)
        return feat

    def _random_crop(self, feat, labels):
        T = feat.shape[0]
        if T >= self.seq_len:
            start  = np.random.randint(0, T - self.seq_len + 1)
            feat   = feat[start : start + self.seq_len]
            labels = labels[start : start + self.seq_len]
        else:
            pad = self.seq_len - T
            feat   = np.concatenate([feat,
                np.zeros((pad, feat.shape[1]), dtype=np.float32)])
            labels = np.concatenate([labels,
                np.full(pad, self.background_id, dtype=np.int64)])

        labels = self._temporal_shift_labels(labels)
        feat   = self._augment_features(feat)
