<<<<<<< HEAD
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import EGTEADataset, load_split


class EGTEADataModule(pl.LightningDataModule):
    def __init__(
        self,
        egtea_root: str,
        annotation_dir: str,
        split: int          = 1,
        use_flow: bool      = False,
        batch_size: int     = 8,
        seq_len: int        = 128,
        feat_dim: int       = 1024,
        num_workers: int    = 0,
        val_size: int       = 2000,
        test_size: int      = 0,
        val_ratio: float    = 0.1,
        split_seed: int     = 42,
        sliding_window: bool = True,   # sliding window in validation
        stride: int         = 64,      # overlap 50%
        temporal_aug_range: int = 0,
        temporal_aug_prob: float = 0.5,
        feat_noise_std: float = 0.0,
        feat_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _lmdb_path(self, modality: str) -> str:
        s    = self.hparams.split
        name = f"TSN-C_3_egtea_action_CE_s{s}_{modality}_model_best_fcfull_hd"
        return f"{self.hparams.egtea_root}/{name}"

    def setup(self, stage=None):
        hp     = self.hparams
        common = dict(
            lmdb_rgb_path  = self._lmdb_path("rgb"),
            annotation_dir = hp.annotation_dir,
            seq_len        = hp.seq_len,
            feat_dim       = hp.feat_dim,
            use_flow       = hp.use_flow,
            lmdb_flow_path = self._lmdb_path("flow") if hp.use_flow else None,
        )

        # Split train/val a livello clip per evitare leakage.
        train_split_file = f"train_split{hp.split}.txt"
        all_train_clips = load_split(str(Path(hp.annotation_dir) / train_split_file))

        n_total = len(all_train_clips)
        if getattr(hp, "val_size", 0) > 0:
            n_val = min(int(hp.val_size), n_total - 1)
        else:
            n_val = int(n_total * hp.val_ratio)
            n_val = max(1, min(n_val, n_total - 1))

        rng = np.random.default_rng(hp.split_seed)
        idx = np.arange(n_total)
        rng.shuffle(idx)

        val_idx = set(idx[:n_val].tolist())
        train_clips = [all_train_clips[i] for i in range(n_total) if i not in val_idx]
        val_clips   = [all_train_clips[i] for i in range(n_total) if i in val_idx]

        # Training: crop casuale + temporal augmentation + feature augmentation.
        self.train_ds = EGTEADataset(
            split_file     = train_split_file,
            clips          = train_clips,
            sliding_window = False,
            temporal_aug_range = getattr(hp, "temporal_aug_range", 0),
            temporal_aug_prob  = getattr(hp, "temporal_aug_prob", 0.5),
            feat_noise_std     = getattr(hp, "feat_noise_std", 0.0),
            feat_drop_prob     = getattr(hp, "feat_drop_prob", 0.0),
            **common,
        )

        # Validation: subset del train senza augmentation.
        self.val_ds = EGTEADataset(
            split_file     = train_split_file,
            clips          = val_clips,
            sliding_window = hp.sliding_window,
            stride         = hp.stride,
            temporal_aug_range = 0,
            temporal_aug_prob  = 0.0,
            **common,
        )

        # Test: split test separato.
        test_split_file = f"test_split{hp.split}.txt"
        all_test_clips = load_split(str(Path(hp.annotation_dir) / test_split_file))
        test_clips = all_test_clips

        self.test_ds = EGTEADataset(
            split_file     = test_split_file,
            clips          = test_clips,
            sliding_window = hp.sliding_window,
            stride         = hp.stride,
            temporal_aug_range = 0,
            temporal_aug_prob  = 0.0,
            **common,
        )

        print(
            f"[DataModule] Train clips: {len(train_clips)} | "
            f"Val clips: {len(val_clips)} | Test clips: {len(test_clips)}"
        )
        if hp.sliding_window:
            print(f"[DataModule] Validation: sliding window "
                  f"(seq={hp.seq_len}, stride={hp.stride}) → "
                  f"{len(self.val_ds)} finestre da {len(self.val_ds.clips)} clip")
            print(f"[DataModule] Test: sliding window "
                  f"(seq={hp.seq_len}, stride={hp.stride}) → "
                  f"{len(self.test_ds)} finestre da {len(self.test_ds.clips)} clip")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size  = self.hparams.batch_size,
            shuffle     = True,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size  = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size  = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.num_workers > 0,
=======
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import EGTEADataset, load_split


class EGTEADataModule(pl.LightningDataModule):
    def __init__(
        self,
        egtea_root: str,
        annotation_dir: str,
        split: int          = 1,
        use_flow: bool      = False,
        batch_size: int     = 8,
        seq_len: int        = 128,
        feat_dim: int       = 1024,
        num_workers: int    = 0,
        val_size: int       = 2000,
        test_size: int      = 0,
        val_ratio: float    = 0.1,
        split_seed: int     = 42,
        sliding_window: bool = True,   # sliding window in validation
        stride: int         = 64,      # overlap 50%
        temporal_aug_range: int = 0,
        temporal_aug_prob: float = 0.5,
        feat_noise_std: float = 0.0,
        feat_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _lmdb_path(self, modality: str) -> str:
        s    = self.hparams.split
        name = f"TSN-C_3_egtea_action_CE_s{s}_{modality}_model_best_fcfull_hd"
        return f"{self.hparams.egtea_root}/{name}"

    def setup(self, stage=None):
        hp     = self.hparams
        common = dict(
            lmdb_rgb_path  = self._lmdb_path("rgb"),
            annotation_dir = hp.annotation_dir,
            seq_len        = hp.seq_len,
            feat_dim       = hp.feat_dim,
            use_flow       = hp.use_flow,
            lmdb_flow_path = self._lmdb_path("flow") if hp.use_flow else None,
        )

        # Split train/val a livello clip per evitare leakage.
        train_split_file = f"train_split{hp.split}.txt"
        all_train_clips = load_split(str(Path(hp.annotation_dir) / train_split_file))

        n_total = len(all_train_clips)
        if getattr(hp, "val_size", 0) > 0:
            n_val = min(int(hp.val_size), n_total - 1)
        else:
            n_val = int(n_total * hp.val_ratio)
            n_val = max(1, min(n_val, n_total - 1))

        rng = np.random.default_rng(hp.split_seed)
        idx = np.arange(n_total)
        rng.shuffle(idx)

        val_idx = set(idx[:n_val].tolist())
        train_clips = [all_train_clips[i] for i in range(n_total) if i not in val_idx]
        val_clips   = [all_train_clips[i] for i in range(n_total) if i in val_idx]

        # Training: crop casuale + temporal augmentation + feature augmentation.
        self.train_ds = EGTEADataset(
            split_file     = train_split_file,
            clips          = train_clips,
            sliding_window = False,
            temporal_aug_range = getattr(hp, "temporal_aug_range", 0),
            temporal_aug_prob  = getattr(hp, "temporal_aug_prob", 0.5),
            feat_noise_std     = getattr(hp, "feat_noise_std", 0.0),
            feat_drop_prob     = getattr(hp, "feat_drop_prob", 0.0),
            **common,
        )

        # Validation: subset del train senza augmentation.
        self.val_ds = EGTEADataset(
            split_file     = train_split_file,
            clips          = val_clips,
            sliding_window = hp.sliding_window,
            stride         = hp.stride,
            temporal_aug_range = 0,
            temporal_aug_prob  = 0.0,
            **common,
        )

        # Test: split test separato.
        test_split_file = f"test_split{hp.split}.txt"
        all_test_clips = load_split(str(Path(hp.annotation_dir) / test_split_file))
        test_clips = all_test_clips

        self.test_ds = EGTEADataset(
            split_file     = test_split_file,
            clips          = test_clips,
            sliding_window = hp.sliding_window,
            stride         = hp.stride,
            temporal_aug_range = 0,
            temporal_aug_prob  = 0.0,
            **common,
        )

        print(
            f"[DataModule] Train clips: {len(train_clips)} | "
            f"Val clips: {len(val_clips)} | Test clips: {len(test_clips)}"
        )
        if hp.sliding_window:
            print(f"[DataModule] Validation: sliding window "
                  f"(seq={hp.seq_len}, stride={hp.stride}) → "
                  f"{len(self.val_ds)} finestre da {len(self.val_ds.clips)} clip")
            print(f"[DataModule] Test: sliding window "
                  f"(seq={hp.seq_len}, stride={hp.stride}) → "
                  f"{len(self.test_ds)} finestre da {len(self.test_ds.clips)} clip")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size  = self.hparams.batch_size,
            shuffle     = True,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size  = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size  = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.num_workers > 0,
>>>>>>> 42c5e63ee0570321b67744f8b5f40cc8e0fffff0
        )