import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from .dataset import EGTEADataset, load_split


class EGTEADataModule(pl.LightningDataModule):
    def __init__(
        self,
        features_dir: str,
        annotation_dir: str,
        train_splits: list  = None,
        val_split: int      = 3,
        batch_size: int     = 8,
        seq_len: int        = 128,
        feat_dim: int       = 768,
        num_workers: int    = 0,
        sliding_window: bool = True,
        stride: int         = 64,
        temporal_aug_range: int   = 0,
        temporal_aug_prob: float  = 0.5,
        feat_noise_std: float     = 0.0,
        feat_drop_prob: float     = 0.0,
    ):
        super().__init__()
        if train_splits is None:
            train_splits = [1, 2]
        self.save_hyperparameters()

    def setup(self, stage=None):
        hp      = self.hparams
        ann_dir = Path(hp.annotation_dir)

        common = dict(
            features_dir   = hp.features_dir,
            annotation_dir = hp.annotation_dir,
            seq_len        = hp.seq_len,
            feat_dim       = hp.feat_dim,
        )

        train_datasets = []
        for s in hp.train_splits:
            clips = load_split(str(ann_dir / f"train_split{s}.txt"))
            ds = EGTEADataset(
                split_file         = f"train_split{s}.txt",
                clips              = clips,
                sliding_window     = False,
                temporal_aug_range = hp.temporal_aug_range,
                temporal_aug_prob  = hp.temporal_aug_prob,
                feat_noise_std     = hp.feat_noise_std,
                feat_drop_prob     = hp.feat_drop_prob,
                **common,
            )
            train_datasets.append(ds)
            print(f"[DataModule] Train split{s}: {len(clips)} clip")

        self.train_ds = (
            ConcatDataset(train_datasets) if len(train_datasets) > 1
            else train_datasets[0]
        )

        val_clips = load_split(str(ann_dir / f"test_split{hp.val_split}.txt"))
        self.val_ds = EGTEADataset(
            split_file         = f"test_split{hp.val_split}.txt",
            clips              = val_clips,
            sliding_window     = hp.sliding_window,
            stride             = hp.stride,
            temporal_aug_range = 0,
            temporal_aug_prob  = 0.0,
            **common,
        )
        self.test_ds = self.val_ds

        n_train = sum(len(d) for d in train_datasets)
        print(
            f"[DataModule] Fold: train={hp.train_splits} val/test={hp.val_split} | "
            f"train clips: {n_train} | val/test windows: {len(self.val_ds)}"
        )

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
        )
