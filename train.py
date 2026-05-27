"""
3-fold cross-validation su EGTEA Gaze+ con feature DINOv3.

Uso:
    python train_cv.py --config experiments/configs/mstcn.yaml
    python train_cv.py --config experiments/configs/mstcn.yaml training.max_epochs=50

I 3 fold seguono il protocollo ufficiale EGTEA:
  Fold 1: train=split1+split2  val/test=split3
  Fold 2: train=split1+split3  val/test=split2
  Fold 3: train=split2+split3  val/test=split1
"""

import torch
torch.set_float32_matmul_precision("high")

import os
import yaml
import argparse
import numpy as np
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers   import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.datasets.datamodule import EGTEADataModule
from src.models.cnn1d        import CNN1DModel
from src.models.lstm         import LSTMModel
from src.models.xlstm        import xLSTMModel
from src.models.mamba        import MambaModel
from src.models.mstcn        import MSTCNModel
from src.training.module     import TemporalSegmentationModule


FOLDS = [
    {"train_splits": [1, 2], "val_split": 3},
    {"train_splits": [1, 3], "val_split": 2},
    {"train_splits": [2, 3], "val_split": 1},
]

METRICS = [
    "acc",
    "edit_score",
    "F1@10",
    "F1@25",
    "F1@50",
    "boundary_f1",
]


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base_path = cfg.pop("base")
        with open(base_path, encoding="utf-8-sig") as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)
    for ov in overrides:
        key, val = ov.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        d[keys[-1]] = val
    return cfg


def build_model(cfg: dict):
    m    = cfg["model"]
    name = m["name"]
    if name == "cnn1d":
        return CNN1DModel(
            feat_dim    = m["feat_dim"],
            num_classes = m["num_classes"],
            hidden      = m["hidden"],
            n_layers    = m.get("n_layers",    4),
            kernel_size = m.get("kernel_size", 3),
            dropout     = m.get("dropout",     0.5),
        )
    elif name == "lstm":
        return LSTMModel(
            feat_dim      = m["feat_dim"],
            num_classes   = m["num_classes"],
            hidden        = m["hidden"],
            n_layers      = m.get("n_layers",      2),
            dropout       = m.get("dropout",       0.5),
            bidirectional = m.get("bidirectional", True),
        )
    elif name == "xlstm":
        return xLSTMModel(
            feat_dim    = m["feat_dim"],
            num_classes = m["num_classes"],
            hidden      = m["hidden"],
            n_layers    = m.get("n_layers", 2),
            dropout     = m.get("dropout",  0.5),
        )
    elif name == "mamba":
        return MambaModel(
            feat_dim    = m["feat_dim"],
            num_classes = m["num_classes"],
            hidden      = m["hidden"],
            n_layers    = m.get("n_layers", 2),
            d_state     = m.get("d_state",  16),
            d_conv      = m.get("d_conv",   4),
            expand      = m.get("expand",   2),
            dropout     = m.get("dropout",  0.5),
        )
    elif name == "mstcn":
        return MSTCNModel(
            feat_dim    = m["feat_dim"],
            num_classes = m["num_classes"],
            hidden      = m["hidden"],
            n_stages    = m.get("n_stages", 4),
            n_layers    = m.get("n_layers", 10),
            dropout     = m.get("dropout",  0.5),
        )
    else:
        raise ValueError(f"Modello non riconosciuto: {name}")


def run_fold(fold_idx: int, fold: dict, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx+1}/3 — train={fold['train_splits']}  val/test={fold['val_split']}")
    print(f"{'='*60}\n")

    cfg["data"]["train_splits"] = fold["train_splits"]
    cfg["data"]["val_split"]    = fold["val_split"]

    datamodule = EGTEADataModule(**cfg["data"])
    datamodule.setup("fit")

    backbone  = build_model(cfg)
    lit_model = TemporalSegmentationModule(
        model           = backbone,
        num_classes     = cfg["model"]["num_classes"],
        lr              = cfg["training"]["lr"],
        weight_decay    = cfg["training"]["weight_decay"],
        label_smoothing = cfg["training"]["label_smoothing"],
        smooth_weight   = cfg["training"].get("smooth_weight",   0.2),
        boundary_weight = cfg["training"].get("boundary_weight", 0.3),
    )

    run_name = f"{cfg['wandb']['name']}-fold{fold_idx+1}"
    logger = WandbLogger(
        project  = cfg["wandb"]["project"],
        name     = run_name,
        group    = cfg["wandb"]["name"],
        save_dir = "experiments/logs",
        log_model= False,
        config   = {**cfg, "fold": fold_idx + 1},
    )

    ckpt_dir = os.path.join(
        "experiments", "checkpoints",
        cfg["wandb"]["project"], str(logger.version), "checkpoints",
    )
    callbacks = [
        ModelCheckpoint(
            dirpath    = ckpt_dir,
            monitor    = "val/edit_score",
            mode       = "max",
            save_top_k = 1,
            filename   = f"fold{fold_idx+1}-{{epoch:02d}}-{{val/edit_score:.1f}}",
        ),
        EarlyStopping(
            monitor  = "val/edit_score",
            patience = 20,
            mode     = "max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs           = cfg["training"]["max_epochs"],
        logger               = logger,
        callbacks            = callbacks,
        default_root_dir     = "experiments/checkpoints",
        accelerator          = "cuda",
        log_every_n_steps    = 10,
        num_sanity_val_steps = 0,
        gradient_clip_val    = 1.0,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    lit_model.test_prefix = f"fold{fold_idx+1}_test"
    results = trainer.test(
        lit_model,
        dataloaders = datamodule.test_dataloader(),
        ckpt_path   = "best",
        verbose     = False,
    )

    metrics = results[0] if results else {}
    fold_metrics = {}
    for m_name in METRICS:
        key = f"fold{fold_idx+1}_test/{m_name}"
        if key in metrics:
            fold_metrics[m_name] = metrics[key]

    print(f"\n  Fold {fold_idx+1} risultati:")
    for k, v in fold_metrics.items():
        print(f"    {k:<22}: {v*100:.1f}%")

    wandb.finish()
    return fold_metrics


ALL_MODELS = ["cnn1d", "lstm", "xlstm", "mamba", "mstcn"]


def run_all_folds(config_path: str, overrides: list[str]):
    cfg = load_config(config_path, overrides)
    model_name = cfg["model"]["name"].upper()

    all_fold_metrics = []
    for i, fold in enumerate(FOLDS):
        fold_metrics = run_fold(i, fold, cfg)
        all_fold_metrics.append(fold_metrics)

    print(f"\n{'='*60}")
    print(f"  RISULTATI FINALI — {model_name} — media su 3 fold")
    print(f"{'='*60}")
    print(f"  {'Metrica':<22}  {'Media':>8}  {'Std':>8}")
    print(f"  {'-'*40}")
    for m_name in METRICS:
        vals = [fm[m_name] for fm in all_fold_metrics if m_name in fm]
        if vals:
            mean = np.mean(vals) * 100
            std  = np.std(vals)  * 100
            print(f"  {m_name:<22}  {mean:>7.1f}%  {std:>7.1f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="3-fold CV su EGTEA. Senza --model allena tutti e 5 i modelli."
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=ALL_MODELS,
        help="Modello da allenare (default: tutti). Carica experiments/configs/{model}.yaml",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path esplicito al config YAML (alternativa a --model)",
    )
    parser.add_argument("overrides", nargs="*",
                        help="Override dotted: training.lr=0.0002 model.hidden=256")
    args = parser.parse_args()

    if args.config is not None:
        configs = [args.config]
    elif args.model is not None:
        configs = [f"experiments/configs/{args.model}.yaml"]
    else:
        configs = [f"experiments/configs/{m}.yaml" for m in ALL_MODELS]

    for config_path in configs:
        run_all_folds(config_path, args.overrides)


if __name__ == "__main__":
    main()
