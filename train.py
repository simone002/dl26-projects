"""
Punto di ingresso per il training.

Uso:
    python train.py --config experiments/configs/xlstm.yaml
    python train.py --config experiments/configs/lstm.yaml
    python train.py --config experiments/configs/cnn1d.yaml
    python train.py --config experiments/configs/mamba.yaml
    python train.py --config experiments/configs/xlstm.yaml model.hidden=256
"""


import torch

torch.set_float32_matmul_precision('high')
import yaml
import argparse
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


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "base" in cfg:
        base_path = cfg.pop("base")
        with open(base_path) as f:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/xlstm.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    # --- Data ---
    datamodule = EGTEADataModule(**cfg["data"])
    datamodule.setup("fit")
    print(
        "[Train] Split summary -> "
        f"train: {len(datamodule.train_ds)} | "
        f"val: {len(datamodule.val_ds)} | "
        f"test: {len(datamodule.test_ds)}"
    )

    # --- Model ---
    model     = build_model(cfg)
    lit_model = TemporalSegmentationModule(
        model           = model,
        num_classes     = cfg["model"]["num_classes"],
        lr              = cfg["training"]["lr"],
        weight_decay    = cfg["training"]["weight_decay"],
        label_smoothing = cfg["training"]["label_smoothing"],
        smooth_weight   = cfg["training"].get("smooth_weight", 0.2),
        boundary_weight = cfg["training"].get("boundary_weight", 0.3),
    )

    # --- Logger W&B ---
    logger = WandbLogger(
        project  = cfg["wandb"]["project"],
        name     = cfg["wandb"]["name"],
        log_model= True,
        config   = cfg,
    )
    logger.watch(model, log="gradients", log_freq=50)

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            monitor    = "val/edit_score",
            mode       = "max",
            save_top_k = 3,
            filename   = "{epoch:02d}-{val/edit_score:.3f}",
        ),
        EarlyStopping(
            monitor  = "val/edit_score",
            patience = 20,
            mode     = "max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs        = cfg["training"]["max_epochs"],
        logger            = logger,
        callbacks         = callbacks,
        accelerator       = "cuda",
        log_every_n_steps = 10,
        num_sanity_val_steps = 0,   
        gradient_clip_val    = 1.0,   
    )

    trainer.fit(lit_model, datamodule=datamodule)

    print("\n[Post-fit] Evaluating on TEST split...")
    lit_model.test_prefix = "test"
    trainer.test(lit_model, dataloaders=datamodule.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    main()