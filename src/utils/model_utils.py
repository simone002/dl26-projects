"""
Utility condivise tra train.py ed evaluate.py: config loading e model dispatch.
"""

import yaml

from src.models.cnn1d  import CNN1DModel
from src.models.lstm   import LSTMModel
from src.models.xlstm  import xLSTMModel
from src.models.mamba  import MambaModel
from src.models.mstcn  import MSTCNModel


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base_path = cfg.pop("base")
        with open(base_path, encoding="utf-8-sig") as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)
    for ov in (overrides or []):
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
