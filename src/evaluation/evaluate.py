"""
Analisi qualitativa degli errori sistematici.

Uso (dalla root del progetto):
    python -m src.evaluation.evaluate --checkpoint path/to/checkpoint.ckpt --config experiments/configs/mstcn.yaml
    python -m src.evaluation.evaluate --checkpoint path/to/checkpoint.ckpt --config experiments/configs/lstm.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.datasets.dataset  import EGTEADataset
from src.models.cnn1d      import CNN1DModel
from src.models.lstm       import LSTMModel
from src.models.mamba      import MambaModel
from src.models.xlstm      import xLSTMModel
from src.models.mstcn      import MSTCNModel
from src.training.module   import TemporalSegmentationModule, edit_score


# ── Config ────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base_path = cfg.pop("base")
        with open(base_path, encoding="utf-8-sig") as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


# ── Utilità ───────────────────────────────────────────────────────────────────

def load_class_names(annotation_dir: str) -> dict:
    cls_path = Path(annotation_dir) / "raw_annotations" / "cls_label_index.csv"
    mapping  = {0: "background"}
    with open(cls_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 2:
                try:
                    mapping[int(parts[0])] = parts[1]
                except ValueError:
                    continue
    return mapping


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


def get_segments(seq: np.ndarray, bg_class: int = 0) -> list[dict]:
    segments, i = [], 0
    while i < len(seq):
        label, j = seq[i], i
        while j < len(seq) and seq[j] == label:
            j += 1
        if label != bg_class:
            segments.append({"label": label, "start": i, "end": j - 1})
        i = j
    return segments


# ── Analisi errori di confine ─────────────────────────────────────────────────

def analyze_boundary_errors(pred: np.ndarray, target: np.ndarray) -> list[dict]:
    errors    = []
    gt_segs   = get_segments(target)
    pred_segs = get_segments(pred)

    for gt in gt_segs:
        best, best_overlap = None, 0
        for ps in pred_segs:
            if ps["label"] != gt["label"]:
                continue
            overlap = max(0, min(ps["end"], gt["end"]) - max(ps["start"], gt["start"]) + 1)
            if overlap > best_overlap:
                best_overlap, best = overlap, ps

        if best is not None and best_overlap > 0:
            errors.append({
                "label":       gt["label"],
                "start_error": best["start"] - gt["start"],
                "end_error":   best["end"]   - gt["end"],
                "gt_duration": gt["end"] - gt["start"] + 1,
            })
    return errors


# ── Visualizzazione clip ──────────────────────────────────────────────────────

def plot_clip(pred: np.ndarray, target: np.ndarray,
              class_names: dict, title: str, save_path: str,
              probs: np.ndarray | None = None):
    n_rows  = 3 if probs is not None else 2
    heights = [1, 1, 0.5] if probs is not None else [1, 1]
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(16, 2 + n_rows * 1.3), sharex=True,
        gridspec_kw={"height_ratios": heights},
    )
    fig.suptitle(title, fontsize=10, y=1.01)

    unique = sorted(set(target.tolist() + pred.tolist()) - {0})
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique), 1)))
    cmap   = {c: colors[i] for i, c in enumerate(unique)}
    cmap[0] = (0.92, 0.92, 0.92, 1.0)

    T = len(target)
    for ax, seq, label in zip(axes[:2], [target, pred], ["Ground Truth", "Predizione"]):
        ax.set_xlim(0, T)
        ax.set_ylim(0, 1)
        ax.set_ylabel(label, fontsize=9)
        ax.set_yticks([])
        i = 0
        while i < T:
            c, j = seq[i], i
            while j < T and seq[j] == c:
                j += 1
            ax.axvspan(i, j, alpha=0.85, color=cmap.get(c, (0.5, 0.5, 0.5)))
            if c != 0 and (j - i) > 4:
                name = class_names.get(c, str(c))[:15]
                ax.text((i + j) / 2, 0.5, name, ha="center", va="center",
                        fontsize=7, color="black")
            i = j

    if probs is not None:
        ax = axes[2]
        ax.fill_between(range(T), probs, alpha=0.6, color="#4a9e6b")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", ".5", "1"], fontsize=7)
        ax.set_ylabel("Conf.", fontsize=9)

    axes[-1].set_xlabel("Frame", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def _softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path al .ckpt")
    parser.add_argument("--config",     default="experiments/configs/mstcn.yaml")
    parser.add_argument("--split",      type=int, default=None,
                        help="Split EGTEA da valutare (1,2,3); default: val_split da config")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Numero max di clip da analizzare (default: tutti)")
    parser.add_argument("--sliding-window", action="store_true", default=True)
    parser.add_argument("--no-sliding-window", dest="sliding_window", action="store_false")
    parser.add_argument("--stride",     type=int, default=None)
    parser.add_argument("--n_plots",    type=int, default=50,
                        help="Max plot da salvare; 0 = tutti")
    args = parser.parse_args()

    cfg = load_config(args.config)

    features_dir   = cfg["data"]["features_dir"]
    annotation_dir = cfg["data"]["annotation_dir"]
    seq_len        = cfg["data"]["seq_len"]
    feat_dim       = cfg["data"]["feat_dim"]

    if args.split is not None:
        split = args.split
    else:
        import re
        m = re.search(r"fold(\d+)", Path(args.checkpoint).name)
        split = int(m.group(1)) if m else cfg["data"]["val_split"]
        if m:
            # fold1→split3, fold2→split2, fold3→split1
            fold_to_split = {1: 3, 2: 2, 3: 1}
            split = fold_to_split.get(int(m.group(1)), cfg["data"]["val_split"])
    num_classes    = cfg["model"]["num_classes"]
    stride         = args.stride if args.stride is not None else seq_len // 2

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(annotation_dir)
    model_name  = cfg["model"]["name"]

    print(f"\nModello:     {model_name.upper()}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Split:       {split}  {'(rilevato dal nome checkpoint)' if args.split is None else '(da --split)'}")
    print(f"Device:      {device}")

    backbone  = build_model(cfg)
    lit_model = TemporalSegmentationModule.load_from_checkpoint(
        args.checkpoint, model=backbone, map_location=device
    )
    lit_model.eval()
    lit_model.to(device)

    ds = EGTEADataset(
        features_dir   = features_dir,
        annotation_dir = annotation_dir,
        split_file     = f"test_split{split}.txt",
        seq_len        = seq_len,
        feat_dim       = feat_dim,
        sliding_window = args.sliding_window,
        stride         = stride,
    )

    run_id  = Path(args.checkpoint).parent.parent.parent.name
    out_dir = Path(f"eval/{model_name}_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_boundary_errors = []
    confusion           = defaultdict(lambda: defaultdict(int))
    plots_saved         = 0
    pending_plots       = []

    if args.sliding_window:
        clip_to_windows: dict[int, list] = defaultdict(list)
        for sample_idx, (clip_idx, window_start) in enumerate(ds._samples):
            clip_to_windows[clip_idx].append((sample_idx, window_start))

        clip_indices = sorted(clip_to_windows.keys())
        n_clips = len(clip_indices) if args.n_samples is None \
                  else min(args.n_samples, len(clip_indices))

        print(f"Clip totali: {len(ds.clips)} | Finestre totali: {len(ds)}")
        print(f"Modalità: sliding window + aggregazione logit (stride={stride})")
        print(f"Clip analizzati: {n_clips}\n")

        with torch.no_grad():
            for ci, clip_idx in enumerate(clip_indices[:n_clips]):
                clip    = ds.clips[clip_idx]
                T       = clip["frame_end"] - clip["frame_start"] + 1
                windows = clip_to_windows[clip_idx]

                logit_sum = np.zeros((T, num_classes), dtype=np.float32)
                logit_cnt = np.zeros(T,                dtype=np.float32)

                for sample_idx, window_start in windows:
                    feat, _ = ds[sample_idx]
                    actual_len = min(seq_len, T - window_start)
                    logits_np  = lit_model(feat.unsqueeze(0).to(device)) \
                                     .squeeze(0).cpu().numpy()
                    logit_sum[window_start:window_start + actual_len] += \
                        logits_np[:actual_len]
                    logit_cnt[window_start:window_start + actual_len] += 1

                averaged = logit_sum / np.maximum(logit_cnt[:, None], 1)
                preds_np = averaged.argmax(axis=-1)
                probs_np = _softmax_np(averaged).max(axis=-1)

                labels_np = ds._build_dense_labels(
                    clip["video_session"], clip["frame_start"], clip["frame_end"]
                )

                all_boundary_errors.extend(analyze_boundary_errors(preds_np, labels_np))
                for gt, pr in zip(labels_np, preds_np):
                    if gt != 0:
                        confusion[int(gt)][int(pr)] += 1

                has_fg    = (labels_np != 0).any()
                under_cap = (args.n_plots == 0) or (plots_saved < args.n_plots)
                if has_fg and under_cap:
                    edit_clip = edit_score(preds_np, labels_np)
                    pending_plots.append(dict(
                        pred=preds_np, target=labels_np, class_names=class_names,
                        title=(f"#{plots_saved} — {model_name.upper()} "
                               f"clip {clip_idx} ({len(windows)} finestre) "
                               f"| edit={edit_clip:.2f}"),
                        save_path=str(out_dir / f"clip_{plots_saved:03d}.png"),
                        probs=probs_np,
                    ))
                    plots_saved += 1

                if (ci + 1) % 50 == 0:
                    print(f"  {ci + 1}/{n_clips} clip... ({plots_saved} plot salvati)")

    else:
        n = len(ds) if args.n_samples is None else min(args.n_samples, len(ds))
        print(f"Dataset: {len(ds)} sample | Analizzati: {n}\n")

        with torch.no_grad():
            for i in range(n):
                feat, labels = ds[i]
                labels_np    = labels.numpy()

                logits   = lit_model(feat.unsqueeze(0).to(device))
                probs_np = torch.softmax(logits, dim=-1).max(-1).values \
                               .squeeze(0).cpu().numpy()
                preds_np = logits.argmax(-1).squeeze(0).cpu().numpy()

                all_boundary_errors.extend(analyze_boundary_errors(preds_np, labels_np))
                for gt, pr in zip(labels_np, preds_np):
                    if gt != 0:
                        confusion[int(gt)][int(pr)] += 1

                has_fg    = (labels_np != 0).any()
                under_cap = (args.n_plots == 0) or (plots_saved < args.n_plots)
                if has_fg and under_cap:
                    edit_clip = edit_score(preds_np, labels_np)
                    pending_plots.append(dict(
                        pred=preds_np, target=labels_np, class_names=class_names,
                        title=(f"#{plots_saved} — {model_name.upper()} (idx={i}) "
                               f"| edit={edit_clip:.2f}"),
                        save_path=str(out_dir / f"clip_{plots_saved:03d}.png"),
                        probs=probs_np,
                    ))
                    plots_saved += 1

                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{n} sample... ({plots_saved} plot salvati)")

    # ── Report errori sistematici ─────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ANALISI ERRORI SISTEMATICI — {model_name.upper()}")
    print(sep)

    if all_boundary_errors:
        start_errors = np.array([e["start_error"] for e in all_boundary_errors])
        end_errors   = np.array([e["end_error"]   for e in all_boundary_errors])

        print(f"\n  Errori di confine (frame, + = ritardo, - = anticipo):")
        print(f"  Inizio: media {np.mean(start_errors):+.1f}  "
              f"std {np.std(start_errors):.1f}  "
              f"mediana {np.median(start_errors):+.1f}")
        print(f"  Fine:   media {np.mean(end_errors):+.1f}  "
              f"std {np.std(end_errors):.1f}  "
              f"mediana {np.median(end_errors):+.1f}")

        print()
        for label, errors in [("inizio", start_errors), ("fine", end_errors)]:
            m = np.mean(errors)
            if m > 2:
                print(f"  Il modello predice l'{label} con {m:.1f} frame di RITARDO")
            elif m < -2:
                print(f"  Il modello predice l'{label} con {abs(m):.1f} frame di ANTICIPO")
            else:
                print(f"  Errore di {label} contenuto ({m:+.1f} frame)")

        print(f"\n  Top 5 classi con errore di confine maggiore:")
        class_errors = defaultdict(list)
        for e in all_boundary_errors:
            class_errors[e["label"]].append(
                abs(e["start_error"]) + abs(e["end_error"])
            )
        for cls_id, errs in sorted(class_errors.items(),
                                    key=lambda x: np.mean(x[1]),
                                    reverse=True)[:5]:
            name = class_names.get(cls_id, str(cls_id))
            print(f"    [{cls_id:3d}] {name:<35} "
                  f"errore medio: {np.mean(errs):.1f} frame  ({len(errs)} segmenti)")

    print(f"\n  Top 5 confusioni tra classi:")
    conf_list = [
        (gt, pr, cnt)
        for gt, pd in confusion.items()
        for pr, cnt in pd.items()
        if pr != gt and pr != 0
    ]
    conf_list.sort(key=lambda x: x[2], reverse=True)
    for gt_c, pr_c, cnt in conf_list[:5]:
        gt_n = class_names.get(gt_c, str(gt_c))
        pr_n = class_names.get(pr_c, str(pr_c))
        print(f"    '{gt_n}' -> '{pr_n}'  ({cnt} frame)")

    print(f"\n{sep}\n")

    if all_boundary_errors:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Distribuzione errori di confine — {model_name.upper()}")

        for ax, errors, title, color in [
            (axes[0], start_errors, "Errore inizio azione", "#5B8DB8"),
            (axes[1], end_errors,   "Errore fine azione",   "#E07B54"),
        ]:
            ax.hist(errors, bins=30, color=color, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="zero")
            ax.axvline(np.mean(errors), color="orange", linestyle="--",
                       linewidth=1.5, label=f"media {np.mean(errors):+.1f}")
            ax.set_title(title)
            ax.set_xlabel("Ritardo (+) / Anticipo (-) in frame")
            ax.set_ylabel("Conteggio")
            ax.legend()

        plt.tight_layout()
        plt.savefig(str(out_dir / "boundary_errors.png"), dpi=150, bbox_inches="tight")
        print(f"  Grafici salvati in: {out_dir}/")
        plt.show()

    for p in pending_plots:
        plot_clip(p["pred"], p["target"], p["class_names"],
                  title=p["title"], save_path=p["save_path"], probs=p["probs"])


if __name__ == "__main__":
    main()
