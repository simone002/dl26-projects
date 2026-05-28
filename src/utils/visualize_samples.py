"""
Visualizzazione esempi del training set EGTEA Gaze+.
Uso:
    python -m src.utils.visualize_samples
    python -m src.utils.visualize_samples --n_clips 4 --split train_split1.txt
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from pathlib import Path

from src.datasets.dataset import EGTEADataset


def _load_base_config(path: str = "experiments/configs/base.yaml") -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def load_class_names(annotation_dir: str) -> dict:
    cls_path = Path(annotation_dir) / "raw_annotations" / "cls_label_index.csv"
    mapping  = {0: "Background"}
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


def get_color_map(labels_list: list) -> dict:
    all_classes = sorted(set(c for labels in labels_list
                             for c in labels.tolist() if c != 0))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(all_classes), 1)))
    cmap   = {c: colors[i] for i, c in enumerate(all_classes)}
    cmap[0] = (0.90, 0.90, 0.90, 1.0)
    return cmap


def draw_label_bar(ax, labels, cmap, class_names, T):
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    i = 0
    while i < T:
        c, j = int(labels[i]), i
        while j < T and int(labels[j]) == c:
            j += 1
        ax.axvspan(i, j, alpha=0.9, color=cmap.get(c, (0.5, 0.5, 0.5)), linewidth=0)
        if c != 0 and (j - i) > 5:
            name = class_names.get(c, str(c))[:14]
            ax.text((i + j) / 2, 0.5, name,
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="none", edgecolor="none"))
        i = j


def draw_feature_pca(ax, feat, labels, cmap):
    """PCA 2D manuale sulle feature DINOv3 (d=768), colorata per classe."""
    X    = feat - feat.mean(axis=0)
    cov  = X.T @ X / len(X)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, np.argsort(vals)[::-1][:2]]
    proj = X @ vecs  # (T, 2)

    colors_pt = [cmap.get(int(c), (0.5, 0.5, 0.5)) for c in labels]
    ax.scatter(proj[:, 0], proj[:, 1], c=colors_pt, s=15, alpha=0.7, edgecolors="none")
    ax.set_xlabel("PC1", fontsize=7)
    ax.set_ylabel("PC2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Feature space (PCA)", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clips", type=int, default=4)
    parser.add_argument("--split",   type=str, default="train_split1.txt")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    cfg            = _load_base_config()
    features_dir   = cfg["data"]["features_dir"]
    annotation_dir = cfg["data"]["annotation_dir"]
    seq_len        = cfg["data"]["seq_len"]
    feat_dim       = cfg["data"]["feat_dim"]

    class_names = load_class_names(annotation_dir)

    print(f"Caricamento dataset — split: {args.split}")
    ds = EGTEADataset(
        features_dir   = features_dir,
        annotation_dir = annotation_dir,
        split_file     = args.split,
        seq_len        = seq_len,
        feat_dim       = feat_dim,
        sliding_window = False,
    )

    indices = [i for i in range(len(ds)) if (ds[i][1] != 0).any().item()]
    chosen  = np.random.choice(indices, size=min(args.n_clips, len(indices)), replace=False)

    samples = [(ds[i][0].numpy(), ds[i][1].numpy(), ds.clips[i]) for i in chosen]
    cmap    = get_color_map([s[1] for s in samples])

    n   = len(samples)
    fig = plt.figure(figsize=(16, n * 3.5))
    fig.suptitle(
        f"Esempi training set EGTEA Gaze+ — feature DINOv3 ViT-B (d={feat_dim})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row, (feat, labels, clip_info) in enumerate(samples):
        T = len(labels)

        ax_bar   = fig.add_subplot(n, 3, row * 3 + 1)
        ax_pca   = fig.add_subplot(n, 3, row * 3 + 2)
        ax_stats = fig.add_subplot(n, 3, row * 3 + 3)

        # ── Label bar ────────────────────────────────────────────────────────
        draw_label_bar(ax_bar, labels, cmap, class_names, T)
        fg_pct = (labels != 0).mean() * 100
        ax_bar.set_title(
            f"{clip_info['video_session']}\n"
            f"frames {clip_info['frame_start']}–{clip_info['frame_end']}  "
            f"| FG: {fg_pct:.0f}%",
            fontsize=8, loc="left",
        )
        ax_bar.set_xlabel("Frame", fontsize=7)

        present = sorted(set(labels.tolist()))
        patches = [mpatches.Patch(color=cmap.get(c, (0.5, 0.5, 0.5)),
                                  label=class_names.get(c, str(c))[:18])
                   for c in present]
        ax_bar.legend(handles=patches, loc="lower right",
                      fontsize=5, framealpha=0.7, ncol=2)

        # ── PCA ──────────────────────────────────────────────────────────────
        draw_feature_pca(ax_pca, feat, labels, cmap)

        # ── Statistiche clip ─────────────────────────────────────────────────
        counts    = Counter(labels[labels != 0].tolist())
        bg_cnt    = int((labels == 0).sum())
        all_names  = ["Background"] + [class_names.get(c, str(c))[:18] for c in counts]
        all_values = [bg_cnt] + list(counts.values())
        all_colors = [cmap[0]] + [cmap.get(c, (0.5, 0.5, 0.5)) for c in counts]

        y    = np.arange(len(all_names))
        bars = ax_stats.barh(y, all_values, color=all_colors, edgecolor="white", height=0.6)
        ax_stats.set_yticks(y)
        ax_stats.set_yticklabels(all_names, fontsize=7)
        ax_stats.set_xlabel("Frame count", fontsize=7)
        ax_stats.set_title("Distribuzione classi nel clip", fontsize=8)
        ax_stats.tick_params(labelsize=6)
        ax_stats.grid(axis="x", alpha=0.3)
        ax_stats.spines["top"].set_visible(False)
        ax_stats.spines["right"].set_visible(False)

        total = sum(all_values)
        for bar, val in zip(bars, all_values):
            ax_stats.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                          f"{val/total:.0%}", va="center", fontsize=6)

    plt.tight_layout()
    out_path = "figures/visualize_samples.png"
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Salvato: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
