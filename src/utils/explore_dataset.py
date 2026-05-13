"""
Analisi completa del dataset EGTEA Gaze+.
Esegui con: python explore_dataset.py

Genera:
  - dataset_analysis.png      (background/foreground + top classi)
  - clip_duration_analysis.png (distribuzione durate + CDF)
  - class_distribution.png    (distribuzione classi foreground)
  - temporal_density.png      (densità azioni nel tempo)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict
from pathlib import Path
from src.datasets.dataset import EGTEADataset, load_action_labels, load_split

# ── Config ────────────────────────────────────────────────────────────────────
LMDB_RGB       = r"D:\egtea\TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd"
ANNOTATION_DIR = r".\action_annotation"
SPLIT_FILE     = "train_split1.txt"
SEQ_LEN        = 128
N_SAMPLES      = 8299
# ──────────────────────────────────────────────────────────────────────────────


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


def main():
    print("Caricamento dataset...")
    ds = EGTEADataset(
        lmdb_rgb_path  = LMDB_RGB,
        annotation_dir = ANNOTATION_DIR,
        split_file     = SPLIT_FILE,
        seq_len        = SEQ_LEN,
    )
    class_names = load_class_names(ANNOTATION_DIR)
    clips       = load_split(str(Path(ANNOTATION_DIR) / SPLIT_FILE))
    anns        = load_action_labels(
        str(Path(ANNOTATION_DIR) / "raw_annotations" / "action_labels.csv")
    )

    n = min(N_SAMPLES, len(ds))
    print(f"Analisi su {n} clip...")

    # Raccogli statistiche
    bg_pcts      = []
    fg_pcts      = []
    class_counts = Counter()
    action_durations = defaultdict(list)   # durata per classe
    clips_per_class  = defaultdict(int)    # clip che contengono ogni classe

    for i in range(n):
        _, labels = ds[i]
        labels_np = labels.numpy()
        bg = (labels_np == 0).sum()
        fg = (labels_np != 0).sum()
        bg_pcts.append(bg / SEQ_LEN)
        fg_pcts.append(fg / SEQ_LEN)

        seen = set()
        i_start = None
        prev = 0
        for t, c in enumerate(labels_np):
            if c != 0:
                class_counts[int(c)] += 1
                seen.add(int(c))
                if prev == 0:
                    i_start = t
            elif prev != 0 and i_start is not None:
                action_durations[int(prev)].append(t - i_start)
                i_start = None
            prev = int(c)
        if prev != 0 and i_start is not None:
            action_durations[int(prev)].append(SEQ_LEN - i_start)

        for c in seen:
            clips_per_class[c] += 1

    bg_pcts   = np.array(bg_pcts)
    fg_pcts   = np.array(fg_pcts)
    durations = np.array([c["frame_end"] - c["frame_start"] for c in clips])

    # ── Stampa statistiche ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STATISTICHE DATASET EGTEA Gaze+")
    print(f"{'='*60}")
    print(f"  Clip totali (train split1): {len(clips)}")
    print(f"  Clip analizzati:            {n}")
    print(f"  Classi uniche nel split:    {len(ds.clips)} clip → "
          f"{len(class_counts)} classi osservate")
    print(f"\n  Distribuzione background/foreground:")
    print(f"  Media background per clip:  {bg_pcts.mean():.1%}")
    print(f"  Media foreground per clip:  {fg_pcts.mean():.1%}")
    print(f"  Clip con 100% background:   "
          f"{(bg_pcts == 1.0).sum()} ({(bg_pcts == 1.0).mean():.1%})")
    print(f"\n  Durate clip (frame):")
    print(f"  Media:    {durations.mean():.0f}")
    print(f"  Mediana:  {np.median(durations):.0f}")
    print(f"  75° perc: {np.percentile(durations, 75):.0f}")
    print(f"  Max:      {durations.max():.0f}")
    print(f"\n  Top 10 classi più frequenti:")
    for cls_id, count in class_counts.most_common(10):
        name = class_names.get(cls_id, f"class_{cls_id}")
        n_clips = clips_per_class[cls_id]
        print(f"    [{cls_id:3d}] {name:<35} {count:6d} frame  {n_clips:4d} clip")
    print(f"\n  Classi rare (< 5 clip):")
    rare = [(c, n) for c, n in clips_per_class.items() if n < 5]
    print(f"  {len(rare)} classi appaiono in meno di 5 clip")
    print(f"{'='*60}\n")

    # ── FIGURA 1: Background/Foreground ───────────────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle("EGTEA Gaze+ — Distribuzione Background vs Foreground",
                  fontsize=13, fontweight="bold")

    # Pie chart
    ax = axes[0]
    total_bg = bg_pcts.sum()
    total_fg = fg_pcts.sum()
    ax.pie(
        [total_bg, total_fg],
        labels     = [f"Background\n{total_bg/(total_bg+total_fg):.1%}",
                      f"Foreground\n{total_fg/(total_bg+total_fg):.1%}"],
        colors     = ["#B0C4DE", "#E07B54"],
        startangle = 90,
        wedgeprops = {"edgecolor": "white", "linewidth": 2},
        textprops  = {"fontsize": 11},
    )
    ax.set_title("Distribuzione globale")

    # Istogramma % background per clip
    ax = axes[1]
    ax.hist(bg_pcts * 100, bins=25, color="#B0C4DE", edgecolor="white", linewidth=0.5)
    ax.axvline(bg_pcts.mean() * 100, color="#333", linestyle="--",
               linewidth=1.5, label=f"Media {bg_pcts.mean():.1%}")
    ax.set_xlabel("% Background per clip")
    ax.set_ylabel("Numero di clip")
    ax.set_title("Distribuzione background per clip")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Top 15 classi
    ax = axes[2]
    top15   = class_counts.most_common(15)
    names15 = [class_names.get(c, str(c))[:22] for c, _ in top15]
    cnt15   = [cnt for _, cnt in top15]
    ax.barh(names15[::-1], cnt15[::-1], color="#E07B54",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Frame totali")
    ax.set_title("Top 15 classi più frequenti")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=150, bbox_inches="tight")
    print("Salvato: dataset_analysis.png")

    # ── FIGURA 2: Durate clip ─────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("EGTEA Gaze+ — Distribuzione durate clip",
                  fontsize=13, fontweight="bold")

    ax = axes2[0]
    ax.hist(durations, bins=50, color="#7B9EBF", edgecolor="white", linewidth=0.5)
    ax.axvline(np.median(durations), color="red", linestyle="--",
               linewidth=1.5, label=f"Mediana {np.median(durations):.0f}")
    ax.axvline(np.percentile(durations, 75), color="orange", linestyle="--",
               linewidth=1.5, label=f"75° perc {np.percentile(durations, 75):.0f}")
    ax.axvline(128, color="green", linestyle="-", linewidth=2,
               label="seq_len=128 (scelto)")
    ax.set_xlabel("Durata clip (frame)")
    ax.set_ylabel("Numero di clip")
    ax.set_title("Distribuzione durate clip")
    ax.set_xlim(0, 600)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes2[1]
    sorted_dur = np.sort(durations)
    cdf = np.arange(1, len(sorted_dur) + 1) / len(sorted_dur)
    ax.plot(sorted_dur, cdf * 100, color="#7B9EBF", linewidth=2)
    ax.axvline(128, color="green", linestyle="-", linewidth=2,
               label=f"seq_len=128 → copre {(durations<=128).mean():.1%}")
    ax.axvline(256, color="orange", linestyle="--", linewidth=1.5,
               label=f"seq_len=256 → copre {(durations<=256).mean():.1%}")
    ax.set_xlabel("Durata clip (frame)")
    ax.set_ylabel("% clip coperti (CDF)")
    ax.set_title("Copertura in funzione di seq_len")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("clip_duration_analysis.png", dpi=150, bbox_inches="tight")
    print("Salvato: clip_duration_analysis.png")

    # ── FIGURA 3: Distribuzione classi ────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5))
    fig3.suptitle("EGTEA Gaze+ — Distribuzione classi foreground",
                  fontsize=13, fontweight="bold")

    # Distribuzione ordinata (tutte le classi)
    ax = axes3[0]
    all_classes = sorted(class_counts.keys())
    all_counts  = [class_counts[c] for c in all_classes]
    sorted_idx  = np.argsort(all_counts)[::-1]
    sorted_counts = [all_counts[i] for i in sorted_idx]
    ax.bar(range(len(sorted_counts)), sorted_counts,
           color="#7B9EBF", edgecolor="none", width=1.0)
    ax.set_xlabel("Classi (ordinate per frequenza)")
    ax.set_ylabel("Frame totali")
    ax.set_title(f"Distribuzione frequenza classi ({len(all_classes)} classi osservate)")
    ax.axhline(np.mean(sorted_counts), color="red", linestyle="--",
               linewidth=1.5, label=f"Media {np.mean(sorted_counts):.0f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Durata media per classe (top 15)
    ax = axes3[1]
    class_avg_dur = {c: np.mean(d) for c, d in action_durations.items()
                     if len(d) >= 3}
    top_dur = sorted(class_avg_dur.items(), key=lambda x: x[1], reverse=True)[:15]
    dur_names = [class_names.get(c, str(c))[:22] for c, _ in top_dur]
    dur_vals  = [d for _, d in top_dur]
    ax.barh(dur_names[::-1], dur_vals[::-1], color="#5B8A6E",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Durata media (frame)")
    ax.set_title("Top 15 classi per durata media")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
    print("Salvato: class_distribution.png")

    # ── FIGURA 4: Clip con 0 foreground ───────────────────────────────────────
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle("EGTEA Gaze+ — Analisi sbilanciamento",
                  fontsize=13, fontweight="bold")

    # % foreground per clip (scatter)
    ax = axes4[0]
    ax.scatter(range(n), sorted(fg_pcts * 100), s=2,
               color="#E07B54", alpha=0.5)
    ax.axhline(fg_pcts.mean() * 100, color="red", linestyle="--",
               linewidth=1.5, label=f"Media {fg_pcts.mean():.1%}")
    ax.set_xlabel("Clip (ordinati per % foreground)")
    ax.set_ylabel("% Foreground")
    ax.set_title("Foreground per clip")
    ax.legend()
    ax.grid(alpha=0.3)

    # Clips con 0% foreground
    ax = axes4[1]
    zero_fg = (bg_pcts == 1.0).sum()
    low_fg  = ((bg_pcts > 0.9) & (bg_pcts < 1.0)).sum()
    mid_fg  = ((bg_pcts > 0.5) & (bg_pcts <= 0.9)).sum()
    high_fg = (bg_pcts <= 0.5).sum()

    labels_pie = [
        f"100% BG\n({zero_fg})",
        f">90% BG\n({low_fg})",
        f">50% BG\n({mid_fg})",
        f"≤50% BG\n({high_fg})",
    ]
    sizes  = [zero_fg, low_fg, mid_fg, high_fg]
    colors_pie = ["#2C3E50", "#7F8C8D", "#B0C4DE", "#E07B54"]
    ax.pie(sizes, labels=labels_pie, colors=colors_pie,
           startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
           textprops={"fontsize": 9})
    ax.set_title(f"Composizione clip per % background\n(su {n} clip analizzati)")

    plt.tight_layout()
    plt.savefig("imbalance_analysis.png", dpi=150, bbox_inches="tight")
    print("Salvato: imbalance_analysis.png")

    plt.show()
    print("\nAnalisi completata!")


if __name__ == "__main__":
    main()