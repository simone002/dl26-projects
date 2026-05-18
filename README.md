# Temporal Action Segmentation from Video

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information
- **Group ID**: G36
- **Project ID**: 13

## 📝 Project Description

Dense frame-level action segmentation on the EGTEA Gaze+ egocentric dataset. Given pre-extracted TSN features (1024-d per frame), the model predicts one of 106 action classes for every frame in the sequence. Four architectures are compared — CNN1D, LSTM, xLSTM, MS-TCN++ and Mamba — all sharing the same training pipeline with a combined Cross-Entropy + Smooth + Boundary loss.

> 📖 **Full Report**: task formulation, architecture details, metric definitions and results analysis → **[REPORT.md](docs/REPORT.md)**

---

## 🛠 Technical Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/simone002/dl26-projects.git
cd dl26-projects
conda env create -f environment.yml
conda activate temporal-action-seg
```

### 2. Dataset

Feature pre-extracted with TSN are distributed as LMDB archives.  
Download from the course dataset page and place them at:

```
D:/egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd
```

The path can be changed in `experiments/configs/base.yaml` (`egtea_root`).  
Split files and action labels are already included in `data/annotations/`.

### 3. Training

```bash
# MS-TCN++ (best performing model)
python train.py --config experiments/configs/mstcn.yaml

# MS-TCN++ su split specifici
python train.py --config experiments/configs/mstcn.yaml data.split=2
python train.py --config experiments/configs/mstcn.yaml data.split=3

# Mamba
python train.py --config experiments/configs/mamba.yaml

# xLSTM
python train.py --config experiments/configs/xlstm.yaml

# LSTM
python train.py --config experiments/configs/lstm.yaml

# CNN1D (baseline)
python train.py --config experiments/configs/cnn1d.yaml
```

Override any hyperparameter without editing files:

```bash
python train.py --config experiments/configs/mstcn.yaml training.lr=0.0002 model.hidden=256
```

Training logs and checkpoints are saved automatically via Weights & Biases.

### 4. Evaluation

Qualitative error analysis on a saved checkpoint:

```bash
python -m src.evaluation.evaluate \
    --checkpoint path/to/checkpoint.ckpt \
    --config experiments/configs/mstcn.yaml
```

Dataset statistics and figures:

```bash
python -m src.utils.explore_dataset
python -m src.utils.visualize_samples --n_clips 4
```

---

## 📊 Results

Test set, split 1. All values in %. † val metrics (test eval incomplete).

| Model    | mIoU | Edit Score | F1@10 | F1@25 | F1@50 | Boundary F1 |
|----------|:----:|:----------:|:-----:|:-----:|:-----:|:-----------:|
| CNN1D    |  3.4 |    5.9     |  5.3  |  3.6  |  2.5  |    21.2     |
| LSTM     |  4.5 |   11.1     | 10.9  |  9.8  |  8.4  |    27.4     |
| xLSTM   |  4.6 |    7.0     |  6.5  |  5.0  |  3.9  |    15.2     |
| Mamba    |  7.2† |  13.7†    | 14.4† | 12.4† | 10.4† |    17.2†    |
| **MS-TCN++** | **4.1** | **11.0** | **10.6** | **9.9** | **9.3** | **46.3** |

---

*For individual contributions and AI tool usage, see [`docs/REPORT.md`](docs/REPORT.md).*
