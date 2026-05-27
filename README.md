# Temporal Action Segmentation from Video

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information
- **Group ID**: G36
- **Project ID**: 13

## 📝 Project Description

Dense frame-level action segmentation on the EGTEA Gaze+ egocentric dataset. Given pre-extracted DINOv3 ViT-B features (768-d per frame), the model predicts one of 106 action classes for every frame in the sequence. Four architectures are compared — CNN1D, LSTM, xLSTM, MS-TCN++ — all sharing the same training pipeline with a combined Cross-Entropy + Smooth + Boundary loss.

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

Features are extracted from the raw EGTEA videos with DINOv3 ViT-B and stored as one `.npy` file per video session:

```bash
python -m src.utils.extract_dinov3_features --videos_dir D:/egtea/videos --output_dir D:/egtea/dinov3_features
```

The output path can be changed in `experiments/configs/base.yaml` (`features_dir`, default `D:/egtea/dinov3_features`).  
Split files and action labels are already included in `data/annotations/`.

### 3. Training

```bash
# Train all 5 models, 3-fold cross-validation each
python train.py

# Train a single model (3 folds): cnn1d | lstm | xlstm | mstcn
python train.py --model mstcn
```

Override any nested hyperparameter without editing files:

```bash
python train.py --model mstcn training.lr=0.0002 model.hidden=256
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

Fold 1 (train split 1+2, test split 3), DINOv3 ViT-B features. All values in %.

| Model    | Acc  | Edit Score | F1@10 | F1@25 | F1@50 | Boundary F1 |
|----------|:----:|:----------:|:-----:|:-----:|:-----:|:-----------:|
| CNN1D    | 92.7 |    73.5    | 74.7  | 74.3  | 72.3  |    57.0     |
| LSTM     | 96.6 |    94.8    | 88.9  | 88.7  | 88.2  |    77.9     |
| xLSTM    | 98.0 |    91.4    | 86.7  | 86.6  | 86.3  |    85.3     |
| **MS-TCN++** | 96.2 | **94.5** | 88.6  | 88.4  | 87.8  |    77.4     |

---

*For individual contributions and AI tool usage, see [`docs/REPORT.md`](docs/REPORT.md).*
