"""
LightningModule per temporal action segmentation.
Loss: CE + 0.5*SmoothLoss + 0.5*BoundaryLoss
Metriche: mIoU, F1@{10,25,50}, Edit Score, Boundary F1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import defaultdict


# ── Smooth Loss ───────────────────────────────────────────────────────────────

def smooth_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    MS-TCN style smooth loss: MSE clamped su log-probabilità adiacenti,
    solo dove il GT non cambia etichetta.
    logits:  (B, T, C)
    targets: (B, T)
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    diff      = log_probs[:, 1:, :] - log_probs[:, :-1, :]
    mask      = (targets[:, 1:] == targets[:, :-1]).float()
    smooth    = torch.clamp(diff ** 2, max=16).mean(dim=-1)  # mean su C, non sum
    return (smooth * mask).mean()


# ── Boundary Loss ─────────────────────────────────────────────────────────────

def boundary_loss(logits: torch.Tensor, targets: torch.Tensor,
                  k: int = 3) -> torch.Tensor:
    """
    Penalizza gli errori nei frame vicini ai confini delle azioni.
    Espande la maschera di +-k frame per maggiore robustezza.
    logits:  (B, T, C)
    targets: (B, T)
    k:       ampiezza finestra di espansione
    """
    B, T, C = logits.shape

    # Confini base
    change = (targets[:, 1:] != targets[:, :-1])
    boundary_mask = torch.zeros(B, T, dtype=torch.bool, device=targets.device)
    boundary_mask[:, 1:]  |= change
    boundary_mask[:, :-1] |= change

    # Espansione finestra +-k frame
    for shift in range(1, k + 1):
        if shift < T:
            boundary_mask[:, shift:]  |= boundary_mask[:, :-shift].clone()
            boundary_mask[:, :-shift] |= boundary_mask[:, shift:].clone()

    if boundary_mask.sum().item() == 0:
        return torch.tensor(0.0, device=logits.device)

    logits_flat  = logits[boundary_mask]
    targets_flat = targets[boundary_mask]
    return F.cross_entropy(logits_flat, targets_flat)


# ── Metriche frame-level ──────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor,
                    ignore_index: int = 0):
    acc  = (preds == targets).float().mean()
    mask = targets != ignore_index
    if mask.sum().item() == 0:
        return acc, torch.tensor(0.0, device=preds.device)
    acc_fg = (preds[mask] == targets[mask]).float().mean()
    return acc, acc_fg


# ── Utilita' segmenti ─────────────────────────────────────────────────────────

def get_segments(seq, bg_class: int = 0) -> list:
    segments, i = [], 0
    seq = seq.tolist() if hasattr(seq, 'tolist') else seq
    while i < len(seq):
        label, j = seq[i], i
        while j < len(seq) and seq[j] == label:
            j += 1
        if label != bg_class:
            segments.append((label, i, j - 1))
        i = j
    return segments


# ── F1@k ─────────────────────────────────────────────────────────────────────

def f1_at_k(pred_seq, target_seq, overlap_thresh, bg_class=0):
    pred_segs   = get_segments(pred_seq, bg_class)
    target_segs = get_segments(target_seq, bg_class)
    tp, used    = 0, [False] * len(target_segs)

    for p_label, p_start, p_end in pred_segs:
        best_iou, best_idx = 0, -1
        for i, (t_label, t_start, t_end) in enumerate(target_segs):
            if used[i] or t_label != p_label:
                continue
            inter = max(0, min(p_end, t_end) - max(p_start, t_start) + 1)
            union = max(p_end, t_end) - min(p_start, t_start) + 1
            iou   = inter / union
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= overlap_thresh and best_idx >= 0:
            tp += 1
            used[best_idx] = True

    fp = len(pred_segs) - tp
    fn = len(target_segs) - tp
    pr = tp / (tp + fp + 1e-6)
    re = tp / (tp + fn + 1e-6)
    return 2 * pr * re / (pr + re + 1e-6)


# ── Edit Score ────────────────────────────────────────────────────────────────

def edit_score(pred_seq, target_seq, bg_class=0):
    """
    Edit Score basato sulla distanza di Levenshtein
    tra le sequenze di segmenti (non frame).
    Score = 1 - edit_dist / max(|pred|, |gt|)
    Range [0,1], 1 = perfetto.
    """
    pred_labels   = [s[0] for s in get_segments(pred_seq,   bg_class)]
    target_labels = [s[0] for s in get_segments(target_seq, bg_class)]

    n, m = len(pred_labels), len(target_labels)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    # Levenshtein con programmazione dinamica
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if pred_labels[i-1] == target_labels[j-1]:
                new_dp[j] = dp[j-1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j-1], dp[j-1])
        dp = new_dp

    return 1.0 - dp[m] / max(n, m)


# ── Boundary F1 ───────────────────────────────────────────────────────────────

def boundary_f1(pred_seq, target_seq, tolerance=2):
    """
    F1 sui confini temporali delle azioni.
    Un confine predetto e' TP se e' entro 'tolerance' frame
    da un confine reale.
    """
    def get_boundaries(seq):
        seq = seq.tolist() if hasattr(seq, 'tolist') else seq
        return [t for t in range(1, len(seq)) if seq[t] != seq[t-1]]

    pred_bounds   = get_boundaries(pred_seq)
    target_bounds = get_boundaries(target_seq)

    if not pred_bounds and not target_bounds:
        return 1.0
    if not pred_bounds or not target_bounds:
        return 0.0

    used = [False] * len(target_bounds)
    tp   = 0
    for pb in pred_bounds:
        for i, tb in enumerate(target_bounds):
            if not used[i] and abs(pb - tb) <= tolerance:
                tp += 1
                used[i] = True
                break

    fp = len(pred_bounds) - tp
    fn = len(target_bounds) - tp
    pr = tp / (tp + fp + 1e-6)
    re = tp / (tp + fn + 1e-6)
    return 2 * pr * re / (pr + re + 1e-6)


# ── LightningModule ───────────────────────────────────────────────────────────

class TemporalSegmentationModule(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        num_classes: int       = 106,
        lr: float              = 1e-3,
        weight_decay: float    = 1e-4,
        label_smoothing: float = 0.05,
        bg_weight: float       = 0.05,
        smooth_weight: float   = 0.2,
        boundary_weight: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model           = model
        self.smooth_weight   = smooth_weight
        self.boundary_weight = boundary_weight

        weights    = torch.ones(num_classes)
        weights[0] = bg_weight
        self.criterion = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=label_smoothing
        )

        self._tp: dict = defaultdict(float)
        self._fp: dict = defaultdict(float)
        self._fn: dict = defaultdict(float)

        self._seg_preds:   list = []
        self._seg_targets: list = []
        self.test_prefix: str = "test"

        self._train_tp: dict = defaultdict(float)
        self._train_fp: dict = defaultdict(float)
        self._train_fn: dict = defaultdict(float)
        self._train_seg_preds:   list = []
        self._train_seg_targets: list = []

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y    = batch
        logits  = self(x)
        B, T, C = logits.shape
        N       = B * T

        targets_flat = y.reshape(N)

        # Multi-stage loss: average CE/smooth/boundary over all stages (MS-TCN++)
        # Single-stage models don't expose all_logits → falls back to [logits]
        stage_logits = (
            self.model.all_logits
            if hasattr(self.model, "all_logits") and self.model.all_logits is not None
            else [logits]
        )
        n_stages = len(stage_logits)
        ce   = sum(self.criterion(lg.reshape(N, C), targets_flat) for lg in stage_logits) / n_stages
        sl   = sum(smooth_loss(lg, y)               for lg in stage_logits) / n_stages
        bl   = sum(boundary_loss(lg, y)             for lg in stage_logits) / n_stages
        loss = ce + self.smooth_weight * sl + self.boundary_weight * bl

        # ── Metriche frame-level (solo ultimo stage) ──────────────────────────
        logits_flat = logits.reshape(N, C)
        preds = logits_flat.argmax(dim=-1)
        acc, acc_fg = compute_metrics(preds, targets_flat)

        on_step = (stage == "train")
        if stage != "test":
            self.log(f"{stage}/loss",    loss, prog_bar=True,  on_epoch=True, on_step=on_step)
            self.log(f"{stage}/loss_ce", ce,   prog_bar=False, on_epoch=True, on_step=False)
        if stage == "train":
            self.log("train/loss_smooth",   sl, prog_bar=False, on_epoch=True, on_step=False)
            self.log("train/loss_boundary", bl, prog_bar=False, on_epoch=True, on_step=False)
        if stage in ("train", "val"):
            self.log(f"{stage}/acc_fg", acc_fg, prog_bar=(stage == "val"),
                     on_epoch=True, on_step=False)

        if stage == "train":
            mask = targets_flat != 0
            if mask.sum().item() > 0:
                p_fg = preds[mask]
                t_fg = targets_flat[mask]
                for c in range(1, self.hparams.num_classes):
                    pred_c = (p_fg == c)
                    tgt_c  = (t_fg == c)
                    self._train_tp[c] += (pred_c & tgt_c).sum().item()
                    self._train_fp[c] += (pred_c & ~tgt_c).sum().item()
                    self._train_fn[c] += (~pred_c & tgt_c).sum().item()

            pred_seq = logits.argmax(dim=-1)
            for b in range(B):
                self._train_seg_preds.append(pred_seq[b].detach().cpu())
                self._train_seg_targets.append(y[b].detach().cpu())

        # ── Accumula per metriche epoch-level (val/test) ─────────────────────
        if stage in ("val", "test"):
            mask = targets_flat != 0
            if mask.sum().item() > 0:
                p_fg = preds[mask]
                t_fg = targets_flat[mask]
                for c in range(1, self.hparams.num_classes):
                    pred_c = (p_fg == c)
                    tgt_c  = (t_fg == c)
                    self._tp[c] += (pred_c & tgt_c).sum().item()
                    self._fp[c] += (pred_c & ~tgt_c).sum().item()
                    self._fn[c] += (~pred_c & tgt_c).sum().item()

            for b in range(B):
                self._seg_preds.append(logits[b].argmax(-1).cpu())
                self._seg_targets.append(y[b].cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        ious = []
        for c in range(1, self.hparams.num_classes):
            tp = self._train_tp[c]; fp = self._train_fp[c]; fn = self._train_fn[c]
            denom = tp + fp + fn
            if denom > 0:
                ious.append(tp / denom)
        if ious:
            self.log("train/mIoU_epoch", sum(ious) / len(ious), prog_bar=False)
        self._train_tp.clear(); self._train_fp.clear(); self._train_fn.clear()

        if self._train_seg_preds:
            edit_scores = [edit_score(p, t)
                           for p, t in zip(self._train_seg_preds, self._train_seg_targets)]
            self.log("train/edit_score", sum(edit_scores) / len(edit_scores), prog_bar=False)

        self._train_seg_preds.clear()
        self._train_seg_targets.clear()

    def _log_epoch_metrics(self, prefix: str):
        # mIoU epoch-level
        ious = []
        for c in range(1, self.hparams.num_classes):
            tp = self._tp[c]; fp = self._fp[c]; fn = self._fn[c]
            denom = tp + fp + fn
            if denom > 0:
                ious.append(tp / denom)
        if ious:
            self.log(f"{prefix}/mIoU_epoch", sum(ious)/len(ious), prog_bar=True)
        self._tp.clear(); self._fp.clear(); self._fn.clear()

        if self._seg_preds:
            # F1@{10, 25, 50}
            for k in [0.10, 0.25, 0.50]:
                scores = [f1_at_k(p, t, k)
                          for p, t in zip(self._seg_preds, self._seg_targets)]
                self.log(f"{prefix}/F1@{int(k*100)}", sum(scores)/len(scores))

            # Edit Score
            edit_scores = [edit_score(p, t)
                           for p, t in zip(self._seg_preds, self._seg_targets)]
            self.log(f"{prefix}/edit_score", sum(edit_scores)/len(edit_scores),
                     prog_bar=True)

            # Boundary F1 (tolerance = 2 frame ~ 0.08s a 24fps)
            bf1_scores = [boundary_f1(p, t, tolerance=2)
                          for p, t in zip(self._seg_preds, self._seg_targets)]
            self.log(f"{prefix}/boundary_f1", sum(bf1_scores)/len(bf1_scores),
                     prog_bar=False)

        self._seg_preds.clear()
        self._seg_targets.clear()

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics(self.test_prefix)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        t_max = self.trainer.max_epochs if self.trainer is not None else 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=1e-6
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler}}
