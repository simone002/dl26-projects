"""
Mamba (State Space Model) per temporal action segmentation.
Usa mamba-ssm nativo con kernel CUDA ottimizzati.

Input:  (B, T, feat_dim)
Output: (B, T, num_classes)
"""

import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = Mamba(d_model=d_model, d_state=d_state,
                          d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.ssm(self.norm(x)))


class MambaModel(nn.Module):
    """
    Stack di blocchi Mamba per segmentazione temporale.
    Input:  (B, T, feat_dim)
    Output: (B, T, num_classes)
    """

    def __init__(
        self,
        feat_dim: int    = 1024,
        num_classes: int = 106,
        hidden: int      = 256,
        n_layers: int    = 2,
        d_state: int     = 16,
        d_conv: int      = 4,
        expand: int      = 2,
        dropout: float   = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(feat_dim, hidden)
        self.blocks = nn.ModuleList([
            MambaBlock(hidden, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)

        total = sum(p.numel() for p in self.parameters())
        print(f"[Mamba] {n_layers} layer(s), hidden={hidden}, "
              f"d_state={d_state} → {total:,} parametri")

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
