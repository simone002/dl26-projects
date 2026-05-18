<<<<<<< HEAD
"""
xLSTM per temporal action segmentation.
Usa la libreria ufficiale degli autori del paper (NX-AI/xlstm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class xLSTMModel(nn.Module):
    def __init__(self, feat_dim=1024, num_classes=106, hidden=256,
                 n_layers=2, dropout=0.3):
        super().__init__()

        from xlstm import (
            xLSTMBlockStack, xLSTMBlockStackConfig,
            mLSTMBlockConfig, mLSTMLayerConfig,
        )
        cfg = xLSTMBlockStackConfig(
            mlstm_block   = mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                )
            ),
            context_length = 512,
            num_blocks     = n_layers,
            embedding_dim  = hidden,
            dropout        = dropout,
            slstm_at       = [],
        )
        self.input_proj    = nn.Linear(feat_dim, hidden)
        self.xlstm         = xLSTMBlockStack(cfg)
        self.head          = nn.Linear(hidden, num_classes)
        self._use_official = True
        print("[xLSTM] OK usando libreria ufficiale NX-AI/xlstm")


    def get_features(self, x):
        """Ritorna feature prima della head. (B, T, hidden)"""
        x = self.input_proj(x)
        return self.xlstm(x)

    def forward(self, x):
        return self.head(self.get_features(x))


=======
"""
xLSTM per temporal action segmentation.
Usa la libreria ufficiale degli autori del paper (NX-AI/xlstm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class xLSTMModel(nn.Module):
    def __init__(self, feat_dim=1024, num_classes=106, hidden=256,
                 n_layers=2, dropout=0.3):
        super().__init__()

        from xlstm import (
            xLSTMBlockStack, xLSTMBlockStackConfig,
            mLSTMBlockConfig, mLSTMLayerConfig,
        )
        cfg = xLSTMBlockStackConfig(
            mlstm_block   = mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                )
            ),
            context_length = 512,
            num_blocks     = n_layers,
            embedding_dim  = hidden,
            dropout        = dropout,
            slstm_at       = [],
        )
        self.input_proj    = nn.Linear(feat_dim, hidden)
        self.xlstm         = xLSTMBlockStack(cfg)
        self.head          = nn.Linear(hidden, num_classes)
        self._use_official = True
        print("[xLSTM] OK usando libreria ufficiale NX-AI/xlstm")


    def get_features(self, x):
        """Ritorna feature prima della head. (B, T, hidden)"""
        x = self.input_proj(x)
        return self.xlstm(x)

    def forward(self, x):
        return self.head(self.get_features(x))


>>>>>>> 42c5e63ee0570321b67744f8b5f40cc8e0fffff0
