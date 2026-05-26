import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, channels: int, dropout: float):
        super().__init__()
        self.conv_dilated = nn.Conv1d(channels, channels, 3,
                                      padding=dilation, dilation=dilation)
        self.conv_1x1     = nn.Conv1d(channels, channels, 1)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class _PredictionGeneration(nn.Module):
    """
    Primo stage di MS-TCN2.
    Usa due stream di conv dilatate (una decrescente, una crescente)
    fusi ad ogni layer con una conv 1x1, poi residual connection.
    """
    def __init__(self, in_ch: int, hidden: int, num_classes: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.n_layers    = n_layers
        self.conv_in     = nn.Conv1d(in_ch, hidden, 1)

        # Stream 1: dilatazione decrescente  2^(N-1), ..., 2^0
        self.conv_dil_1 = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 3,
                      padding=2 ** (n_layers - 1 - i),
                      dilation=2 ** (n_layers - 1 - i))
            for i in range(n_layers)
        ])
        # Stream 2: dilatazione crescente    2^0, ..., 2^(N-1)
        self.conv_dil_2 = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 3,
                      padding=2 ** i,
                      dilation=2 ** i)
            for i in range(n_layers)
        ])
        # Fusione dei due stream
        self.conv_fusion = nn.ModuleList([
            nn.Conv1d(2 * hidden, hidden, 1)
            for _ in range(n_layers)
        ])

        self.dropout  = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(hidden, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv_in(x)
        for i in range(self.n_layers):
            f_in = f
            f = self.conv_fusion[i](
                torch.cat([self.conv_dil_1[i](f), self.conv_dil_2[i](f)], dim=1)
            )
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        return self.conv_out(f)


class _Refinement(nn.Module):
    """Stage di raffinamento (stage 2, 3, 4 di MS-TCN2)."""
    def __init__(self, in_ch: int, hidden: int, num_classes: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.conv_in  = nn.Conv1d(in_ch, hidden, 1)
        self.layers   = nn.ModuleList([
            copy.deepcopy(_DilatedResidualLayer(2 ** i, hidden, dropout))
            for i in range(n_layers)
        ])
        self.conv_out = nn.Conv1d(hidden, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class MSTCNModel(nn.Module):
    """
    MS-TCN2: Multi-Stage Temporal Convolutional Network.
    Input:  (B, T, feat_dim)
    Output: (B, T, num_classes)  — ultimo stage

    self.all_logits: lista [(B, T, C), ...] per ogni stage,
    usata dal training module per la multi-stage loss.
    """

    def __init__(
        self,
        feat_dim:    int   = 1024,
        num_classes: int   = 106,
        hidden:      int   = 128,
        n_stages:    int   = 4,
        n_layers:    int   = 10,
        dropout:     float = 0.5,
    ):
        super().__init__()
        self.pg = _PredictionGeneration(feat_dim, hidden, num_classes,
                                        n_layers, dropout)
        self.refinements = nn.ModuleList([
            _Refinement(num_classes, hidden, num_classes, n_layers, dropout)
            for _ in range(n_stages - 1)
        ])
        self.all_logits: list | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)               # (B, feat_dim, T)

        out  = self.pg(x)                     # (B, num_classes, T)
        outs = [out]

        for refinement in self.refinements:
            out = refinement(F.softmax(out, dim=1))
            outs.append(out)

        self.all_logits = [o.permute(0, 2, 1) for o in outs]  
        return self.all_logits[-1]
