import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv(x))
        return x + self.drop(self.proj(out))


class _SingleStageTCN(nn.Module):
    def __init__(self, in_ch: int, hidden: int, num_classes: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.conv_in  = nn.Conv1d(in_ch, hidden, 1)
        self.layers   = nn.ModuleList([
            _DilatedResidualLayer(hidden, 2 ** i, dropout)
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
    MS-TCN++: Multi-Stage Temporal Convolutional Network.
    Input:  (B, T, feat_dim)
    Output: (B, T, num_classes)  — last stage logits

    After forward(), self.all_logits holds [(B, T, C), ...] for every stage.
    TemporalSegmentationModule uses all_logits to compute the multi-stage loss.
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
        # First stage reads raw features; refinement stages read softmax outputs
        self.stages = nn.ModuleList([
            _SingleStageTCN(feat_dim,     hidden, num_classes, n_layers, dropout),
            *[_SingleStageTCN(num_classes, hidden, num_classes, n_layers, dropout)
              for _ in range(n_stages - 1)],
        ])
        self.all_logits: list | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)            # (B, feat_dim, T)

        out  = self.stages[0](x)           # (B, num_classes, T)
        outs = [out]

        for stage in self.stages[1:]:
            out = stage(F.softmax(out, dim=1))
            outs.append(out)

        # Store per-stage logits as (B, T, C) for the loss in module.py
        self.all_logits = [o.permute(0, 2, 1) for o in outs]
        return self.all_logits[-1]
