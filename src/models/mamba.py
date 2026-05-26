"""
Mamba (State Space Model) per temporal action segmentation.
Implementazione PyTorch pura — funziona su Windows senza mamba-ssm.
Basata su: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
           Selective State Spaces", 2023 (arXiv:2312.00752)

Input:  (B, T, feat_dim)
Output: (B, T, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model — il cuore di Mamba.
    
    A differenza di S4 (parametri fissi), i parametri B, C, Δ
    dipendono dall'input — questo è il "selective scan".
    
    Dimensioni:
        d_model: dimensione input/output
        d_state: dimensione stato interno (N)
        d_conv:  larghezza convoluzione locale
        expand:  fattore di espansione interno
    """

    def __init__(
        self,
        d_model: int,
        d_state: int  = 16,
        d_conv: int   = 4,
        expand: int   = 2,
        dt_rank: str  = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        self.d_inner  = int(expand * d_model)

        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_rank = dt_rank

        # Proiezione input → (z, x, B, C, Δ)
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convoluzione causale locale (depthwise)
        self.conv1d   = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            padding      = d_conv - 1,
            groups       = self.d_inner,
            bias         = True,
        )

        # Proiezioni per B, C, Δ (selettive — dipendono dall'input)
        self.x_proj   = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Parametri SSM fissi: A (log), D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))

        # Proiezione output
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Inizializzazione Δ bias
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, _ = x.shape

        # Proiezione ingresso
        xz  = self.in_proj(x)                          # (B, T, 2*d_inner)
        x_h, z = xz.chunk(2, dim=-1)                   # (B, T, d_inner) ciascuno

        # Convoluzione causale locale
        x_h = x_h.transpose(1, 2)                      # (B, d_inner, T)
        x_h = self.conv1d(x_h)[..., :T]                # (B, d_inner, T) — tronca padding
        x_h = x_h.transpose(1, 2)                      # (B, T, d_inner)
        x_h = F.silu(x_h)

        # Parametri selettivi (dipendono dall'input)
        x_dbl = self.x_proj(x_h)                       # (B, T, dt_rank + 2*d_state)
        dt, B_ssm, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt    = F.softplus(self.dt_proj(dt))            # (B, T, d_inner)
        A     = -torch.exp(self.A_log.float())          # (d_inner, d_state)

        # Selective scan — checkpointed per non tenere i tensori intermedi in memoria
        y = checkpoint(self._parallel_scan, x_h, dt, A, B_ssm, C, use_reentrant=False)

        # Gating + output
        y = y * F.silu(z)
        return self.out_proj(y)

    def _parallel_scan(
        self,
        u: torch.Tensor,    # (B, T, d_inner)
        dt: torch.Tensor,   # (B, T, d_inner)
        A: torch.Tensor,    # (d_inner, d_state)
        B: torch.Tensor,    # (B, T, d_state)
        C: torch.Tensor,    # (B, T, d_state)
    ) -> torch.Tensor:
        """
        Parallel prefix scan in O(log T) passi PyTorch invece di O(T) loop Python.

        La ricorrenza h_t = a_t * h_{t-1} + b_t è un monoide con operazione:
            (a_r, b_r) ⊕ (a_l, b_l) = (a_r * a_l,  a_r * b_l + b_r)
        Il prefix scan inclusivo si calcola in ceil(log2(T)) passi paralleli:
        ogni passo combina ogni elemento con quello 'stride' posizioni prima.

        Complessità: O(T log T) operazioni, O(log T) depth  (vs O(T) depth sequenziale)
        Memoria:     O(T * d_inner * d_state * log T)        (vs O(T) sequenziale)
        """
        B_sz, T, d_inner = u.shape
        d_state = A.shape[1]

        # Discretizzazione ZOH: dA = exp(Δ·A),  b_t = (Δ·B) ⊙ u_t
        dA = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, T, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)          # (B, T, d_inner, d_state)
        Bu = dB * u.unsqueeze(-1)                        # (B, T, d_inner, d_state)

        a, b = dA, Bu

        # Parallel inclusive prefix scan
        stride = 1
        while stride < T:
            # Shifta di 'stride' a destra; padding con identità (a=1, b=0)
            pad_a = torch.ones (B_sz, stride, d_inner, d_state, device=a.device, dtype=a.dtype)
            pad_b = torch.zeros(B_sz, stride, d_inner, d_state, device=b.device, dtype=b.dtype)
            a_prev = torch.cat([pad_a, a[:, :T - stride]], dim=1)  # (B, T, d_inner, d_state)
            b_prev = torch.cat([pad_b, b[:, :T - stride]], dim=1)  # (B, T, d_inner, d_state)

            # Usa 'a' originale per entrambe (b PRIMA di aggiornare a)
            b = a * b_prev + b
            a = a * a_prev

            stride *= 2

        # b[:, t] == h_t  (stato latente al passo t, con h_{-1} = 0)
        y = (b * C.unsqueeze(2)).sum(-1)               # (B, T, d_inner)
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)   # skip connection D
        return y


class MambaBlock(nn.Module):
    """Mamba block: SSM + residual + LayerNorm."""

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = SelectiveSSM(d_model, d_state=d_state,
                                  d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.ssm(self.norm(x)))


class MambaModel(nn.Module):
    """
    Stack di MambaBlock per segmentazione temporale.
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
        self.blocks     = nn.ModuleList([
            MambaBlock(hidden, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)

        total = sum(p.numel() for p in self.parameters())
        print(f"[Mamba] {n_layers} layer(s), hidden={hidden}, "
              f"d_state={d_state} -> {total:,} parametri")

    def forward(self, x):
        x = self.input_proj(x)   # (B, T, hidden)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)      # (B, T, num_classes)
