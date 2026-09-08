"""
Global Information Network, Section 4.2.1.

"a Transformer variant (Global Information Network) containing only the
encoder structure ... ProbSparse Self-attention mechanism is used ...
error in computing attention is reduced by increasing the number of
attention heads." Followed by a residual connection, a 1D conv over the
time dimension (Eq. 14), and a 1x1 spatial conv to unify output dims.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from models.attention import ProbSparseSelfAttention


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Eq. 10: `PositionEncoding(...)`)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GlobalEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # Eq. (14): 1D conv, kernel 3, stride 1, over the time dimension
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # residual connection around attention ("residual connections are
        # used between each sub-layer")
        a = self.attn(x)
        x = self.norm1(x + self.dropout(a))

        c = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + self.dropout(c))
        return x


class GlobalInformationNet(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_heads: int, n_layers: int,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)   # W_h^(I) in Eq. (10)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            GlobalEncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        # final 1x1 spatial conv to keep output dims uniform across branches
        self.out_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        """x: (B, L, in_dim) already augmented + concatenated with S1,S2,S3."""
        h = self.input_proj(x)
        h = self.pos_enc(h)
        for layer in self.layers:
            h = layer(h)
        h = self.out_proj(h.transpose(1, 2)).transpose(1, 2)
        # global pooling to a single vector representation per sequence
        return h.mean(dim=1)  # (B, d_model)
