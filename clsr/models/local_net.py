"""
Local Information Network, Section 4.2.2, Eq. (15).

"TCN sequentially extracts the information in the sequence by using causal
convolution and dilated convolution ... filter kernel is 2, the dilation
stride is j and the number of layers is l." Standard TCN residual block
design (Bai, Kolter & Koltun 2018, ref [23]), reimplemented from scratch.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List


class Chomp1d(nn.Module):
    """Removes the extra right-padding introduced by causal (left-only
    logical) padding, so output length == input length."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                                padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                                padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class LocalInformationNet(nn.Module):
    def __init__(self, in_dim: int, channels: List[int], kernel_size: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_ch = in_dim
        for i, ch in enumerate(channels):
            dilation = 2 ** i   # "gradually expanding the receptive field"
            layers.append(TemporalBlock(prev_ch, ch, kernel_size, dilation, dropout))
            prev_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.out_proj = nn.Conv1d(prev_ch, prev_ch, kernel_size=1)

    def forward(self, x):
        """x: (B, L, in_dim) -> pooled representation (B, C)."""
        h = x.transpose(1, 2)          # (B, in_dim, L)
        h = self.tcn(h)                # (B, C, L)
        h = self.out_proj(h)
        return h.mean(dim=-1)          # (B, C) — "last output contains all input info"
