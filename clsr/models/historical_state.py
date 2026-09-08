"""
Dynamic historical state, Section 4.1, Eqs. (6)-(9), Fig. 2.

    H_x = LSTM(X')                                  Eq. 6
    alpha_t = softmax_t( u^T tanh(W h_t + b) )       Eq. 8-9
    alpha = sum_t alpha_t * h_t                      Eq. 7  (context vector)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)   # W, b in Eq. 9
        self.u = nn.Linear(attn_dim, 1, bias=False)            # u^T in Eq. 9

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, T, hidden_dim) -> context vector (B, hidden_dim), Eq. 7."""
        scores = self.u(torch.tanh(self.W(H))).squeeze(-1)     # (B, T),  alpha~_t
        weights = torch.softmax(scores, dim=1)                  # Eq. 8
        context = torch.bmm(weights.unsqueeze(1), H).squeeze(1)  # Eq. 7
        return context


class HistoricalStateEncoder(nn.Module):
    """Feature-mapping layer -> LSTM layer -> temporal attention layer,
    exactly matching the three-layer stack drawn in Fig. 2."""

    def __init__(self, in_dim: int, lstm_hidden: int, attn_dim: int):
        super().__init__()
        self.feature_map = nn.Linear(in_dim, lstm_hidden)   # "Feature mapping layer"
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, batch_first=True)
        self.attn = TemporalAttention(lstm_hidden, attn_dim)

    def forward(self, x_prime: torch.Tensor) -> torch.Tensor:
        """x_prime: (B, m, F') inter-day series -> dynamic history state
        alpha of shape (B, lstm_hidden)."""
        h0 = self.feature_map(x_prime)
        H, _ = self.lstm(h0)
        alpha = self.attn(H)
        return alpha
