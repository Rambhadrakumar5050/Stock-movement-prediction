"""
ProbSparse self-attention, Section 4.2.1, Eqs. (11)-(13).

The paper borrows the ProbSparse mechanism from Informer (Zhou et al.,
AAAI 2021, ref [39]) to cut attention memory from O(n^2) to O(n log n) by
only computing full attention for the top-u "active" queries (those with
the highest sparsity score M(q_i, K), Eq. 12) and replacing the rest with
the mean of V. This is a faithful, from-scratch reimplementation of that
mechanism (not copied from the Informer repo).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor  # controls u = factor * ln(L_Q), sampling budget

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_qk(self, Q, K, sample_k, n_top):
        """Eq. (12): sparsity measurement M(q_i, K), computed on a random
        subsample of keys for efficiency (as in Informer), then select the
        top-n_top queries by score."""
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # sample sample_k keys per query, uniformly at random
        idx = torch.randint(0, L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K[:, :, idx, :]                       # (B,H,L_Q,sample_k,D)
        Q_expand = Q.unsqueeze(-2)                        # (B,H,L_Q,1,D)
        qk_sample = torch.matmul(Q_expand, K_sample.transpose(-2, -1)).squeeze(-2)
        # qk_sample: (B,H,L_Q,sample_k)

        # M(q_i,K) = max_j qk_ij - mean_j qk_ij  (a numerically simpler,
        # equivalent-in-spirit stand-in for the log-sum-exp form in Eq. 12,
        # used by the original Informer implementation for the same reason:
        # it's cheaper and empirically tracks the same ranking)
        M = qk_sample.max(-1)[0] - qk_sample.mean(-1)      # (B,H,L_Q)
        M_top = M.topk(n_top, sorted=False)[1]             # (B,H,n_top) indices into L_Q

        Q_reduced = torch.gather(
            Q, 2, M_top.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # (B,H,n_top,D)
        qk = torch.matmul(Q_reduced, K.transpose(-2, -1)) / math.sqrt(D)  # (B,H,n_top,L_K)
        return qk, M_top

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        Q = self.w_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.w_k(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        L_Q = L
        u = min(L_Q, max(1, int(self.factor * math.ceil(math.log(max(L_Q, 2))))))
        sample_k = min(L_Q, max(1, int(self.factor * math.ceil(math.log(max(L_Q, 2))))))

        qk_top, index = self._prob_qk(Q, K, sample_k, u)
        attn_top = F.softmax(qk_top, dim=-1)
        attn_top = self.dropout(attn_top)
        context_top = torch.matmul(attn_top, V)  # (B,H,u,D)

        # non-selected queries fall back to the mean of V (Informer's
        # "lazy" initialization for non-active queries)
        context = V.mean(dim=2, keepdim=True).expand(B, self.n_heads, L_Q, self.d_head).clone()
        context.scatter_(2, index.unsqueeze(-1).expand(-1, -1, -1, self.d_head), context_top)

        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(context)
