"""
Full CLSR model: hybrid encoder (Section 4.2) + historical state (4.1)
+ contrastive head (4.3) + supervised head (4.4).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from models.global_net import GlobalInformationNet
from models.local_net import LocalInformationNet
from models.historical_state import HistoricalStateEncoder


class HybridEncoder(nn.Module):
    """Runs the (augmented, S1-S3-concatenated) intraday series through
    both the global (Transformer/ProbSparse) and local (TCN) branches in
    parallel, then concatenates ("splices") their outputs, Section 4.2:
    "the two features are spliced to obtain the encoder's outputs Z_i and
    Z_j"."""

    def __init__(self, in_dim: int, d_model: int, n_heads: int, n_layers: int,
                 tcn_channels, tcn_kernel: int, dropout: float, max_len: int):
        super().__init__()
        self.global_net = GlobalInformationNet(
            in_dim, d_model, n_heads, n_layers, dropout, max_len)
        self.local_net = LocalInformationNet(
            in_dim, tcn_channels, tcn_kernel, dropout)
        self.out_dim = d_model + tcn_channels[-1]

    def forward(self, x):
        g = self.global_net(x)   # (B, d_model)
        l = self.local_net(x)    # (B, tcn_channels[-1])
        return torch.cat([g, l], dim=-1)  # (B, out_dim)


class ProjectionHead(nn.Module):
    """Non-linear projection used before the contrastive loss, following
    SimCLR practice (paper explicitly cites Chen et al. [24] SimCLR's use
    of "a nonlinear transformation ... to simplify the contrastive learning
    algorithm")."""

    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class CLSRModel(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        in_dim = mcfg.feature_dim + mcfg.static_hist_dim
        self.encoder = HybridEncoder(
            in_dim=in_dim,
            d_model=mcfg.d_model,
            n_heads=mcfg.n_heads,
            n_layers=mcfg.n_encoder_layers,
            tcn_channels=mcfg.tcn_channels,
            tcn_kernel=mcfg.tcn_kernel_size,
            dropout=mcfg.dropout,
            max_len=512,
        )
        self.proj_head = ProjectionHead(self.encoder.out_dim, mcfg.proj_dim)

        # inter-day feature dim is 1 in our dataset (close-only); expose as
        # a constructor arg so users with a richer inter-day feature set can change it.
        self.hist_encoder = HistoricalStateEncoder(
            in_dim=1, lstm_hidden=mcfg.lstm_hidden, attn_dim=mcfg.attn_hidden)

        fusion_dim = self.encoder.out_dim + mcfg.lstm_hidden
        self.classifier = nn.Linear(fusion_dim, 1)   # Eq. 18: y_hat = sigmoid(Wf + b)

    def encode(self, x_aug):
        """x_aug: (B, L, F+3) -> (raw_repr, projected_repr) for one view."""
        z = self.encoder(x_aug)
        p = self.proj_head(z)
        return z, p

    def forward(self, x_i, x_j, inter_day):
        """Full forward pass for one training step: two augmented views,
        plus the inter-day series for historical state.
        Returns: proj_i, proj_j (for contrastive loss), logits (for BCE),
        using the FIRST view's raw representation concatenated with the
        historical state per Eq. (18)-(19)'s f = concat[Z, alpha]. (Paper
        text says "Z (Z_o or Z_a)" — either augmented view's encoder output
        may be used for the classifier; we use view i.)
        """
        z_i, p_i = self.encode(x_i)
        z_j, p_j = self.encode(x_j)

        alpha = self.hist_encoder(inter_day)          # (B, lstm_hidden)
        f = torch.cat([z_i, alpha], dim=-1)             # Eq. 19's f
        logits = self.classifier(f).squeeze(-1)

        return p_i, p_j, logits

    @torch.no_grad()
    def predict(self, x, inter_day):
        """Inference: single (unaugmented) view."""
        z, _ = self.encode(x)
        alpha = self.hist_encoder(inter_day)
        f = torch.cat([z, alpha], dim=-1)
        logits = self.classifier(f).squeeze(-1)
        return torch.sigmoid(logits)
