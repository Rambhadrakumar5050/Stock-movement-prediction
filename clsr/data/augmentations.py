"""
Data augmentation strategies for intra-day stock sequences, Section 4.5.

The paper lists four augmentations used to build the two "views" X_i, X_j
fed into the hybrid encoder for contrastive learning:
    - Smoothing (via FFT low-pass filtering)
    - Noise (Gaussian noise injection)
    - Cutoff (token cutoff: zero out interval-by-interval segments; and
      span cutoff: delete the last L segments of the sequence)
    - Dropout (randomly zero elements)

Input/output convention: a sequence tensor of shape (L, F) — L timesteps,
F features (open, high, low, close, volume). Augmentations operate on a
single sequence (batched by the caller via vmap or a python loop).
"""
from __future__ import annotations
import numpy as np
import torch


def smoothing_fft(x: torch.Tensor, keep_ratio: float = 0.9) -> torch.Tensor:
    """Low-pass filter each feature channel by zeroing out the top
    (1 - keep_ratio) fraction of high-frequency FFT components, then
    inverting back to the time domain. x: (L, F)."""
    L = x.shape[0]
    keep = max(1, int(L * keep_ratio))
    xf = torch.fft.rfft(x, dim=0)
    mask = torch.zeros_like(xf)
    mask[:keep] = 1.0
    xf = xf * mask
    out = torch.fft.irfft(xf, n=L, dim=0)
    return out.to(x.dtype)


def noise_injection(x: torch.Tensor, std_ratio: float = 0.02) -> torch.Tensor:
    """Add Gaussian noise scaled to each feature's own std so differently
    scaled columns (price vs. volume) are perturbed proportionally."""
    per_feat_std = x.std(dim=0, keepdim=True) + 1e-8
    noise = torch.randn_like(x) * per_feat_std * std_ratio
    return x + noise


def token_cutoff(x: torch.Tensor, ratio: float = 0.15) -> torch.Tensor:
    """Zero out a contiguous interval of timesteps of length ratio*L,
    starting at a random offset ("erase some tokens" — token cutoff)."""
    L = x.shape[0]
    cut_len = max(1, int(L * ratio))
    start = np.random.randint(0, max(1, L - cut_len + 1))
    out = x.clone()
    out[start:start + cut_len] = 0.0
    return out


def span_cutoff(x: torch.Tensor, ratio: float = 0.15) -> torch.Tensor:
    """Delete (zero) the trailing `ratio` fraction of the sequence
    ("deleting the latter L segments of the sequence")."""
    L = x.shape[0]
    cut_len = max(1, int(L * ratio))
    out = x.clone()
    out[L - cut_len:] = 0.0
    return out


def feature_dropout(x: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """Randomly zero individual elements (not whole timesteps) with
    probability p, matching standard dropout-as-augmentation."""
    mask = (torch.rand_like(x) > p).float()
    return x * mask


AUGMENTATION_REGISTRY = {
    "smoothing": smoothing_fft,
    "noise": noise_injection,
    "cutoff": token_cutoff,      # token cutoff variant used by default
    "span_cutoff": span_cutoff,
    "dropout": feature_dropout,
}


class RandomAugmentation:
    """Callable that draws one augmentation function uniformly at random
    from the registry each time it's applied — this is the `t ~ T` operator
    in Algorithm 1 ("draw two augmentation functions t ~ T, t' ~ T")."""

    def __init__(self, aug_cfg, names=None):
        self.cfg = aug_cfg
        self.names = names or ["smoothing", "noise", "cutoff", "dropout"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        name = np.random.choice(self.names)
        if name == "smoothing":
            return smoothing_fft(x, self.cfg.fft_keep_ratio)
        if name == "noise":
            return noise_injection(x, self.cfg.noise_std)
        if name == "cutoff":
            return token_cutoff(x, self.cfg.cutoff_ratio)
        if name == "span_cutoff":
            return span_cutoff(x, self.cfg.span_cutoff_len_ratio)
        if name == "dropout":
            return feature_dropout(x, self.cfg.dropout_p)
        raise ValueError(f"Unknown augmentation {name}")


def compose(x: torch.Tensor, names, cfg) -> torch.Tensor:
    """Apply a fixed sequence of augmentations in order — used by the
    ablation grid in Fig. 3 ("individual or composition of data
    augmentations")."""
    out = x
    for name in names:
        if name == "smoothing":
            out = smoothing_fft(out, cfg.fft_keep_ratio)
        elif name == "noise":
            out = noise_injection(out, cfg.noise_std)
        elif name == "cutoff":
            out = token_cutoff(out, cfg.cutoff_ratio)
        elif name == "span_cutoff":
            out = span_cutoff(out, cfg.span_cutoff_len_ratio)
        elif name == "dropout":
            out = feature_dropout(out, cfg.dropout_p)
    return out
