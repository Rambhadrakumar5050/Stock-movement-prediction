"""
Loss functions, Section 4.3 (Eq. 16, contrastive) and Section 4.4
(Eq. 17-19, supervised + joint).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss (Eq. 16 / Algorithm 1).

    z_i, z_j: (B, D) projected representations of the two augmented views
    of the same B sequences in the batch. Builds the 2B x 2B similarity
    matrix and treats each sample's paired augmentation as the positive,
    all other 2(B-1) in-batch samples as negatives -- exactly the SimCLR
    NT-Xent construction Algorithm 1 spells out.
    """
    B = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)                     # (2B, D)
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.t()) / temperature            # (2B, 2B), s_{i,j}

    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))                  # exclude self-similarity (1_[k!=i])

    # positive pairs: (k, k+B) and (k+B, k)
    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = (pos_idx + B) % (2 * B)

    loss = F.cross_entropy(sim, pos_idx)
    return loss


def supervised_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Eq. (17): cross-entropy for binary movement classification.
    labels in {0, 1}."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def joint_loss(logits, labels, z_i, z_j, temperature: float, alpha: float):
    """Eq. (19): L_joint = L_ce + alpha * L_con."""
    l_ce = supervised_bce_loss(logits, labels)
    l_con = nt_xent_loss(z_i, z_j, temperature)
    return l_ce + alpha * l_con, l_ce.detach(), l_con.detach()
