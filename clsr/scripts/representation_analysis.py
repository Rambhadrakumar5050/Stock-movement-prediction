"""
Reproduces Section 5.6 / Figs 4-9: compares (a) raw intraday data,
(b) Transformer-only encoder features, (c) CLSR hybrid-encoder features
for a randomly chosen stock's 72 days of data, via:
    - KMeans (k=2) clustering + PCA-reduced scatter plot
    - cosine-similarity correlation heatmap across the 72 days
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLSRConfig
from models.clsr import CLSRModel
from scripts.ablation import TransformerOnlyEncoder
from scripts.train import build_dataloaders, set_seed


@torch.no_grad()
def get_representations(model, intraday_batch, static_hist_batch, device):
    B, L, F_ = intraday_batch.shape
    static_exp = static_hist_batch.unsqueeze(1).expand(B, L, static_hist_batch.shape[-1])
    x_cat = torch.cat([intraday_batch, static_exp], dim=-1).to(device)
    z, _ = model.encode(x_cat)
    return z.cpu().numpy()


def plot_cluster_and_corr(data_2d_list, titles, out_prefix):
    for data, title in zip(data_2d_list, titles):
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data)
        pca = PCA(n_components=2).fit_transform(data)

        fig, ax = plt.subplots(figsize=(5, 5))
        for c in [0, 1]:
            mask = km.labels_ == c
            ax.scatter(pca[mask, 0], pca[mask, 1], label=str(c), s=15)
        ax.set_title(f"KMeans Clusters (2) Derived from {title}")
        ax.legend()
        fig.tight_layout()
        fname = f"{out_prefix}_cluster_{title.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"saved {fname}")

        sim = data @ data.T
        norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-8
        sim = sim / (norms @ norms.T)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        im = ax2.imshow(sim, cmap="viridis")
        ax2.set_title(f"cosine similarity from {title}")
        fig2.colorbar(im)
        fig2.tight_layout()
        fname2 = f"{out_prefix}_cosine_{title.replace(' ', '_')}.png"
        fig2.savefig(fname2, dpi=120)
        plt.close(fig2)
        print(f"saved {fname2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clsr_ckpt", default="./checkpoints/best.pt")
    parser.add_argument("--transformer_ckpt", default="./checkpoints_ablation/CLSR_T_best.pt")
    parser.add_argument("--backend", default="synthetic")
    parser.add_argument("--n_stocks", type=int, default=5)
    parser.add_argument("--n_days", type=int, default=72)
    parser.add_argument("--out_prefix", default="./repr_analysis")
    args = parser.parse_args()

    cfg = CLSRConfig()
    cfg.data.backend = args.backend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.train.seed)

    _, _, test_dl = build_dataloaders(cfg, args.n_stocks)
    # gather n_days consecutive samples from one stock (test_dl is already
    # grouped/sequential per-stock within CLSRDataset's construction order)
    intraday_all, static_all = [], []
    for intraday, static_hist, inter_day, labels in test_dl:
        intraday_all.append(intraday)
        static_all.append(static_hist)
        if sum(x.shape[0] for x in intraday_all) >= args.n_days:
            break
    intraday_batch = torch.cat(intraday_all, dim=0)[:args.n_days]
    static_batch = torch.cat(static_all, dim=0)[:args.n_days]

    raw_flat = intraday_batch.reshape(args.n_days, -1).numpy()

    clsr_model = CLSRModel(cfg.model).to(device)
    if os.path.exists(args.clsr_ckpt):
        clsr_model.load_state_dict(torch.load(args.clsr_ckpt, map_location=device))
    clsr_repr = get_representations(clsr_model, intraday_batch, static_batch, device)

    tf_model = CLSRModel(cfg.model)
    in_dim = cfg.model.feature_dim + cfg.model.static_hist_dim
    tf_model.encoder = TransformerOnlyEncoder(
        in_dim, cfg.model.d_model, cfg.model.n_heads,
        cfg.model.n_encoder_layers, cfg.model.dropout, 512)
    tf_model = tf_model.to(device)
    if os.path.exists(args.transformer_ckpt):
        tf_model.load_state_dict(torch.load(args.transformer_ckpt, map_location=device))
    tf_repr = get_representations(tf_model, intraday_batch, static_batch, device)

    plot_cluster_and_corr(
        [raw_flat, tf_repr, clsr_repr],
        ["original data", "Transformer", "CLSR"],
        args.out_prefix,
    )


if __name__ == "__main__":
    main()
