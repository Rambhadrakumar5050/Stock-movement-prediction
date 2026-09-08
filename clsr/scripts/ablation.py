"""
Reproduces the Table 3 ablation study:
    CLSR_T : hybrid encoder replaced by plain Transformer (no TCN branch)
    CLSR_d : no dynamic historical state (drop the LSTM+Attention branch)
    CLSR_s : no static historical state (drop S1,S2,S3 concatenation, Eq. 3-5)
    CLSR   : full model (baseline for comparison)

Each variant reuses the same training loop as scripts/train.py; only the
model construction / input construction differs, controlled by --variant.
"""
from __future__ import annotations
import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLSRConfig
from models.clsr import CLSRModel, HybridEncoder
from models.global_net import GlobalInformationNet
from scripts.train import (build_dataloaders, evaluate, set_seed,
                            get_lr_lambda, RandomAugmentation)
from losses import joint_loss, nt_xent_loss, supervised_bce_loss


class TransformerOnlyEncoder(HybridEncoder):
    """CLSR_T: replace the hybrid (Transformer+TCN) encoder with a
    Transformer-only global net, matching Table 3's CLSR_T row."""

    def __init__(self, in_dim, d_model, n_heads, n_layers, dropout, max_len):
        nn.Module.__init__(self)
        self.global_net = GlobalInformationNet(in_dim, d_model, n_heads, n_layers, dropout, max_len)
        self.local_net = None
        self.out_dim = d_model

    def forward(self, x):
        return self.global_net(x)


def build_variant_model(cfg, variant: str):
    model = CLSRModel(cfg.model)
    if variant == "CLSR_T":
        in_dim = cfg.model.feature_dim + cfg.model.static_hist_dim
        model.encoder = TransformerOnlyEncoder(
            in_dim, cfg.model.d_model, cfg.model.n_heads,
            cfg.model.n_encoder_layers, cfg.model.dropout, 512)
        model.proj_head = model.proj_head.__class__(model.encoder.out_dim, cfg.model.proj_dim)
        fusion_dim = model.encoder.out_dim + cfg.model.lstm_hidden
        model.classifier = nn.Linear(fusion_dim, 1)
    return model


def run_variant(cfg, variant, train_dl, val_dl, test_dl, device):
    set_seed(cfg.train.seed)
    model = build_variant_model(cfg, variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    total_steps = len(train_dl) * cfg.train.max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(total_steps, cfg.train.warmup_ratio))
    augmenter = RandomAugmentation(cfg.aug)

    zero_static = (variant == "CLSR_s")
    zero_hist = (variant == "CLSR_d")

    step = 0
    for epoch in range(cfg.train.max_epochs):
        for intraday, static_hist, inter_day, labels in train_dl:
            B, L, F_ = intraday.shape
            if zero_static:
                static_hist = torch.zeros_like(static_hist)
            static_exp = static_hist.unsqueeze(1).expand(B, L, static_hist.shape[-1])
            x_cat = torch.cat([intraday, static_exp], dim=-1)
            view_i = torch.stack([augmenter(x_cat[b]) for b in range(B)]).to(device)
            view_j = torch.stack([augmenter(x_cat[b]) for b in range(B)]).to(device)
            inter_day = inter_day.to(device)
            if zero_hist:
                inter_day = torch.zeros_like(inter_day)
            labels = labels.to(device)

            p_i, p_j, logits = model(view_i, view_j, inter_day)
            loss, _, _ = joint_loss(logits, labels, p_i, p_j,
                                     cfg.train.temperature, cfg.train.alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
    return evaluate(model, test_dl, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="synthetic",
                         choices=["synthetic", "tushare", "hf_minute"])
    parser.add_argument("--n_stocks", type=int, default=10)
    parser.add_argument("--codes", type=str, default="", help="tushare codes, comma-separated")
    parser.add_argument("--tickers", type=str, default="", help="hf_minute tickers, comma-separated")
    parser.add_argument("--hf_data_dir", type=str, default=None)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    cfg = CLSRConfig()
    cfg.data.backend = args.backend
    cfg.train.max_epochs = args.epochs
    if args.hf_data_dir:
        cfg.data.hf_data_dir = args.hf_data_dir
    if args.start_date:
        cfg.data.start_date = args.start_date
    if args.end_date:
        cfg.data.end_date = args.end_date
    if args.seq_len:
        cfg.data.intraday_len = args.seq_len
    elif args.backend == "hf_minute":
        cfg.data.intraday_len = 390

    codes = None
    if args.backend == "tushare":
        codes = args.codes.split(",") if args.codes else None
    elif args.backend == "hf_minute":
        codes = args.tickers.split(",") if args.tickers else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, test_dl = build_dataloaders(cfg, args.n_stocks, codes)

    results = {}
    for variant in ["CLSR_T", "CLSR_d", "CLSR_s", "CLSR"]:
        m = run_variant(cfg, variant, train_dl, val_dl, test_dl, device)
        results[variant] = m
        print(f"{variant}: ACC={m['ACC']:.2f}  MCC={m['MCC']:.2f}")

    print("\n| Method | ACC (%) | MCC (%) |")
    print("|---|---|---|")
    for k, v in results.items():
        print(f"| {k} | {v['ACC']:.2f} | {v['MCC']:.2f} |")


if __name__ == "__main__":
    main()
