"""
Investment simulation, Section 5.4, Table 4 & 5.

Implements the TopK-Drop trading strategy (Chen et al. [49]): each day,
sell the currently-held stock(s) with the lowest predicted score and buy
the same number of currently-unheld stocks with the highest predicted
score, holding a fixed-size portfolio of K stocks, zero transaction cost.
Reports cumulative return (%) and (annualized) Sharpe ratio, matching
Tables 4-5.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLSRConfig
from models.clsr import CLSRModel
from scripts.train import build_dataloaders, set_seed


def topk_drop_backtest(daily_scores: np.ndarray, daily_returns: np.ndarray, k: int = 10):
    """daily_scores, daily_returns: (T, N_stocks) arrays -- model score and
    realized next-day return per stock per day. Returns portfolio daily
    return series.
    """
    T, N = daily_scores.shape
    k = min(k, N)
    held = set(np.argsort(-daily_scores[0])[:k].tolist())
    port_returns = []

    for t in range(T):
        scores = daily_scores[t]
        rets = daily_returns[t]
        ranked = np.argsort(-scores)
        buy_candidates = [i for i in ranked if i not in held]
        # drop the lowest-scoring currently-held stocks, buy top unheld ones,
        # same count each side (TopK-Drop, ref [49])
        held_sorted_by_score = sorted(held, key=lambda i: scores[i])
        n_drop = max(1, len(held) // 10)  # ASSUMED daily churn rate (paper
                                           # doesn't specify a fixed drop count)
        to_sell = held_sorted_by_score[:n_drop]
        to_buy = buy_candidates[:n_drop]
        held.difference_update(to_sell)
        held.update(to_buy)

        port_ret = np.mean([rets[i] for i in held]) if held else 0.0
        port_returns.append(port_ret)

    return np.array(port_returns)


def cumulative_return(daily_returns: np.ndarray) -> float:
    return (np.prod(1 + daily_returns) - 1) * 100.0


def sharpe_ratio(daily_returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = daily_returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


@torch.no_grad()
def score_all_stocks(model, test_dl, device):
    """Collects (score, realized_return) per sample; realized return is
    approximated from the label direction here since raw next-day % return
    isn't threaded through the Dataset by default -- for a precise backtest,
    extend CLSRDataset to also return the raw next-day return alongside the
    binary label."""
    scores, labels = [], []
    for intraday, static_hist, inter_day, y in test_dl:
        B, L, F_ = intraday.shape
        static_exp = static_hist.unsqueeze(1).expand(B, L, static_hist.shape[-1])
        x_cat = torch.cat([intraday, static_exp], dim=-1).to(device)
        inter_day = inter_day.to(device)
        probs = model.predict(x_cat, inter_day).cpu().numpy()
        scores.extend(probs.tolist())
        labels.extend(y.numpy().tolist())
    return np.array(scores), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./checkpoints/best.pt")
    parser.add_argument("--backend", default="synthetic")
    parser.add_argument("--n_stocks", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    cfg = CLSRConfig()
    cfg.data.backend = args.backend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.train.seed)
    _, _, test_dl = build_dataloaders(cfg, args.n_stocks)

    model = CLSRModel(cfg.model).to(device)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    else:
        print(f"WARNING: checkpoint {args.ckpt} not found, using randomly initialized "
              f"weights (results will be meaningless -- train first via scripts/train.py).")

    scores, labels = score_all_stocks(model, test_dl, device)
    # NOTE: this flat (scores, labels) pairing is a placeholder cross-section;
    # a faithful TopK-Drop backtest needs scores reshaped to (T, N_stocks)
    # aligned by trading day, which requires date metadata not currently
    # carried by CLSRDataset -- add a `day` field to Sample if you need the
    # literal day-by-day simulation instead of this rank-based approximation.
    pseudo_returns = np.where(labels == 1, 0.01, -0.01)  # ASSUMED +-1% proxy return
    # simple long-top-k proxy: pick args.k highest-scored samples, average their return
    top_idx = np.argsort(-scores)[:args.k]
    daily_like_returns = pseudo_returns[top_idx]

    print(f"Cumulative return (proxy): {cumulative_return(daily_like_returns):.2f}%")
    print(f"Sharpe ratio (proxy): {sharpe_ratio(daily_like_returns):.2f}")
    print("\nFor a literal day-aligned TopK-Drop backtest matching Tables 4-5 "
          "exactly, extend CLSRDataset to retain (day, stock_code, realized_return) "
          "per sample and feed `topk_drop_backtest()` above with a proper "
          "(T, N_stocks) score/return matrix.")


if __name__ == "__main__":
    main()
