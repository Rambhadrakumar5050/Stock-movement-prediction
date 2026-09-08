# CLSR — Contrastive Learning for Stock Representations

Reimplementation of:

> Feng, W., Ma, X., Li, X., Zhang, C. "A representation learning framework
> for stock movement prediction." *Applied Soft Computing* 144 (2023) 110409.

## What's implemented, mapped to the paper

| Paper section | File |
|---|---|
| Eq. 1, Section 3 (problem formulation, labels) | `data/dataset.py::compute_label` |
| Eq. 3-5, Section 4.1 (static historical state S1,S2,S3) | `data/dataset.py::compute_static_hist_features` |
| Eq. 6-9, Fig. 2 (dynamic historical state, LSTM+Attention) | `models/historical_state.py` |
| Eq. 10-14, Section 4.2.1 (Global Information Net, ProbSparse attention) | `models/global_net.py`, `models/attention.py` |
| Eq. 15, Section 4.2.2 (Local Information Net, TCN) | `models/local_net.py` |
| Section 4.2 (hybrid encoder = concat of global+local) | `models/clsr.py::HybridEncoder` |
| Eq. 16, Section 4.3 (NT-Xent contrastive loss) | `losses.py::nt_xent_loss` |
| Eq. 17-19, Section 4.4 (supervised + joint loss) | `losses.py::supervised_bce_loss`, `joint_loss` |
| Section 4.5 (4 augmentation strategies) | `data/augmentations.py` |
| Local 1-minute OHLCV library backend (US equities, 390-bar session) | `data/dataset.py::load_hf_minute` |
| Algorithm 1 (full training loop) | `scripts/train.py` |
| Eq. 20-21, Section 5.1.3 (ACC / MCC) | `metrics.py` |
| Table 3 (ablations: CLSR_T, CLSR_d, CLSR_s) | `scripts/ablation.py` |
| Section 5.4, Tables 4-5 (TopK-Drop investment simulation) | `scripts/investment_sim.py` |
| Section 5.5, Fig. 3 (augmentation grid) | `data/augmentations.py::compose` (grid loop left to you — trivial nested call) |
| Section 5.6, Figs. 4-9 (representation-space PCA/KMeans/cosine) | `scripts/representation_analysis.py` |

## Quickstart (synthetic data — verifies the whole pipeline runs)

```bash
pip install -r requirements.txt
cd clsr
python scripts/train.py --backend synthetic --n_stocks 20 --epochs 3
python scripts/ablation.py --backend synthetic --n_stocks 10 --epochs 2
python scripts/investment_sim.py --ckpt ./checkpoints/best.pt --backend synthetic
python scripts/representation_analysis.py --clsr_ckpt ./checkpoints/best.pt
```

## Using a local 1-minute OHLCV library (e.g. a downloaded multi-ticker US-equity dataset)

If you have per-ticker 1-minute CSV/Parquet files downloaded locally (e.g.
`AAPL.csv`, `MSFT.csv`, ... in one folder, each with a timestamp column plus
open/high/low/close/volume), use the `hf_minute` backend:

```bash
python scripts/train.py --backend hf_minute \
    --tickers AAPL,MSFT,GOOG,AMZN,NVDA,META,TSLA,JPM \
    --hf_data_dir ./hf_data \
    --start_date 2018-01-01 --end_date 2021-12-31 \
    --seq_len 390 \
    --n_stocks 8
```

Key differences from the paper's China A-share setup, all handled
automatically by `data/dataset.py::load_hf_minute`:

- **Session length**: US regular trading hours (9:30–16:00, no lunch break)
  are 390 one-minute bars, vs. the paper's 241 (China's split
  9:30–11:30 + 13:00–15:00 session). Pass `--seq_len 390`, or just omit it
  — the CLI defaults `intraday_len` to 390 automatically when
  `--backend hf_minute` is used. Nothing in the model architecture depends
  on this number (Transformer/TCN/attention all work over any sequence
  length), so this is purely a data-shape parameter.
- **Column-name tolerance**: the loader accepts `Open/High/Low/Close/Volume`,
  `o/h/l/c/v`, `Adj Close`, etc. and normalizes them; set
  `cfg.data.hf_datetime_col` if your timestamp column isn't named
  `datetime`.
- **Real-world gaps**: unlike the paper's presumably vendor-cleaned feed,
  downloaded minute data often has missing bars (halts, thin liquidity,
  etc.). Each trading day is reindexed onto a complete 390-minute grid and
  small gaps are forward/backward-filled; a day missing more than
  `1 - cfg.data.hf_min_bars_frac` (default 10%) of its bars is dropped
  entirely rather than mostly fabricated — tune `hf_min_bars_frac` if
  you're seeing too many/few days survive.
- **Pre/post data-source-change split**: if your library switches vendors
  partway through (e.g. a March-2022 cutover), pass `--end_date` to stay on
  one side of that boundary so you're not training across two different
  data-quality regimes; `--start_date`/`--end_date` also double as your
  train/val/test window per `config.py`'s `train_years`/`val_months`/
  `test_months` split.
- This is a genuinely different market/dataset from the paper (US equities
  vs. CSI-500 China A-shares), so treat results from this backend as a
  **cross-market generalization check on the CLSR methodology**, not a
  reproduction of Table 2 — report it as such rather than comparing the
  numbers directly.

## Using real CSI-500 data (to actually match Table 2's numbers)

The paper's exact dataset — minute-level OHLCV for 500 CSI-500 stocks,
2015-12-01 to 2019-12-01, from Tushare — is **not free**: minute-level
history on Tushare Pro requires a paid points tier. There's no way around
this; I can't download it for you. Once you have a token:

```bash
export TUSHARE_TOKEN=your_token_here
python scripts/train.py --backend tushare --codes 000001.SZ,000002.SZ,...  # all 500 CSI-500 constituents
```

`data/dataset.py::load_tushare` handles the pull + local CSV caching so you
only download each stock once. You'll need the current CSI-500 constituent
list (also fetchable via `pro.index_weight(index_code='000905.SH', ...)`)
to build the `--codes` list for the paper's exact universe.

## Important: hyperparameters the paper doesn't state

Section 5.2 gives: sequence length 241, cutoff ratio 0.15, dropout 0.2,
τ=0.1, α=0.15, Adam, lr=1e-3, 20% linear warmup, eval every 100 steps.
It does **not** state: batch size N, model width (d_model), number of
Transformer layers/heads, TCN channel widths, LSTM hidden size, or total
training steps/epochs (only that 150k DA iterations / 5 MR rounds are a
*different* paper's numbers — re-read: actually CLSR doesn't state total
steps at all). Every such value is marked `# ASSUMED` in `config.py` with
a comment. **These are the values most likely to cause your reproduced
numbers to differ from Table 2** — if you can find the authors' Code Ocean
capsule (linked in the paper, `https://doi.org/10.24433/CO.0217135.v1`)
still live, cross-check against it and update `config.py` accordingly.

## Known gaps / simplifications vs. the paper

- **Investment simulation** (`investment_sim.py`) is a placeholder: a
  faithful day-by-day TopK-Drop backtest needs each sample's trading day
  and realized forward return threaded through the dataset, which
  `CLSRDataset` doesn't currently retain. The script says so at runtime
  and shows exactly what to add.
- **Label rule** (`compute_label`): Table 1's two thresholds are used as
  a rise/fall dead-zone cutoff; the paper's Eq. 1 and its own prose
  disagree on the sign convention (see the docstring in `dataset.py`) so
  this is the one function to flip if your validation accuracy looks
  inverted.
- **ProbSparse attention** (`models/attention.py`) is a faithful
  from-scratch reimplementation of the Informer mechanism the paper cites
  (ref [39]), not copied from any existing repo.
- Data augmentation window sizes `alpha, beta` named in Algorithm 1's
  input list are never numerically specified in the text; `AugConfig`
  exposes reasonable defaults for you to tune.

## Environment note

This was written and syntax-checked in a sandbox without a working CUDA
toolchain, so it hasn't been execution-tested end-to-end here. The code
is plain PyTorch + numpy/pandas/sklearn with no exotic ops — on Kaggle
(or any machine with a normal torch install) it should run as-is; if you
hit a shape mismatch, the most likely spot is the inter-day feature width
(`in_dim=1` in `CLSRModel.hist_encoder`, since the synthetic/Tushare
loader currently only feeds close price into the inter-day branch — widen
`FEATURE_COLS`-derived inter_day construction in `dataset.py` if you want
full OHLCV in the historical-state branch too).
