"""
Dataset construction for CLSR (Section 3, Section 4.1, Section 5.1.1).

Three backends:
  - "tushare": pulls real CSI-500 minute bars via the Tushare Pro API
    (https://tushare.pro/), matching the paper's stated data source. This
    REQUIRES your own paid Tushare token (minute-level history is not on
    the free tier) — set TUSHARE_TOKEN env var or config.data.tushare_token.
  - "hf_minute": loads local CSV/Parquet 1-minute OHLCV files (one file per
    ticker) such as those exported from a public 1-minute US-equity data
    library. Handles the different (390-bar, no lunch break) US session
    structure, reindexes each trading day onto a complete minute grid, and
    forward-fills small gaps -- real free/downloaded data is rarely as
    clean as the paper's vendor-cleaned feed. See `load_hf_minute` below.
  - "synthetic": generates geometric-Brownian-motion-like minute bars with
    realistic OHLCV structure, purely so you can validate the full pipeline
    (shapes, losses, training loop, metrics) end-to-end without any data
    download at all.
All three backends emit the same schema (a dict of {ticker: DataFrame} with
columns FEATURE_COLS + "day" + "timestamp"), so nothing downstream of
loading needs to change when you swap backends.

Label definition (Eq. 1):
    y = 1[ p_{T-dt+1} > p_T ]     (paper's eq. literally: 1 if price fell)
The paper's prose says y in {1, -1} corresponding to "fall and rise", which
is inverted relative to how Eq. 1 reads literally (p_{T-dt+1} > p_T means
price DECREASED over the window, yet the values 1/-1 are labelled "fall"/
"rise" in that order matching the eq.). We implement the *tradeable*
convention explicitly: label = 1 if next-day return > rising_threshold,
0 (treated as -1 in the loss) if below falling_threshold-derived cutoff;
see `compute_label` for the exact rule taken from Table 1's two
thresholds. This choice is documented so you can flip it in one place if
your reproduction disagrees.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset

from config import DataConfig

FEATURE_COLS = ["open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------
# Backend 1: synthetic data (for pipeline verification)
# --------------------------------------------------------------------------
def _make_synthetic_minute_bars(n_days: int, bars_per_day: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = n_days * bars_per_day
    # log-return random walk with small daily drift + occasional regime shift
    drift = rng.normal(0, 1e-5, size=n)
    vol = np.abs(rng.normal(3e-4, 1e-4, size=n))
    log_ret = rng.normal(drift, vol)
    close = 10.0 * np.exp(np.cumsum(log_ret))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 5e-4, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 5e-4, size=n)))
    volume = np.abs(rng.normal(1e5, 3e4, size=n))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
    return df


def load_synthetic(cfg: DataConfig, n_stocks: int, seed: int = 0) -> dict:
    """Returns {stock_id: DataFrame} each with n_days * 241 minute rows,
    a 'day' column, and 'timestamp' column (integer minute index)."""
    total_days = int(365 * (cfg.train_years) + 30 * (cfg.val_months + cfg.test_months))
    out = {}
    for i in range(n_stocks):
        df = _make_synthetic_minute_bars(total_days, cfg.intraday_len, seed=seed + i)
        df["day"] = np.repeat(np.arange(total_days), cfg.intraday_len)
        df["timestamp"] = np.tile(np.arange(cfg.intraday_len), total_days)
        out[f"SYN{i:04d}"] = df
    return out


# --------------------------------------------------------------------------
# Backend 2: Tushare (real data)
# --------------------------------------------------------------------------
def load_tushare(cfg: DataConfig, stock_codes: list) -> dict:
    """Pulls minute bars for each code via tushare pro API. Requires
    `pip install tushare` and a paid token (minute-level data is gated).
    Caches each stock's CSV under cfg.data_dir so re-runs don't re-download.
    """
    try:
        import tushare as ts
    except ImportError as e:
        raise ImportError(
            "tushare is not installed. Run `pip install tushare --break-system-packages`."
        ) from e

    token = cfg.tushare_token or os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise ValueError(
            "No Tushare token found. Set config.data.tushare_token or the "
            "TUSHARE_TOKEN environment variable. Minute-level CSI-500 data "
            "requires a paid Tushare Pro subscription."
        )
    ts.set_token(token)
    pro = ts.pro_api()
    os.makedirs(cfg.data_dir, exist_ok=True)

    out = {}
    for code in stock_codes:
        cache_path = os.path.join(cfg.data_dir, f"{code}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
        else:
            # pro_bar with freq='1min' is the documented way to fetch
            # minute bars; date range from the paper's Table 1.
            df = ts.pro_bar(
                ts_code=code,
                api=pro,
                freq="1min",
                start_date=cfg.start_date.replace("-", ""),
                end_date=cfg.end_date.replace("-", ""),
            )
            if df is None or len(df) == 0:
                continue
            df = df.sort_values("trade_time").reset_index(drop=True)
            df = df.rename(columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "vol": "volume",
            })
            df.to_csv(cache_path, index=False)
        df["day"] = pd.to_datetime(df["trade_time"]).dt.date
        df["day"] = df["day"].astype("category").cat.codes
        df["timestamp"] = df.groupby("day").cumcount()
        out[code] = df[["open", "high", "low", "close", "volume", "day", "timestamp"]]
    return out


# --------------------------------------------------------------------------
# Backend 3: local CSV/Parquet 1-minute OHLCV library (e.g. a downloaded
# multi-ticker US-equity 1-minute dataset)
# --------------------------------------------------------------------------
def _read_one_ticker_file(path: str, file_format: str, datetime_col: str) -> pd.DataFrame:
    if file_format == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # normalize column names: accept common variants (Open/High/Low/Close/
    # Volume, o/h/l/c/v, etc.) so this works across differently-formatted
    # exports without the user having to rename columns by hand.
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("open", "o"):
            col_map[c] = "open"
        elif cl in ("high", "h"):
            col_map[c] = "high"
        elif cl in ("low", "l"):
            col_map[c] = "low"
        elif cl in ("close", "c", "adj close", "adj_close"):
            col_map[c] = "close"
        elif cl in ("volume", "vol", "v"):
            col_map[c] = "volume"
        elif cl == datetime_col.lower():
            col_map[c] = "datetime"
    df = df.rename(columns=col_map)

    missing = [c for c in ["datetime"] + FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing expected column(s) {missing} after normalization; "
            f"found columns {list(df.columns)}. Set cfg.data.hf_datetime_col if "
            f"your timestamp column has a different name."
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[["datetime"] + FEATURE_COLS].sort_values("datetime").reset_index(drop=True)


def _reindex_trading_day(day_df: pd.DataFrame, session_start: str, session_end: str,
                          min_bars_frac: float) -> pd.DataFrame | None:
    """Reindexes one calendar day's bars onto a complete 1-minute grid from
    session_start to session_end (exclusive of the closing minute, matching
    standard "9:30-16:00 -> 390 bars" convention), forward-filling gaps.
    Returns None if too much of the day is missing (> (1-min_bars_frac) of
    expected bars absent BEFORE fill), signalling the day should be dropped
    rather than papered over with mostly-fabricated fills.
    """
    if day_df.empty:
        return None
    date = day_df["datetime"].iloc[0].normalize()
    start = date + pd.Timedelta(session_start + ":00")
    end = date + pd.Timedelta(session_end + ":00")
    full_index = pd.date_range(start, end, freq="1min", inclusive="left")

    present_frac = len(day_df) / max(1, len(full_index))
    if present_frac < min_bars_frac:
        return None  # too many real gaps -- don't fabricate most of a day

    day_df = day_df.set_index("datetime")
    day_df = day_df[~day_df.index.duplicated(keep="last")]
    day_df = day_df.reindex(full_index)
    day_df[FEATURE_COLS] = day_df[FEATURE_COLS].ffill().bfill()
    day_df = day_df.reset_index().rename(columns={"index": "datetime"})
    return day_df


def load_hf_minute(cfg: DataConfig, tickers: list) -> dict:
    """Loads local per-ticker 1-minute OHLCV files (CSV or Parquet) from
    cfg.hf_data_dir, e.g. files downloaded from a public 1-minute US-equity
    data library such as the one described in the project notes (1,391 US
    stocks/ETFs, 2002-present, PiTrading tape pre-March-2022 / IEX after).

    Expected file layout: `{cfg.hf_data_dir}/{TICKER}.csv` (or `.parquet`),
    each with a timestamp column (name configurable via
    cfg.hf_datetime_col) plus open/high/low/close/volume columns (case- and
    naming-variant tolerant, see `_read_one_ticker_file`).

    Trading-day length in this backend is whatever cfg.intraday_len is set
    to -- for the full US regular session (9:30-16:00, no lunch break) that
    should be 390, not the paper's 241 (which reflects China A-shares'
    split morning/afternoon session with a lunch break). Set
    cfg.intraday_len = 390 when using this backend.
    """
    out = {}
    for ticker in tickers:
        path_csv = os.path.join(cfg.hf_data_dir, f"{ticker}.csv")
        path_parquet = os.path.join(cfg.hf_data_dir, f"{ticker}.parquet")
        if cfg.hf_file_format == "parquet" and os.path.exists(path_parquet):
            path, fmt = path_parquet, "parquet"
        elif os.path.exists(path_csv):
            path, fmt = path_csv, "csv"
        elif os.path.exists(path_parquet):
            path, fmt = path_parquet, "parquet"
        else:
            print(f"WARNING: no data file found for {ticker} in {cfg.hf_data_dir}, skipping.")
            continue

        raw = _read_one_ticker_file(path, fmt, cfg.hf_datetime_col)

        # restrict to the configured date range (e.g. pre-March-2022 to stay
        # on the consolidated-tape portion of the library rather than the
        # IEX-only post-2022 portion)
        start = pd.Timestamp(cfg.start_date)
        end = pd.Timestamp(cfg.end_date)
        raw = raw[(raw["datetime"] >= start) & (raw["datetime"] <= end)]
        if raw.empty:
            print(f"WARNING: {ticker} has no rows in [{cfg.start_date}, {cfg.end_date}], skipping.")
            continue

        raw["cal_date"] = raw["datetime"].dt.date
        day_frames = []
        for _, day_df in raw.groupby("cal_date", sort=True):
            reindexed = _reindex_trading_day(
                day_df.drop(columns="cal_date"),
                cfg.hf_session_start, cfg.hf_session_end, cfg.hf_min_bars_frac)
            if reindexed is not None and len(reindexed) == cfg.intraday_len:
                day_frames.append(reindexed)

        if not day_frames:
            print(f"WARNING: {ticker} had no usable complete trading days, skipping.")
            continue

        n_days = len(day_frames)
        full = pd.concat(day_frames, ignore_index=True)
        full["day"] = np.repeat(np.arange(n_days), cfg.intraday_len)
        full["timestamp"] = np.tile(np.arange(cfg.intraday_len), n_days)
        out[ticker] = full[FEATURE_COLS + ["day", "timestamp"]]
        print(f"{ticker}: loaded {n_days} complete trading days "
              f"({cfg.intraday_len} bars/day).")
    return out


# --------------------------------------------------------------------------
# Feature engineering: normalization, static historical features S1,S2,S3
# --------------------------------------------------------------------------
def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each of the 5 feature columns, matching the
    paper's "normalized following [8,45]" (Section 5.1.1)."""
    out = df.copy()
    for c in FEATURE_COLS:
        mu, sigma = out[c].mean(), out[c].std() + 1e-8
        out[c] = (out[c] - mu) / sigma
    return out


def compute_static_hist_features(daily_close: np.ndarray, day_idx_of_last: int) -> np.ndarray:
    """Eq. (3)-(5): S1, S2, S3 computed from the inter-day close series
    X' = [x'_1, ..., x'_m], using integer day index as the time function
    t(.) (the paper leaves t(.) abstract; using day index is the natural
    reading since only relative time differences matter)."""
    m = len(daily_close)
    if m < 2:
        return np.zeros(3, dtype=np.float32)
    t = np.arange(m)
    x_m = daily_close[-1]
    t_m = t[-1]

    x_1 = daily_close[0]
    t_1 = t[0]
    s1 = (x_1 - x_m) / (t_1 - t_m + 1e-8) if t_1 != t_m else 0.0

    i_max = np.argmax(daily_close)
    s2 = (daily_close[i_max] - x_m) / (t[i_max] - t_m + 1e-8) if t[i_max] != t_m else 0.0

    i_min = np.argmin(daily_close)
    s3 = (daily_close[i_min] - x_m) / (t[i_min] - t_m + 1e-8) if t[i_min] != t_m else 0.0

    return np.array([s1, s2, s3], dtype=np.float32)


def compute_label(next_day_close: float, cur_day_close: float, cfg: DataConfig) -> int:
    """Table 1 gives two thresholds (beta_rise = -0.5%, beta_fall = 0.105%)
    used to filter/label samples. We follow the common formulation in this
    line of work (e.g. Ding et al. [19]): compute the return
        r = (next_day_close - cur_day_close) / cur_day_close
    label 1 (rise) if r > beta_rise%, label 0 (fall) if r < beta_fall%.
    Samples with returns between the two thresholds are ambiguous/near-flat
    and are dropped, matching standard practice for this two threshold
    setup. NOTE: because Table 1's beta_rise (-0.5%) is numerically lower
    than beta_fall (0.105%), we take "rise" as the larger-return side and
    "fall" as the smaller-return side, i.e. the two numbers are used only
    as cutoffs for a middle dead-zone, not as directional signs — treat
    this function as the one place to edit if your data disagrees.
    """
    r = (next_day_close - cur_day_close) / (cur_day_close + 1e-8) * 100.0
    beta_rise = cfg.rising_threshold
    beta_fall = cfg.falling_threshold
    lo, hi = min(beta_rise, beta_fall), max(beta_rise, beta_fall)
    if r > hi:
        return 1
    if r < lo:
        return 0
    return -1  # ambiguous -> caller should drop


# --------------------------------------------------------------------------
# PyTorch Dataset
# --------------------------------------------------------------------------
@dataclass
class Sample:
    intraday: torch.Tensor      # (L_K, F)
    static_hist: torch.Tensor   # (3,)
    inter_day: torch.Tensor     # (m, F) -- daily closes etc. for LSTM branch
    label: int


class CLSRDataset(Dataset):
    """Builds one sample per (stock, day) pair: that day's intraday minute
    sequence (used for contrastive views + prediction), the trailing m-day
    inter-day window (for the historical-state LSTM+Attention branch,
    Section 4.1), the derived static features S1-S3, and the label for
    predicting the NEXT day's movement (Eq. 1).
    """

    def __init__(self, stock_data: dict, cfg: DataConfig, split: str = "train"):
        self.cfg = cfg
        self.samples = []
        for code, df in stock_data.items():
            df = normalize_features(df)
            days = sorted(df["day"].unique())
            daily_close_raw = df.groupby("day")["close"].last().values
            n_days = len(days)

            # sequential split: train / val / test by day index, no shuffling
            n_train = int(n_days * (cfg.train_years * 365) /
                          (cfg.train_years * 365 + cfg.val_months * 30 + cfg.test_months * 30))
            n_val = int(n_days * (cfg.val_months * 30) /
                        (cfg.train_years * 365 + cfg.val_months * 30 + cfg.test_months * 30))
            if split == "train":
                day_range = range(cfg.inter_day_len, n_train - 1)
            elif split == "val":
                day_range = range(max(n_train, cfg.inter_day_len), n_train + n_val - 1)
            else:
                day_range = range(max(n_train + n_val, cfg.inter_day_len), n_days - 1)

            for di in day_range:
                day = days[di]
                day_rows = df[df["day"] == day].sort_values("timestamp")
                if len(day_rows) != cfg.intraday_len:
                    continue  # skip incomplete trading days
                intraday = day_rows[FEATURE_COLS].values.astype(np.float32)

                inter_window = daily_close_raw[di - cfg.inter_day_len: di]
                static_hist = compute_static_hist_features(inter_window, di)

                next_day = days[di + 1]
                next_close = df[df["day"] == next_day]["close"].iloc[-1]
                cur_close = day_rows["close"].iloc[-1]
                label = compute_label(next_close, cur_close, cfg)
                if label == -1:
                    continue  # ambiguous dead-zone sample, dropped

                inter_feat = np.stack([
                    daily_close_raw[max(0, di - cfg.inter_day_len):di]
                ], axis=-1).astype(np.float32)  # (m, 1) -- close-only inter-day series

                self.samples.append(Sample(
                    intraday=torch.from_numpy(intraday),
                    static_hist=torch.from_numpy(static_hist),
                    inter_day=torch.from_numpy(inter_feat),
                    label=label,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s.intraday, s.static_hist, s.inter_day, s.label


def collate_fn(batch):
    intraday = torch.stack([b[0] for b in batch])
    static_hist = torch.stack([b[1] for b in batch])
    # inter_day sequences may have variable length near the start of history;
    # pad to the max length in the batch (paper takes fixed m; with real
    # data this padding path is rarely hit once di >= inter_day_len).
    max_m = max(b[2].shape[0] for b in batch)
    inter_day = torch.zeros(len(batch), max_m, batch[0][2].shape[1])
    for i, b in enumerate(batch):
        inter_day[i, -b[2].shape[0]:] = b[2]
    labels = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return intraday, static_hist, inter_day, labels
