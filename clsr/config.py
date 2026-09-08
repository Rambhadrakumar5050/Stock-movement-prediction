"""
Central configuration for the CLSR reproduction.

All hyperparameters here are taken directly from the paper
("A representation learning framework for stock movement prediction",
Feng, Ma, Li, Zhang; Applied Soft Computing 144 (2023) 110409),
Section 5.2 "Implementation details" and Section 4/Algorithm 1.

Values not explicitly stated in the paper (marked "ASSUMED") are filled
in with reasonable defaults and are exposed here so you can tune them.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # ---- Paper Table 1 ----
    start_date: str = "2015-12-01"
    end_date: str = "2019-12-01"
    total_stocks: int = 500
    rising_threshold: float = -0.5      # beta_rise, %   (Table 1; sign as printed)
    falling_threshold: float = 0.105    # beta_fall, %   (Table 1)

    # ---- Paper Section 3 ----
    # L_K, minute bars per trading day. The paper's CSI-500/China A-share
    # session is 9:30-11:30 + 13:00-15:00 = 241 one-minute bars. A full
    # continuous US regular session (9:30-16:00, no lunch break) is 390
    # one-minute bars -- set this to 390 when backend="hf_minute".
    intraday_len: int = 241
    feature_dim: int = 5                # open, high, low, close, volume
    inter_day_len: int = 20             # m, ASSUMED window of daily closes (paper
                                         # says "N days closing price" but never
                                         # states N explicitly -> ASSUMED 20)

    # split: paper says 3 years / 6 months / 6 months, sequential (no shuffling,
    # to avoid lookahead leakage)
    train_years: float = 3.0
    val_months: float = 6.0
    test_months: float = 6.0

    # Where to find data. Supported backends:
    #   "synthetic"   - generated data, for pipeline verification only
    #   "tushare"     - real CSI-500 minute data (paid Tushare Pro token)
    #   "hf_minute"   - local CSV/Parquet 1-minute OHLCV files per ticker
    #                   (e.g. downloaded from a public 1-minute US-equity
    #                   library) -- see data/dataset.py::load_hf_minute
    backend: str = "synthetic"
    tushare_token: str = ""             # set via env var TUSHARE_TOKEN normally
    data_dir: str = "./raw_data"        # cache dir for downloaded / synthetic csvs

    # ---- hf_minute backend specific ----
    hf_data_dir: str = "./hf_data"      # dir containing one file per ticker
    hf_file_format: str = "csv"         # "csv" or "parquet"
    hf_datetime_col: str = "datetime"   # column holding the bar timestamp
    hf_session_start: str = "09:30"     # US regular session open
    hf_session_end: str = "16:00"       # US regular session close (exclusive of 16:00 bar)
    hf_min_bars_frac: float = 0.9       # drop a trading day if fewer than this
                                         # fraction of expected bars are present
                                         # even after reindex+ffill (data gaps)


@dataclass
class AugConfig:
    # Section 4.5. "window size alpha, beta" mentioned in Algorithm 1 but not
    # numerically specified beyond the cutoff ratio below -> ASSUMED defaults.
    cutoff_ratio: float = 0.15          # stated explicitly in 5.2 ("ratio of the cuts is 0.15")
    noise_std: float = 0.02             # ASSUMED (fraction of per-feature std)
    fft_keep_ratio: float = 0.9         # ASSUMED: fraction of low-freq FFT components kept
    span_cutoff_len_ratio: float = 0.15 # ASSUMED: fraction of tail sequence dropped by span cutoff
    dropout_p: float = 0.2              # stated explicitly in 5.2 ("Dropout is set to 0.2")


@dataclass
class ModelConfig:
    feature_dim: int = 5
    # +3 static historical features S1,S2,S3 concatenated to intraday series (Eq. 10)
    static_hist_dim: int = 3
    d_model: int = 64                   # ASSUMED (not stated) — Transformer/TCN hidden size
    n_heads: int = 8                    # ASSUMED
    n_encoder_layers: int = 2           # ASSUMED
    tcn_channels: List[int] = field(default_factory=lambda: [64, 64, 64])
    tcn_kernel_size: int = 2            # stated: "filter kernel is 2"
    dropout: float = 0.2                # stated in 5.2

    lstm_hidden: int = 64               # ASSUMED, used both for historical-state LSTM (Sec 4.1)
    attn_hidden: int = 64               # ASSUMED, temporal-attention projection size (Eq. 9)

    proj_dim: int = 64                  # ASSUMED: dimension of Z_i / Z_j contrastive embedding


@dataclass
class TrainConfig:
    batch_size: int = 32                # ASSUMED (N in Algorithm 1; not numerically stated)
    max_seq_len: int = 241              # stated in 5.2
    lr: float = 1e-3                    # stated in 5.2
    warmup_ratio: float = 0.2           # stated in 5.2 ("linear warm-up over 20% of steps")
    temperature: float = 0.1            # stated in 5.2 (tau)
    alpha: float = 0.15                 # stated in 5.2 (loss balance hyperparameter)
    eval_every_steps: int = 100         # stated in 5.2
    max_epochs: int = 50                # ASSUMED — paper reports steps, not epochs
    early_stop_patience: int = 10       # ASSUMED
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 2


@dataclass
class CLSRConfig:
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
