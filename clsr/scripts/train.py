"""
Training entry point implementing Algorithm 1 end-to-end:
  for each minibatch of intraday sequences:
    draw two augmentations per sample -> encode both views
    compute NT-Xent contrastive loss over the 2N view batch
    compute BCE supervised loss on view i's representation + historical state
    backprop L_ce + alpha * L_con
    evaluate on val set every `eval_every_steps`; keep best checkpoint

Usage:
    python scripts/train.py --backend synthetic --n_stocks 20 --epochs 3
    python scripts/train.py --backend tushare --codes 000001.SZ,000002.SZ ...
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLSRConfig
from data.dataset import CLSRDataset, collate_fn, load_synthetic, load_tushare, load_hf_minute
from data.augmentations import RandomAugmentation
from models.clsr import CLSRModel
from losses import joint_loss
from metrics import compute_acc_mcc


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_dataloaders(cfg: CLSRConfig, n_stocks: int, codes=None):
    """codes: for "tushare", a list of Tushare-format codes (e.g.
    '000001.SZ'); for "hf_minute", a list of tickers whose files live under
    cfg.data.hf_data_dir (e.g. 'AAPL', 'MSFT', ...); unused for "synthetic".
    """
    if cfg.data.backend == "synthetic":
        stock_data = load_synthetic(cfg.data, n_stocks=n_stocks, seed=cfg.train.seed)
    elif cfg.data.backend == "tushare":
        assert codes, "must pass --codes for tushare backend"
        stock_data = load_tushare(cfg.data, codes)
    elif cfg.data.backend == "hf_minute":
        assert codes, "must pass --tickers for hf_minute backend"
        stock_data = load_hf_minute(cfg.data, codes)
        if not stock_data:
            raise RuntimeError(
                f"No usable data loaded from {cfg.data.hf_data_dir} for tickers {codes}. "
                f"Check that files exist (e.g. {cfg.data.hf_data_dir}/AAPL.csv) and that "
                f"cfg.data.start_date/end_date overlap the file's date range."
            )
    else:
        raise ValueError(cfg.data.backend)

    train_ds = CLSRDataset(stock_data, cfg.data, split="train")
    val_ds = CLSRDataset(stock_data, cfg.data, split="val")
    test_ds = CLSRDataset(stock_data, cfg.data, split="test")

    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=cfg.train.num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=cfg.train.num_workers)
    test_dl = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=cfg.train.num_workers)
    return train_dl, val_dl, test_dl


def make_augmented_views(intraday, static_hist, augmenter: RandomAugmentation, device):
    """intraday: (B, L, F), static_hist: (B, 3). Builds two augmented views
    with S1,S2,S3 concatenated as extra channels (Eq. 10:
    concat([X, (S1,S2,S3)]) before augmentation/positional encoding)."""
    B, L, F_ = intraday.shape
    static_exp = static_hist.unsqueeze(1).expand(B, L, static_hist.shape[-1])
    x_cat = torch.cat([intraday, static_exp], dim=-1)  # (B, L, F+3)

    view_i = torch.stack([augmenter(x_cat[b]) for b in range(B)]).to(device)
    view_j = torch.stack([augmenter(x_cat[b]) for b in range(B)]).to(device)
    return view_i, view_j


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_true, all_pred = [], []
    for intraday, static_hist, inter_day, labels in dataloader:
        B, L, F_ = intraday.shape
        static_exp = static_hist.unsqueeze(1).expand(B, L, static_hist.shape[-1])
        x_cat = torch.cat([intraday, static_exp], dim=-1).to(device)
        inter_day = inter_day.to(device)
        probs = model.predict(x_cat, inter_day)
        preds = (probs > 0.5).long().cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(labels.long().numpy().tolist())
    model.train()
    if len(all_true) == 0:
        return {"ACC": 0.0, "MCC": 0.0}
    return compute_acc_mcc(all_true, all_pred)


def get_lr_lambda(total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="synthetic",
                         choices=["synthetic", "tushare", "hf_minute"])
    parser.add_argument("--n_stocks", type=int, default=20)
    parser.add_argument("--codes", type=str, default="",
                         help="Tushare codes, comma-separated (tushare backend only)")
    parser.add_argument("--tickers", type=str, default="",
                         help="Tickers, comma-separated, e.g. AAPL,MSFT,GOOG "
                              "(hf_minute backend only)")
    parser.add_argument("--hf_data_dir", type=str, default=None,
                         help="Dir containing one file per ticker, e.g. AAPL.csv "
                              "(hf_minute backend only)")
    parser.add_argument("--hf_file_format", type=str, default=None, choices=[None, "csv", "parquet"])
    parser.add_argument("--start_date", type=str, default=None,
                         help="Overrides cfg.data.start_date, e.g. 2018-01-01")
    parser.add_argument("--end_date", type=str, default=None,
                         help="Overrides cfg.data.end_date, e.g. 2021-12-31 "
                              "(stay pre-March-2022 for hf_minute if your library "
                              "switches data source after that date)")
    parser.add_argument("--seq_len", type=int, default=None,
                         help="Bars per trading day: 241 for China A-shares "
                              "(9:30-11:30+13:00-15:00), 390 for a full US "
                              "regular session (9:30-16:00)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--out", type=str, default="./checkpoints")
    args = parser.parse_args()

    cfg = CLSRConfig()
    cfg.data.backend = args.backend
    if args.epochs:
        cfg.train.max_epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.hf_data_dir:
        cfg.data.hf_data_dir = args.hf_data_dir
    if args.hf_file_format:
        cfg.data.hf_file_format = args.hf_file_format
    if args.start_date:
        cfg.data.start_date = args.start_date
    if args.end_date:
        cfg.data.end_date = args.end_date
    if args.seq_len:
        cfg.data.intraday_len = args.seq_len
    elif args.backend == "hf_minute":
        # default to a full continuous US session unless overridden
        cfg.data.intraday_len = 390

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    if args.backend == "tushare":
        codes = args.codes.split(",") if args.codes else None
    elif args.backend == "hf_minute":
        codes = args.tickers.split(",") if args.tickers else None
    else:
        codes = None
    train_dl, val_dl, test_dl = build_dataloaders(cfg, args.n_stocks, codes)
    print(f"train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    model = CLSRModel(cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    total_steps = len(train_dl) * cfg.train.max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(total_steps, cfg.train.warmup_ratio))

    augmenter = RandomAugmentation(cfg.aug)

    best_mcc = -1e9
    patience = 0
    step = 0
    t0 = time.time()

    for epoch in range(cfg.train.max_epochs):
        for intraday, static_hist, inter_day, labels in train_dl:
            labels = labels.to(device)
            inter_day = inter_day.to(device)
            view_i, view_j = make_augmented_views(intraday, static_hist, augmenter, device)

            p_i, p_j, logits = model(view_i, view_j, inter_day)
            loss, l_ce, l_con = joint_loss(
                logits, labels, p_i, p_j, cfg.train.temperature, cfg.train.alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % cfg.train.eval_every_steps == 0:
                val_metrics = evaluate(model, val_dl, device)
                elapsed = time.time() - t0
                print(f"step={step} epoch={epoch} loss={loss.item():.4f} "
                      f"ce={l_ce.item():.4f} con={l_con.item():.4f} "
                      f"val_ACC={val_metrics['ACC']:.2f} val_MCC={val_metrics['MCC']:.2f} "
                      f"({elapsed:.1f}s)")
                if val_metrics["MCC"] > best_mcc:
                    best_mcc = val_metrics["MCC"]
                    patience = 0
                    torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
                else:
                    patience += 1
                if patience >= cfg.train.early_stop_patience:
                    print("Early stopping.")
                    break
        else:
            continue
        break

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt")))
    test_metrics = evaluate(model, test_dl, device)
    print(f"FINAL TEST  ACC={test_metrics['ACC']:.2f}  MCC={test_metrics['MCC']:.2f}")


if __name__ == "__main__":
    main()
