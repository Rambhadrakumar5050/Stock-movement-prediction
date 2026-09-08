"""
Evaluation metrics, Section 5.1.3, Eq. (20)-(21).
"""
from __future__ import annotations
import numpy as np


def compute_acc_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred: 0/1 arrays. Returns ACC and MCC as percentages,
    matching Table 2's reporting convention."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {"ACC": acc * 100.0, "MCC": mcc * 100.0, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
