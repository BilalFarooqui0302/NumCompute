"""Classification and regression metrics implemented with NumPy."""
from __future__ import annotations
import numpy as np


def _labels(y):
    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError("label arrays must not be empty")
    return arr


def _check_pair(y_true, y_pred):
    true, pred = _labels(y_true), _labels(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape; got {true.shape} and {pred.shape}")
    return true, pred


def accuracy(y_true, y_pred):
    
    true, pred = _check_pair(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(y_true, y_pred, labels=None):
    
    true, pred = _check_pair(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate((true, pred)))
    else:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("labels must be a non-empty 1D array")
        if np.unique(labels).size != labels.size:
            raise ValueError("labels must not contain duplicates")
    mapping = {value: i for i, value in enumerate(labels.tolist())}
    try:
        true_idx = np.asarray([mapping[v] for v in true.tolist()], dtype=int)
        pred_idx = np.asarray([mapping[v] for v in pred.tolist()], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Found a label not included in labels: {exc.args[0]!r}") from exc
    cm = np.zeros((labels.size, labels.size), dtype=np.int64)
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm


def _binary_counts(y_true, y_pred, pos_label=1):
    true, pred = _check_pair(y_true, y_pred)
    actual = true == pos_label
    predicted = pred == pos_label
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    return tp, fp, fn, tn


def precision(y_true, y_pred, pos_label=1, zero_division=0):
   
    tp, fp, _, _ = _binary_counts(y_true, y_pred, pos_label)
    denom = tp + fp
    return float(tp / denom) if denom else float(zero_division)


def recall(y_true, y_pred, pos_label=1, zero_division=0):
    
    tp, _, fn, _ = _binary_counts(y_true, y_pred, pos_label)
    denom = tp + fn
    return float(tp / denom) if denom else float(zero_division)


def f1(y_true, y_pred, pos_label=1, zero_division=0):
    
    p = precision(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    r = recall(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    return float(2 * p * r / (p + r)) if p + r else float(zero_division)


def mse(y_true, y_pred):
    
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have identical shapes; got {true.shape} and {pred.shape}")
    if true.size == 0:
        raise ValueError("MSE is undefined for empty arrays")
    return float(np.mean((true - pred) ** 2))


def roc_curve(y_true, y_score, pos_label=1):
    
    true = _labels(y_true)
    score = np.asarray(y_score, dtype=float).reshape(-1)
    if true.shape != score.shape:
        raise ValueError("y_true and y_score must have the same number of elements")
    if not np.all(np.isfinite(score)):
        raise ValueError("y_score must contain only finite values")
    positive = true == pos_label
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC requires at least one positive and one negative example")
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_pos = positive[order].astype(int)
    tp_cum = np.cumsum(sorted_pos)
    fp_cum = np.cumsum(1 - sorted_pos)
    # Keep the final row from each group of equal score to avoid duplicate thresholds.
    group_ends = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), score.size - 1]
    tpr = np.r_[0.0, tp_cum[group_ends] / n_pos]
    fpr = np.r_[0.0, fp_cum[group_ends] / n_neg]
    thresholds = np.r_[np.inf, sorted_score[group_ends]]
    return fpr, tpr, thresholds


def auc(x, y):
    
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape != y_arr.shape:
        raise ValueError("x and y must be 1D arrays of identical shape")
    if x_arr.size < 2:
        raise ValueError("at least two points are required to compute AUC")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("x and y must contain only finite values")
    order = np.argsort(x_arr, kind="stable")
    return float(np.sum(np.diff(x_arr[order]) * (y_arr[order][1:] + y_arr[order][:-1]) / 2.0))
