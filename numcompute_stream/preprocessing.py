"""Incremental NumPy preprocessing transformers."""
from __future__ import annotations
import numpy as np


def _as_2d(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1: a = a.reshape(-1, 1)
    if a.ndim != 2 or a.shape[0] == 0: raise ValueError("X must be a non-empty 2D numeric array")
    return a


class StandardScaler:
    """Column-wise online standardisation using mergeable Welford statistics."""
    def __init__(self): self.count_ = self.mean_ = self.M2_ = None
    def partial_fit(self, X):
        a = _as_2d(X)
        if self.mean_ is None:
            d = a.shape[1]; self.count_ = np.zeros(d); self.mean_ = np.zeros(d); self.M2_ = np.zeros(d)
        if a.shape[1] != len(self.mean_): raise ValueError("Feature count changed between chunks")
        valid = np.isfinite(a); n = valid.sum(axis=0).astype(float)
        sums = np.where(valid, a, 0.0).sum(axis=0)
        means = np.divide(sums, n, out=np.zeros_like(sums), where=n > 0)
        centered = np.where(valid, a - means, 0.0)
        m2 = np.sum(centered * centered, axis=0)
        total = self.count_ + n; delta = means - self.mean_
        self.mean_ += np.divide(delta * n, total, out=np.zeros_like(delta), where=total > 0)
        self.M2_ += m2 + np.divide(delta * delta * self.count_ * n, total, out=np.zeros_like(delta), where=total > 0)
        self.count_ = total
        return self
    def fit(self, X):
        self.count_ = self.mean_ = self.M2_ = None
        return self.partial_fit(X)
    @property
    def var_(self):
        return np.divide(self.M2_, self.count_, out=np.zeros_like(self.M2_), where=self.count_ > 0)
    @property
    def scale_(self):
        s = np.sqrt(self.var_); return np.where(s > 0, s, 1.0)
    def transform(self, X):
        if self.mean_ is None: raise RuntimeError("Scaler has not been fitted")
        a = _as_2d(X)
        if a.shape[1] != len(self.mean_): raise ValueError("Feature count mismatch")
        return (a - self.mean_) / self.scale_
    def fit_transform(self, X): return self.fit(X).transform(X)


class Imputer:
    """Online mean imputer; NaN and infinities are treated as missing."""
    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in ("mean", "constant"): raise ValueError("strategy must be 'mean' or 'constant'")
        self.strategy, self.fill_value = strategy, float(fill_value)
        self.count_ = self.sum_ = None
    def partial_fit(self, X):
        a = _as_2d(X)
        if self.count_ is None: self.count_ = np.zeros(a.shape[1]); self.sum_ = np.zeros(a.shape[1])
        if a.shape[1] != len(self.count_): raise ValueError("Feature count changed between chunks")
        mask = np.isfinite(a); self.count_ += mask.sum(axis=0); self.sum_ += np.where(mask, a, 0).sum(axis=0)
        return self
    @property
    def statistics_(self):
        return np.divide(self.sum_, self.count_, out=np.full_like(self.sum_, self.fill_value), where=self.count_ > 0)
    def fit(self, X): self.count_ = self.sum_ = None; return self.partial_fit(X)
    def transform(self, X):
        if self.count_ is None: raise RuntimeError("Imputer has not been fitted")
        a = _as_2d(X)
        if a.shape[1] != len(self.count_): raise ValueError("Feature count mismatch")
        return np.where(np.isfinite(a), a, self.statistics_)
    def fit_transform(self, X): return self.fit(X).transform(X)


class OneHotEncoder:
    """Incremental encoder whose output columns expand as new categories arrive."""
    def __init__(self, handle_unknown="ignore"):
        if handle_unknown not in ("ignore", "error"): raise ValueError("handle_unknown must be 'ignore' or 'error'")
        self.handle_unknown = handle_unknown; self.categories_ = None
    def partial_fit(self, X):
        a = np.asarray(X, dtype=object)
        if a.ndim == 1: a = a.reshape(-1, 1)
        if a.ndim != 2 or a.shape[0] == 0: raise ValueError("X must be a non-empty 2D array")
        if self.categories_ is None: self.categories_ = [np.array([], dtype=object) for _ in range(a.shape[1])]
        if a.shape[1] != len(self.categories_): raise ValueError("Feature count changed between chunks")
        for j in range(a.shape[1]):
            vals = np.unique(a[:, j][np.array([v is not None for v in a[:, j]])])
            self.categories_[j] = np.unique(np.concatenate((self.categories_[j], vals)))
        return self
    def fit(self, X): self.categories_ = None; return self.partial_fit(X)
    @property
    def n_features_out_(self): return 0 if self.categories_ is None else sum(len(c) for c in self.categories_)
    def transform(self, X):
        if self.categories_ is None: raise RuntimeError("Encoder has not been fitted")
        a = np.asarray(X, dtype=object)
        if a.ndim == 1: a = a.reshape(-1, 1)
        if a.ndim != 2 or a.shape[1] != len(self.categories_): raise ValueError("X feature count mismatch")
        blocks = []
        for j, cats in enumerate(self.categories_):
            block = a[:, j, None] == cats[None, :]
            known = block.any(axis=1)
            if self.handle_unknown == "error" and not np.all(known): raise ValueError(f"Unknown category in column {j}")
            blocks.append(block.astype(float))
        return np.concatenate(blocks, axis=1) if blocks else np.empty((len(a), 0))
    def fit_transform(self, X): return self.fit(X).transform(X)
