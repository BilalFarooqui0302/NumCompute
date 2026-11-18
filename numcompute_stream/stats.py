"""Incremental descriptive statistics for streaming numeric data."""
from __future__ import annotations
import numpy as np

class StreamingStats:
    """Mergeable per-feature count, mean and population variance accumulator.

    Exact quantiles and histograms retain finite observations; this trades bounded
    memory for exact results and is explicitly separate from Welford summaries.
    """
    def __init__(self): self.count_ = self.mean_ = self.M2_ = None; self._values = []
    def update_stats(self, X_chunk):
        a = np.asarray(X_chunk, dtype=float)
        if a.ndim == 1: a = a.reshape(-1, 1)
        if a.ndim != 2 or a.shape[0] == 0: raise ValueError("X_chunk must be non-empty and 2D")
        if self.count_ is None:
            self.count_ = np.zeros(a.shape[1]); self.mean_ = np.zeros(a.shape[1]); self.M2_ = np.zeros(a.shape[1])
        if a.shape[1] != len(self.count_): raise ValueError("Feature count mismatch")
        valid = np.isfinite(a); n = valid.sum(axis=0).astype(float)
        sums = np.where(valid, a, 0).sum(axis=0); m = np.divide(sums, n, out=np.zeros_like(sums), where=n>0)
        m2 = np.sum(np.where(valid, a-m, 0)**2, axis=0); total = self.count_ + n; delta = m-self.mean_
        self.mean_ += np.divide(delta*n, total, out=np.zeros_like(delta), where=total>0)
        self.M2_ += m2 + np.divide(delta**2*self.count_*n, total, out=np.zeros_like(delta), where=total>0)
        self.count_ = total
        self._values.append(a.copy())
        return self
    @property
    def variance_(self): return np.divide(self.M2_, self.count_, out=np.full_like(self.M2_, np.nan), where=self.count_>0)
    @property
    def std_(self): return np.sqrt(self.variance_)
    def result(self):
        if self.count_ is None: raise RuntimeError("No data have been accumulated")
        return {"count": self.count_.copy(), "mean": np.where(self.count_>0,self.mean_,np.nan), "variance":self.variance_.copy(), "std":self.std_.copy()}
    def quantile(self, q):
        if not 0 <= q <= 1: raise ValueError("q must be between 0 and 1")
        if not self._values: raise RuntimeError("No data have been accumulated")
        a=np.concatenate(self._values,axis=0); out=np.full(a.shape[1],np.nan)
        for j in range(a.shape[1]):
            v=a[np.isfinite(a[:,j]),j]
            if v.size: out[j]=np.quantile(v,q)
        return out
    def histogram(self, bins=10, feature=0, range=None):
        if not self._values: raise RuntimeError("No data have been accumulated")
        a=np.concatenate(self._values,axis=0)[:,feature]; a=a[np.isfinite(a)]
        return np.histogram(a,bins=bins,range=range)
    def reset(self): self.__init__(); return self
