"""A NumPy-only classification decision tree with buffered streaming retraining."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array (n_samples, n_features)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D and contain one label per sample")
    if y.dtype.kind in "fc" and np.any(~np.isfinite(y)):
        raise ValueError("y must not contain NaN or infinite labels")
    if y.dtype.kind == "O":
        raise TypeError("y labels must be numeric or strings with a consistent dtype")
    return X, y


def _impurity(y, criterion):
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts.astype(float) / counts.sum()
    if criterion == "gini":
        return float(1.0 - np.dot(p, p))
    positive = p > 0
    return float(-np.sum(p[positive] * np.log2(p[positive])))


def _majority(y):
    labels, counts = np.unique(y, return_counts=True)
    # np.unique sorts labels; argmax therefore resolves ties to the smallest label.
    return labels[int(np.argmax(counts))]


@dataclass
class _Node:
    prediction: object
    feature: int | None = None
    threshold: float | None = None
    nan_left: bool = True
    left: "_Node | None" = None
    right: "_Node | None" = None
    gain: float = 0.0
    proba: np.ndarray | None = None

    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTreeClassifier:
    """Depth-limited CART-style classifier supporting buffered chunk updates.

    Parameters use numeric features with shape ``(n_samples, n_features)``.
    NaNs are routed to whichever child gives lower weighted impurity; exact ties
    route left. Candidate thresholds are midpoints between sorted unique values.
    """
    def __init__(self, max_depth=5, criterion="gini", min_samples_split=2,
                 max_features=None, buffer_size=500, random_state=42,
                 min_impurity_decrease=1e-12):
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if max_features is not None and (not isinstance(max_features, int) or max_features < 1):
            raise ValueError("max_features must be a positive integer or None")
        if not isinstance(buffer_size, int) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer")
        self.max_depth, self.criterion = max_depth, criterion
        self.min_samples_split, self.max_features = min_samples_split, max_features
        self.buffer_size, self.random_state = buffer_size, random_state
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self._buffer_X, self._buffer_y = [], []
        self.retrain_count_ = 0
        self.n_seen_ = 0
        self._rng = np.random.default_rng(random_state)

    @property
    def buffered_samples_(self):
        return int(sum(a.shape[0] for a in self._buffer_X))

    def fit(self, X, y):
        X, y = _xy(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.root_ = self._build(X, y, depth=0)
        self._buffer_X.clear(); self._buffer_y.clear()
        self.retrain_count_ += 1
        self.n_seen_ += len(y)
        return self

    def _best_split(self, X, y):
        parent = _impurity(y, self.criterion)
        best = None
        features = np.arange(X.shape[1])
        if self.max_features is not None and self.max_features < len(features):
            features = np.sort(self._rng.choice(features, self.max_features, replace=False))
        n = len(y)
        for feature in features:
            col = X[:, feature]
            finite = np.isfinite(col)
            vals = np.unique(col[finite])
            if vals.size < 2:
                continue
            thresholds = vals[:-1] * 0.5 + vals[1:] * 0.5
            for threshold in thresholds:
                base_left = finite & (col <= threshold)
                base_right = finite & (col > threshold)
                missing = ~finite
                for nan_left in (True, False):
                    left = base_left | (missing if nan_left else False)
                    right = base_right | (missing if not nan_left else False)
                    nl, nr = int(left.sum()), int(right.sum())
                    if nl == 0 or nr == 0:
                        continue
                    weighted = (nl * _impurity(y[left], self.criterion) +
                                nr * _impurity(y[right], self.criterion)) / n
                    gain = parent - weighted
                    candidate = (gain, int(feature), float(threshold), nan_left, left)
                    # Deterministic ties: lower feature, lower threshold, NaNs-left.
                    if best is None or gain > best[0] + 1e-15:
                        best = candidate
        return best

    def _build(self, X, y, depth):
        prediction = _majority(y)
        counts = np.asarray([np.sum(y == label) for label in self.classes_], dtype=float)
        node = _Node(prediction=prediction, proba=counts / counts.sum())
        pure = np.all(y == y[0])
        depth_stop = self.max_depth is not None and depth >= self.max_depth
        if pure or depth_stop or len(y) < self.min_samples_split:
            return node
        split = self._best_split(X, y)
        if split is None or split[0] <= self.min_impurity_decrease:
            return node
        gain, feature, threshold, nan_left, left_mask = split
        if left_mask.all() or (~left_mask).all():
            return node
        node.feature, node.threshold, node.nan_left, node.gain = feature, threshold, nan_left, gain
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def partial_fit(self, X_chunk, y_chunk):
        X, y = _xy(X_chunk, y_chunk)
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        if self._buffer_X and X.shape[1] != self._buffer_X[0].shape[1]:
            raise ValueError("All buffered chunks must have the same feature count")
        self._buffer_X.append(X.copy()); self._buffer_y.append(y.copy())
        self.n_seen_ += len(y)
        if self.buffered_samples_ >= self.buffer_size:
            bx = np.concatenate(self._buffer_X, axis=0)
            by = np.concatenate(self._buffer_y, axis=0)
            self.n_features_in_ = bx.shape[1]
            self.classes_ = np.unique(by)
            self.root_ = self._build(bx, by, 0)
            self._buffer_X.clear(); self._buffer_y.clear()
            self.retrain_count_ += 1
        return self

    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier is not trained; call fit or wait for buffer_size samples")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        out = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i, row in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                value = row[node.feature]
                go_left = node.nan_left if not np.isfinite(value) else value <= node.threshold
                node = node.left if go_left else node.right
            out[i] = node.prediction
        return out

    def predict_proba(self, X):
        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier is not trained")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        probabilities = np.empty((len(X), len(self.classes_)), dtype=float)
        for i, row in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                value = row[node.feature]
                go_left = node.nan_left if not np.isfinite(value) else value <= node.threshold
                node = node.left if go_left else node.right
            probabilities[i] = node.proba
        return probabilities

    def score(self, X, y):
        y = np.asarray(y)
        pred = self.predict(X)
        if y.shape != pred.shape:
            raise ValueError("y must match the number of rows in X")
        return float(np.mean(pred == y))

    def reset_stream(self):
        self._buffer_X.clear(); self._buffer_y.clear()
        self.root_ = None; self.classes_ = None; self.n_features_in_ = None
        self.retrain_count_ = 0; self.n_seen_ = 0
        return self
