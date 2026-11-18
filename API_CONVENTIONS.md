# API conventions

- Features are two-dimensional `(n_samples, n_features)` arrays; targets and predictions are one-dimensional `(n_samples,)` arrays.
- Public update methods return `self` when they mutate estimator state; plotting functions return a matplotlib Figure.
- `fit` resets learned state; `partial_fit` merges or buffers new observations.
- Empty chunks and mismatched dimensions raise `ValueError`; prediction before model training raises `RuntimeError`.
- Non-finite feature values are missing values. Labels and metric scores must be valid finite values.
- Class and split ties are deterministic to support reproducible tests.
