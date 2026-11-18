# NumCompute Streaming Decision-Tree Framework

An individual extension of the NumCompute scientific-computing toolkit. The package is implemented with plain Python, NumPy and matplotlib; it does not depend on scikit-learn, pandas or another machine-learning/data-processing library.

## Features

- CART-style classification tree trained from scratch with Gini or entropy impurity.
- Buffered `partial_fit`: chunks accumulate until `buffer_size` is reached, then the tree is rebuilt and the buffer is cleared. Predictions between retraining events use the most recently fitted tree.
- Numeric split thresholds, deterministic majority-class ties, deterministic split tie-breaking and explicit NaN routing.
- Incremental StandardScaler (mergeable Welford statistics), mean/constant Imputer and expanding OneHotEncoder.
- Streaming mean/variance, exact retained-data quantiles and histograms.
- Cumulative and rolling accuracy, macro precision/recall/F1, confusion matrix and optional binary ROC AUC.
- Pipeline, prequential StreamTrainer logging, memory estimates and matplotlib visualisations.

## Install

Python 3.10+, NumPy and matplotlib are required.

```bash
python -m pip install numpy matplotlib
```

Run commands from this directory so `numcompute_stream` is importable.

## Quick start

```python
import numpy as np
from numcompute_stream import DecisionTreeClassifier, StreamTrainer

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([0, 0, 1, 1])
model = DecisionTreeClassifier(max_depth=3, buffer_size=4)
trainer = StreamTrainer(model)
trainer.fit_chunk(X[:2], y[:2])
trainer.fit_chunk(X[2:], y[2:])
print(model.predict(X))
print(trainer.logs)
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Run demo

Open `demo/stream_demo.ipynb` in Jupyter. It loads `demo/data/stream_classification.csv` through the package's NumPy CSV reader, processes sequential chunks, records prequential metrics and displays the built-in plots.

## Run benchmarks

```bash
python benchmark/run_benchmarks.py
```

The script writes `benchmark/results.json` and `benchmark/results.md`. It compares a Python loop with a NumPy vectorised variance calculation and evaluates several tree buffer sizes.

## API and numerical conventions

- Tabular input: `X.shape == (n_samples, n_features)`; labels: `y.shape == (n_samples,)`.
- Features are numeric for the decision tree. Non-finite feature values are treated as missing and routed to the child with lower weighted impurity (ties route left).
- Labels must be finite numeric or string labels. Majority-class ties select the first sorted label.
- `partial_fit` appends whole chunks. If accumulated samples are at least `buffer_size`, all buffered samples are used for one retrain and the buffer is emptied; no overflow remainder is carried forward.
- A pipeline whose incremental encoder expands its output width resets the downstream tree and its buffer to avoid mixing incompatible feature spaces. This is an explicit trade-off: earlier model state is discarded when the feature schema changes.
- Exact quantiles/histograms retain raw observations; Welford mean/variance summaries retain only per-feature state.
- StreamTrainer uses prequential evaluation (predict the chunk before training on it) to avoid reporting optimistic same-chunk training accuracy.

## Repository layout

```text
numcompute_stream/  reusable implementation modules
tests/              unit and edge-case tests
demo/               executed streaming notebook and CSV data
benchmark/          loop/vectorised and buffer-size experiments
```
