# Benchmark results

Environment: Python 3.13.5, NumPy 2.3.5, Linux-6.18.57-x86_64-with-glibc2.41.

## Loop versus vectorised variance

- Loop median: 16.314 ms
- NumPy median: 0.096 ms
- Speedup: 170.78x
- Numerical equivalence checked with `assert_allclose`.

## Streaming buffer-size experiment

Synthetic deterministic binary dataset (720 samples, four numeric features), 40-row chunks, depth 5. Accuracy is prequential: each chunk is predicted before it is added to the training buffer.

| Buffer size | Prequential accuracy | Holdout accuracy | Median update (ms) | Median retrain (ms) | Retrains | Remaining buffer |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.772 | 0.908 | 25.440 | 25.440 | 18 | 0 |
| 80 | 0.842 | 0.850 | 23.537 | 56.115 | 9 | 0 |
| 160 | 0.868 | 0.908 | 0.011 | 147.831 | 4 | 80 |
| 240 | 0.844 | 0.900 | 0.004 | 250.869 | 3 | 0 |

Timings are machine-dependent and are descriptive rather than universal. The final holdout set is independently generated and uses the same synthetic decision boundary.
