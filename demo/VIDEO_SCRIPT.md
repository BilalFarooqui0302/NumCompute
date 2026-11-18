# Five-minute demonstration video plan

The assignment requires a screen recording from the student's own environment. This script provides a concise walkthrough; it is not a recorded video.

## Suggested recording sequence (4–5 minutes)

**0:00–0:30 — Project and constraints**
- Show the repository root and `numcompute_stream/` package.
- State that the extension uses Python, NumPy and matplotlib only; the decision tree is implemented from scratch.

**0:30–1:20 — Buffered decision tree**
- Open `tree.py` and point to `DecisionTreeClassifier` and `partial_fit`.
- Explain that incoming chunks are appended to the buffer, the tree retrains once the sample count reaches `buffer_size`, and the buffer is cleared.
- Explain that predictions between retrains use the most recently trained tree.

**1:20–2:20 — Run the notebook**
- Open `demo/stream_demo.ipynb`.
- Run the notebook from top to bottom.
- Show CSV loading through `numcompute_stream.io`, the 600-row training stream split into ten 60-row chunks, and the 120-row holdout set.
- Show the pipeline and `StreamTrainer` logs, including retraining count and buffer size.

**2:20–3:20 — Visualisations and predictions**
- Show prequential accuracy over chunks.
- Show predictions against ground-truth labels for the latest holdout sample subset.
- Show the comparison between buffer sizes 80 and 160 and discuss the trade-off between retraining frequency and adaptation.

**3:20–4:10 — Testing evidence**
- Run `python -m unittest discover -s tests -v`.
- Show the final summary: 55 tests passed.
- Mention edge cases covered: NaNs, ties, constant features, invalid shapes, buffer overflow and incremental encoder schema growth.

**4:10–5:00 — Benchmarks and conclusion**
- Run `python benchmark/run_benchmarks.py` or show `benchmark/results.md`.
- Explain the measured loop-versus-NumPy variance comparison and buffer-size results.
- Close by stating the limitations: exact quantiles retain observations, expanding categories reset the downstream tree, and benchmark results are dataset/hardware dependent.

## Recording checklist

- Record the terminal and notebook clearly; use a readable font size.
- Ensure the notebook outputs are visible before recording.
- Do not claim the GitHub fork or branch has been pushed until you have completed that action yourself.
- Save the screen recording as MP4 or upload it to an approved shareable location.
