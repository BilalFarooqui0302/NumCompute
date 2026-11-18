# Required fork-and-branch workflow

This folder is prepared as the individual-branch working tree. GitHub authentication and repository permissions belong to the student, so no remote fork or push is performed here.

1. On GitHub, fork the original group repository.
2. Clone your fork locally.
3. Keep the original group implementation on `main` unchanged.
4. Create an individual branch, for example:
   `git checkout -b assignment-2.2-streaming-tree`
5. Copy the contents of this prepared folder into the individual branch working tree, preserving the original `numcompute/` package if it already exists. The supplied `numcompute/` directory is a byte-for-byte copy of the original ZIP's package.
6. Commit the individual work and push the branch:
   `git add . && git commit -m "Implement streaming decision tree framework"`
   `git push -u origin assignment-2.2-streaming-tree`
7. Verify the GitHub branch contains `numcompute_stream/`, `tests/`, `demo/stream_demo.ipynb`, `benchmark/`, `README.md` and the report.

Do not merge the individual branch into the protected/original `main` branch unless the course instructions explicitly require it.
