"""CSV input helpers implemented with NumPy and the Python standard library."""
from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Iterator, Sequence, Any
import numpy as np

_DEFAULT_MISSING = ("", "NA", "NaN", "nan", "null", "None")


def read_csv(
    path: str | Path,
    delimiter: str = ",",
    *,
    dtype: Any = str,
    missing_values: Sequence[str] = _DEFAULT_MISSING,
    filling_values: Any = "",
    skip_header: int = 1,
    usecols: Sequence[int] | None = None,
    encoding: str = "utf-8",
    max_rows: int | None = None,
) -> np.ndarray:

    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("delimiter must be a single character")
    if skip_header < 0 or (max_rows is not None and max_rows < 0):
        raise ValueError("skip_header and max_rows must be non-negative")
    try:
        return np.genfromtxt(
            path,
            delimiter=delimiter,
            dtype=dtype,
            missing_values=missing_values,
            filling_values=filling_values,
            skip_header=skip_header,
            usecols=usecols,
            encoding=encoding,
            max_rows=max_rows,
            ndmin=2,
            autostrip=True,
            invalid_raise=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse delimited file {path!s}: {exc}") from exc


def read_numeric_csv(
    path: str | Path,
    delimiter: str = ",",
    *,
    skip_header: int = 1,
    missing_values: Sequence[str] = _DEFAULT_MISSING,
    filling_values: float = np.nan,
    usecols: Sequence[int] | None = None,
    max_rows: int | None = None,
) -> np.ndarray:
    
    return read_csv(
        path,
        delimiter,
        dtype=float,
        missing_values=missing_values,
        filling_values=filling_values,
        skip_header=skip_header,
        usecols=usecols,
        max_rows=max_rows,
    )


def iter_csv_chunks(
    path: str | Path,
    chunksize: int,
    delimiter: str = ",",
    *,
    dtype: Any = str,
    missing_values: Sequence[str] = _DEFAULT_MISSING,
    filling_values: Any = "",
    skip_header: int = 1,
    usecols: Sequence[int] | None = None,
    encoding: str = "utf-8",
) -> Iterator[np.ndarray]:
    
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("delimiter must be a single character")
    if skip_header < 0:
        raise ValueError("skip_header must be non-negative")

    with open(path, "r", encoding=encoding, newline="") as handle:
        for _ in range(skip_header):
            next(handle, None)
        while True:
            lines = list(islice(handle, chunksize))
            if not lines:
                break
            chunk = np.genfromtxt(
                lines,
                delimiter=delimiter,
                dtype=dtype,
                missing_values=missing_values,
                filling_values=filling_values,
                usecols=usecols,
                encoding=encoding,
                ndmin=2,
                autostrip=True,
                invalid_raise=True,
            )
            if chunk.size:
                yield chunk
