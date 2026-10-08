"""Reading score files produced by a detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load a CSV with a header and columns ``score,label`` (one row per frame)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = data.dtype.names or ()
    missing = {"score", "label"} - set(names)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    return np.atleast_1d(data["score"]).astype(np.float64), np.atleast_1d(data["label"]).astype(np.int64)


def save_csv(path: str | Path, scores: NDArray[np.float64], labels: NDArray[np.int64]) -> None:
    rows = np.column_stack([scores, labels])
    np.savetxt(path, rows, delimiter=",", header="score,label", comments="", fmt=["%.6f", "%d"])
