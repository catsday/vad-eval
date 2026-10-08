"""Input checks shared by all public functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_scores(scores: ArrayLike) -> NDArray[np.float64]:
    """Return a 1-D float array of finite anomaly scores."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("scores must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must be finite (no NaN or inf)")
    return arr


def as_labels(labels: ArrayLike, n: int | None = None) -> NDArray[np.bool_]:
    """Return a 1-D boolean array; values must be 0/1 or True/False."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {arr.shape}")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    if n is not None and arr.size != n:
        raise ValueError(f"labels length {arr.size} does not match scores length {n}")
    return arr.astype(bool)


def require_both_classes(labels: NDArray[np.bool_]) -> None:
    if labels.all() or not labels.any():
        raise ValueError("labels must contain both normal (0) and abnormal (1) frames")
