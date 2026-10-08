"""Turning noisy per-frame scores into stable decisions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vadeval._validation import as_scores


def moving_average(scores: ArrayLike, window: int) -> NDArray[np.float64]:
    """Causal moving average over the last ``window`` frames.

    Causal (uses only past frames) because a live system cannot look ahead.
    The first frames average over the frames available so far.
    """
    s = as_scores(scores)
    if window < 1:
        raise ValueError("window must be >= 1")
    csum = np.cumsum(np.r_[0.0, s])
    idx = np.arange(1, s.size + 1)
    start = np.maximum(idx - window, 0)
    return (csum[idx] - csum[start]) / (idx - start)


def binarize(scores: ArrayLike, threshold: float, min_frames: int = 1) -> NDArray[np.bool_]:
    """Frames with ``score >= threshold``, keeping only runs of >= ``min_frames``.

    Short runs are the main source of flicker alarms; dropping them trades a
    little latency for far fewer false alarms.
    """
    s = as_scores(scores)
    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    mask = s >= threshold
    if min_frames == 1:
        return mask
    out = np.zeros_like(mask)
    for start, end in _runs(mask):
        if end - start >= min_frames:
            out[start:end] = True
    return out


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Half-open [start, end) index pairs of consecutive True values."""
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]
