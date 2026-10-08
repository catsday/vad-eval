"""Frame-level ranking metrics.

Implemented with NumPy only, so the package has a single runtime
dependency. Results are checked against scikit-learn in the test suite.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vadeval._validation import as_labels, as_scores, require_both_classes


def _average_ranks(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """1-based ranks; tied values receive the mean of their ranks."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    # boundaries of runs of equal values
    boundaries = np.flatnonzero(np.diff(sorted_x)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [x.size]))
    for s, e in zip(starts, ends, strict=True):
        ranks[order[s:e]] = (s + 1 + e) / 2.0
    return ranks


def roc_auc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Area under the ROC curve (Mann-Whitney U formulation).

    Equals the probability that a random abnormal frame scores higher
    than a random normal frame; ties count as one half.
    """
    s = as_scores(scores)
    y = as_labels(labels, s.size)
    require_both_classes(y)
    ranks = _average_ranks(s)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(
    labels: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """False-positive rate, true-positive rate and the thresholds producing them.

    Thresholds are the distinct score values in decreasing order; a frame is
    predicted abnormal when ``score >= threshold``. The curve starts at (0, 0).
    """
    s = as_scores(scores)
    y = as_labels(labels, s.size)
    require_both_classes(y)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]
    tps = np.cumsum(y_sorted)[last_of_group]
    fps = (last_of_group + 1) - tps
    tpr = np.r_[0.0, tps / y.sum()]
    fpr = np.r_[0.0, fps / (~y).sum()]
    thresholds = np.r_[np.inf, s_sorted[last_of_group]]
    return fpr, tpr, thresholds


def average_precision(labels: ArrayLike, scores: ArrayLike) -> float:
    """Average precision: sum over thresholds of (R_n - R_{n-1}) * P_n."""
    s = as_scores(scores)
    y = as_labels(labels, s.size)
    require_both_classes(y)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]
    tps = np.cumsum(y_sorted)[last_of_group].astype(np.float64)
    predicted = (last_of_group + 1).astype(np.float64)
    precision = tps / predicted
    recall = tps / y.sum()
    recall_steps = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_steps * precision))
