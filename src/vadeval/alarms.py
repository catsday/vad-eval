"""Operator-facing metrics: alarms, not frames.

A detector is judged in the field by how many times it wrongly calls the
operator and how many real incidents it catches. Both are counted per
*event* (a contiguous run of positive frames), not per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from vadeval._validation import as_labels, as_scores
from vadeval.smoothing import _runs, binarize

SHIFT_HOURS = 8.0


def segments(mask: ArrayLike) -> list[tuple[int, int]]:
    """Contiguous positive segments of a 0/1 mask as half-open (start, end)."""
    return _runs(as_labels(mask))


def false_alarms(pred: ArrayLike, truth: ArrayLike) -> int:
    """Number of predicted segments that do not overlap any true segment."""
    p = as_labels(pred)
    t = as_labels(truth, p.size)
    return sum(1 for a, b in _runs(p) if not t[a:b].any())


def event_recall(pred: ArrayLike, truth: ArrayLike) -> float:
    """Share of true events overlapped by at least one predicted frame."""
    p = as_labels(pred)
    t = as_labels(truth, p.size)
    events = _runs(t)
    if not events:
        raise ValueError("truth contains no events")
    return sum(1 for a, b in events if p[a:b].any()) / len(events)


def false_alarms_per_hour(n_false_alarms: int, n_frames: int, fps: float) -> float:
    """Normalise a false-alarm count to a rate per hour of video."""
    if n_frames <= 0 or fps <= 0:
        raise ValueError("n_frames and fps must be positive")
    hours = n_frames / fps / 3600.0
    return n_false_alarms / hours


@dataclass(frozen=True)
class Calibration:
    """Result of :func:`calibrate_threshold`."""

    threshold: float
    false_alarms_per_shift: float
    event_recall: float


def calibrate_threshold(
    scores: ArrayLike,
    labels: ArrayLike,
    fps: float,
    max_false_alarms_per_shift: float = 2.0,
    min_frames: int = 1,
    n_candidates: int = 200,
) -> Calibration:
    """Lowest threshold whose false-alarm rate fits the operator budget.

    Lower thresholds catch more events, so among thresholds that meet the
    budget the lowest one maximises recall. Candidates are score quantiles.
    Raises ``ValueError`` if no candidate meets the budget.
    """
    s = as_scores(scores)
    y = as_labels(labels, s.size)
    candidates = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_candidates)))
    for thr in candidates:  # ascending: first feasible = lowest feasible
        pred = binarize(s, float(thr), min_frames)
        rate = false_alarms_per_hour(false_alarms(pred, y), s.size, fps) * SHIFT_HOURS
        if rate <= max_false_alarms_per_shift:
            recall = event_recall(pred, y) if y.any() else float("nan")
            return Calibration(float(thr), float(rate), float(recall))
    raise ValueError("no threshold meets the false-alarm budget; relax it or smooth more")
