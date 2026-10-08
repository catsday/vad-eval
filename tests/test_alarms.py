import numpy as np
import pytest

from vadeval import (
    calibrate_threshold,
    event_recall,
    false_alarms,
    false_alarms_per_hour,
    moving_average,
    segments,
)
from vadeval.synthetic import synthetic_stream


def test_segments():
    assert segments([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    assert segments([0, 0]) == []


def test_false_alarms_counts_only_non_overlapping_segments():
    truth = [0, 0, 1, 1, 0, 0, 0, 0]
    pred = [1, 0, 0, 1, 1, 0, 1, 1]  # [0] and [6-7] miss, [3-4] hits
    assert false_alarms(pred, truth) == 2


def test_event_recall():
    truth = [1, 1, 0, 0, 1, 1, 0]
    assert event_recall([0, 1, 0, 0, 0, 0, 0], truth) == 0.5
    with pytest.raises(ValueError, match="no events"):
        event_recall([1, 0], [0, 0])


def test_false_alarms_per_hour():
    assert false_alarms_per_hour(3, 90_000, 25.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        false_alarms_per_hour(1, 0, 25.0)


def test_calibration_meets_budget():
    scores, labels = synthetic_stream(n_frames=45_000, seed=3)
    cal = calibrate_threshold(moving_average(scores, 12), labels, fps=25, min_frames=12)
    assert cal.false_alarms_per_shift <= 2.0
