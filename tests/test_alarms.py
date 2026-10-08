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


def test_false_alarms_with_and_without_tolerance():
    truth = [0, 0, 1, 1, 0, 0, 0, 0]
    pred = [1, 0, 0, 1, 1, 0, 1, 1]
    # tolerance 0: frame 0, the late tail at frame 4 and frames 6-7 are all outside the event
    assert false_alarms(pred, truth) == 3
    # tolerance 1 forgives the tail at frame 4 (and frame 0? no: it is 2 frames away)
    assert false_alarms(pred, truth, tolerance=1) == 2


def test_always_alarm_detector_is_penalised():
    truth = np.zeros(1000, dtype=int)
    truth[100:150] = 1
    truth[600:650] = 1
    always = np.ones(1000, dtype=int)
    # regression test: an earlier version counted this as 0 false alarms
    assert false_alarms(always, truth, tolerance=10) == 3


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        false_alarms([1, 0], [0, 1], tolerance=-1)


def test_event_recall():
    truth = [1, 1, 0, 0, 1, 1, 0]
    assert event_recall([0, 1, 0, 0, 0, 0, 0], truth) == 0.5
    with pytest.raises(ValueError, match="no events"):
        event_recall([1, 0], [0, 0])


def test_false_alarms_per_hour():
    assert false_alarms_per_hour(3, 90_000, 25.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        false_alarms_per_hour(1, 0, 25.0)


def test_smoothing_makes_the_alarm_budget_reachable():
    scores, labels = synthetic_stream(n_frames=45_000, seed=3)
    # raw per-frame scores flicker: no threshold keeps false alarms within 2 per shift
    with pytest.raises(ValueError, match="budget"):
        calibrate_threshold(scores, labels, fps=25, max_false_alarms_per_shift=2.0, tolerance=50)
    # after causal smoothing and a minimum alarm length, the budget is met with high recall
    cal = calibrate_threshold(
        moving_average(scores, 12),
        labels,
        fps=25,
        max_false_alarms_per_shift=2.0,
        min_frames=12,
        tolerance=50,
    )
    assert cal.false_alarms_per_shift <= 2.0
    assert cal.event_recall >= 0.9


def test_calibration_fails_loudly_when_budget_impossible():
    scores = np.tile([5.0, 0.0], 50)  # every high frame is a separate false alarm
    labels = np.zeros(100, dtype=int)
    labels[0] = 1
    with pytest.raises(ValueError, match="budget"):
        calibrate_threshold(scores, labels, fps=25, max_false_alarms_per_shift=0.0, n_candidates=5)
