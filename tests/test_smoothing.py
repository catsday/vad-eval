import numpy as np
import pytest

from vadeval import binarize, moving_average


def test_moving_average_is_causal():
    s = np.array([0.0, 0.0, 0.0, 9.0, 0.0])
    out = moving_average(s, 3)
    # the spike at index 3 must not affect earlier frames
    np.testing.assert_allclose(out[:3], 0.0)
    np.testing.assert_allclose(out[3:], [3.0, 3.0])


def test_moving_average_window_one_is_identity():
    s = np.random.default_rng(0).random(50)
    np.testing.assert_allclose(moving_average(s, 1), s)


def test_moving_average_warm_up_uses_available_frames():
    np.testing.assert_allclose(moving_average([2.0, 4.0, 6.0], 10), [2.0, 3.0, 4.0])


def test_binarize_drops_short_runs():
    s = [0, 1, 0, 1, 1, 1, 0, 1, 1]
    assert binarize(s, 0.5, min_frames=3).tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize(
    "fn, kwargs", [(moving_average, {"window": 0}), (binarize, {"threshold": 0.5, "min_frames": 0})]
)
def test_invalid_parameters(fn, kwargs):
    with pytest.raises(ValueError):
        fn([0.1, 0.2], **kwargs)
