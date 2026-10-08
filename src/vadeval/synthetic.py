"""Synthetic score streams for demos and tests (no real video needed)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def synthetic_stream(
    n_frames: int = 90_000,
    n_events: int = 12,
    event_len: tuple[int, int] = (50, 250),
    separation: float = 1.2,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Scores and labels for one camera.

    Normal frames score N(0, 1) with slow drift and occasional spikes;
    abnormal frames are shifted up by ``separation``. 90 000 frames is one
    hour at 25 FPS.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(n_frames, dtype=np.int64)
    for _ in range(n_events):
        length = int(rng.integers(*event_len))
        start = int(rng.integers(0, n_frames - length))
        labels[start : start + length] = 1
    drift = np.convolve(rng.normal(0, 0.3, n_frames), np.ones(500) / 500, mode="same")
    noise = rng.normal(0, 1, n_frames)
    spikes = (rng.random(n_frames) < 0.002) * rng.normal(2.5, 0.5, n_frames)
    scores = noise + drift + spikes + separation * labels
    return scores, labels
