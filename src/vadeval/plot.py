"""Optional plots (requires the ``plot`` extra: ``pip install vadeval[plot]``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vadeval.metrics import roc_auc, roc_curve


def report_figure(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    smoothed: NDArray[np.float64],
    threshold: float,
    path: str | Path,
) -> None:
    """Save a two-panel figure: score timeline and ROC curve."""
    import matplotlib

    matplotlib.use("Agg")  # headless: works on CI runners without a display
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    t = np.arange(scores.size)
    ax1.plot(t, scores, lw=0.3, color="#9aa5b1", label="raw score")
    ax1.plot(t, smoothed, lw=1.0, color="#127C80", label="smoothed")
    ax1.axhline(threshold, color="#E4572E", lw=1.2, ls="--", label=f"threshold {threshold:.2f}")
    ymin, ymax = ax1.get_ylim()
    ax1.fill_between(
        t, ymin, ymax, where=labels.astype(bool), color="#E4572E", alpha=0.15, label="true event"
    )
    ax1.set_xlabel("frame")
    ax1.set_ylabel("anomaly score")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    fpr, tpr, _ = roc_curve(labels, scores)
    ax2.plot(fpr, tpr, color="#1B2A41", lw=1.8)
    ax2.plot([0, 1], [0, 1], color="#bbbbbb", ls=":")
    ax2.set_title(f"ROC (raw scores), AUC = {roc_auc(labels, scores):.3f}")
    ax2.set_xlabel("false-positive rate")
    ax2.set_ylabel("true-positive rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
