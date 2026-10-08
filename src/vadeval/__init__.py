"""vadeval: evaluation toolkit for video anomaly detection.

Frame-level ranking metrics (ROC-AUC, average precision) and the
operator-facing metrics a surveillance pilot is accepted on:
false alarms per hour / per shift and event-level recall.
"""

from vadeval.alarms import (
    calibrate_threshold,
    event_recall,
    false_alarms,
    false_alarms_per_hour,
    segments,
)
from vadeval.metrics import average_precision, roc_auc, roc_curve
from vadeval.smoothing import binarize, moving_average

__all__ = [
    "average_precision",
    "binarize",
    "calibrate_threshold",
    "event_recall",
    "false_alarms",
    "false_alarms_per_hour",
    "moving_average",
    "roc_auc",
    "roc_curve",
    "segments",
]

__version__ = "0.1.0"
