"""Command-line interface: ``vadeval evaluate`` and ``vadeval synth``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from vadeval import __version__
from vadeval.alarms import (
    SHIFT_HOURS,
    calibrate_threshold,
    event_recall,
    false_alarms,
    false_alarms_per_hour,
)
from vadeval.io import load_csv, save_csv
from vadeval.metrics import average_precision, roc_auc
from vadeval.smoothing import binarize, moving_average
from vadeval.synthetic import synthetic_stream


def _evaluate(args: argparse.Namespace) -> dict:
    scores, labels = load_csv(args.scores)
    smoothed = moving_average(scores, args.window)
    tolerance = round(args.tolerance_s * args.fps)
    cal = calibrate_threshold(
        smoothed, labels, args.fps, args.max_fa_per_shift, min_frames=args.min_frames, tolerance=tolerance
    )
    pred = binarize(smoothed, cal.threshold, args.min_frames)
    n_fa = false_alarms(pred, labels, tolerance)
    report = {
        "frames": int(scores.size),
        "hours": round(scores.size / args.fps / 3600, 3),
        "roc_auc_raw": round(roc_auc(labels, scores), 4),
        "roc_auc_smoothed": round(roc_auc(labels, smoothed), 4),
        "average_precision_smoothed": round(average_precision(labels, smoothed), 4),
        "threshold": round(cal.threshold, 4),
        "false_alarms": n_fa,
        "false_alarms_per_shift": round(false_alarms_per_hour(n_fa, scores.size, args.fps) * SHIFT_HOURS, 2),
        "event_recall": round(event_recall(pred, labels), 3),
    }
    if args.plot:
        from vadeval.plot import report_figure

        report_figure(scores, labels, smoothed, cal.threshold, args.plot)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vadeval", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("evaluate", help="evaluate a score file (CSV with score,label)")
    e.add_argument("scores", help="path to CSV file")
    e.add_argument("--fps", type=float, default=25.0, help="frames per second (default 25)")
    e.add_argument("--window", type=int, default=12, help="smoothing window in frames (default 12)")
    e.add_argument("--min-frames", type=int, default=12, help="minimum alarm length in frames")
    e.add_argument("--max-fa-per-shift", type=float, default=2.0, help="false-alarm budget per 8 h")
    e.add_argument(
        "--tolerance-s", type=float, default=2.0, help="seconds around an event not counted as false alarm"
    )
    e.add_argument("--plot", help="optional path for a PNG report figure")

    s = sub.add_parser("synth", help="write a synthetic score file for testing")
    s.add_argument("output", help="path to CSV file to create")
    s.add_argument("--frames", type=int, default=90_000)
    s.add_argument("--events", type=int, default=12)
    s.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            print(json.dumps(_evaluate(args), indent=2))
        else:
            scores, labels = synthetic_stream(args.frames, args.events, seed=args.seed)
            save_csv(args.output, scores, labels)
            print(f"wrote {args.frames} frames to {args.output}")
    except (OSError, ValueError) as exc:
        print(f"vadeval: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
