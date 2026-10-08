# Changelog

## [0.1.0] - 2026-10-07
### Added
- ROC-AUC, ROC curve and average precision (NumPy only, verified against scikit-learn).
- Causal moving-average smoothing and minimum-duration binarisation.
- Event-level metrics: false alarms, false alarms per hour/shift, event recall.
- Threshold calibration under a false-alarm budget.
- `vadeval` command line (`evaluate`, `synth`) and optional report figure.
- CI (lint, test matrix, coverage gate, build) and release workflow.
