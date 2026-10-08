import json

import pytest

from vadeval.cli import main


def test_synth_then_evaluate(tmp_path, capsys):
    csv = tmp_path / "scores.csv"
    assert main(["synth", str(csv), "--frames", "20000", "--events", "6", "--seed", "1"]) == 0
    capsys.readouterr()
    assert main(["evaluate", str(csv), "--fps", "25"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frames"] == 20000
    assert 0.5 < report["roc_auc_raw"] <= 1.0
    assert report["roc_auc_smoothed"] >= report["roc_auc_raw"]
    assert report["false_alarms_per_shift"] <= 2.0


def test_plot_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    csv, png = tmp_path / "s.csv", tmp_path / "r.png"
    main(["synth", str(csv), "--frames", "5000", "--events", "3"])
    assert main(["evaluate", str(csv), "--plot", str(png)]) == 0
    assert png.stat().st_size > 10_000


def test_missing_file_returns_error_code(tmp_path, capsys):
    assert main(["evaluate", str(tmp_path / "nope.csv")]) == 1
    assert "error" in capsys.readouterr().err


def test_missing_column(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("score,foo\n0.1,0\n0.2,1\n")
    assert main(["evaluate", str(bad)]) == 1
    assert "label" in capsys.readouterr().err
