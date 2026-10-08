import numpy as np
import pytest

from vadeval import average_precision, roc_auc, roc_curve


def test_perfect_and_inverted_ranking():
    y = [0, 0, 1, 1]
    assert roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_all_ties_give_half():
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_known_value_with_tie():
    # pairs (pos, neg): (0.8>0.1), (0.8>0.4), (0.4=0.4 -> 0.5), (0.4>0.1) => 3.5 / 4
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.8]) == pytest.approx(0.875)


@pytest.mark.parametrize("seed", range(5))
def test_matches_scikit_learn(seed):
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 2000)
    s = np.round(rng.normal(size=2000) + y, 1)  # rounding creates many ties
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)
    assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-12)
    fpr, tpr, _ = roc_curve(y, s)
    sk_fpr, sk_tpr, _ = sk.roc_curve(y, s, drop_intermediate=False)
    np.testing.assert_allclose(fpr, sk_fpr)
    np.testing.assert_allclose(tpr, sk_tpr)


def test_roc_curve_is_monotone_and_ends_at_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    fpr, tpr, thr = roc_curve(y, rng.random(500))
    assert fpr[0] == tpr[0] == 0.0 and fpr[-1] == tpr[-1] == 1.0
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    assert np.all(np.diff(thr) < 0)


@pytest.mark.parametrize(
    "labels, scores, message",
    [
        ([1, 1], [0.1, 0.2], "both"),
        ([0, 2], [0.1, 0.2], "only 0 and 1"),
        ([0, 1], [0.1, np.nan], "finite"),
        ([0, 1, 1], [0.1, 0.2], "does not match"),
        ([[0, 1]], [[0.1, 0.2]], "1-D"),
    ],
)
def test_invalid_input_is_rejected(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        roc_auc(labels, scores)


def test_empty_and_two_dimensional_labels():
    with pytest.raises(ValueError, match="empty"):
        roc_auc([], [])
    with pytest.raises(ValueError, match="labels must be 1-D"):
        roc_auc([[0, 1]], [0.1, 0.2])
