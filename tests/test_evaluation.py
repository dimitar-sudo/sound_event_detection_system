import numpy as np
from sklearn.metrics import f1_score

from sed.evaluation import best_f1_thresholds, binarize, per_class_metrics


def test_binarize_uses_absolute_thresholds_inclusive():
    proba = np.array([[0.04, 0.6], [0.05, 0.4]])
    pred = binarize(proba, np.array([0.05, 0.5]))
    assert pred.tolist() == [[0, 1], [1, 0]]


def test_best_f1_threshold_beats_every_other_cutoff():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    p = np.clip(0.3 * y + rng.random(2000) * 0.7, 0, 1)
    thr = best_f1_thresholds(y[:, None], p[:, None])[0]
    best = f1_score(y, p >= thr)
    for t in np.linspace(0, 1, 101):
        assert f1_score(y, p >= t, zero_division=0) <= best + 1e-9


def test_constant_scores_give_chance_level_pr_auc():
    y = np.array([[1], [0], [0], [0]])
    df = per_class_metrics(y, np.full((4, 1), 0.25), np.array([0.25]), ["a"])
    assert df.loc["a", "pr_auc"] == 0.25          # = prevalence
    assert df.loc["a", "recall"] == 1.0
