"""Metrics and decision thresholds shared by every model.

Events are rare (1–16% of frames per class), so accuracy-style metrics are
dominated by the "absent" frames. We therefore report, per class:

* **PR-AUC** (average precision) — threshold-free ranking quality; the primary
  model-selection metric. A random scorer gets PR-AUC ≈ class prevalence.
* **F1, precision, recall of the positive class** at a decision threshold.

Thresholds are chosen per class on the validation split (the one maximising
F1) and then applied unchanged to the test split via ``binarize``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_score,
    recall_score,
)

METRICS = ["pr_auc", "f1", "precision", "recall"]


def binarize(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """(N, C) probabilities -> (N, C) predictions; positive when proba >= threshold.

    Thresholds are absolute probabilities, compared directly with no rescaling.
    """
    return (proba >= np.asarray(thresholds)[None, :]).astype(np.uint8)


def best_f1_thresholds(Y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Per class, the probability threshold that maximises F1 on (Y, proba)."""
    thresholds = np.empty(Y.shape[1])
    for c in range(Y.shape[1]):
        precision, recall, thr = precision_recall_curve(Y[:, c], proba[:, c])
        # precision/recall have one more entry than thr (the final (1, 0) point).
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        thresholds[c] = thr[np.argmax(f1)]
    return thresholds


def per_class_metrics(Y: np.ndarray, proba: np.ndarray, thresholds: np.ndarray,
                      class_names: list[str]) -> pd.DataFrame:
    """PR-AUC plus thresholded F1 / precision / recall for every class."""
    pred = binarize(proba, thresholds)
    rows = []
    for c, name in enumerate(class_names):
        y, p = Y[:, c], pred[:, c]
        rows.append({
            "class": name,
            "prevalence": y.mean(),
            "pr_auc": average_precision_score(y, proba[:, c]),
            "f1": f1_score(y, p, zero_division=0),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "threshold": thresholds[c],
        })
    return pd.DataFrame(rows).set_index("class")


def macro(df: pd.DataFrame) -> pd.Series:
    """Unweighted mean over classes (every class counts equally)."""
    return df[["prevalence", *METRICS]].mean()


def pr_auc_per_class(Y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    return np.array([average_precision_score(Y[:, c], proba[:, c]) for c in range(Y.shape[1])])
