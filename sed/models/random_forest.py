import glob
import os
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import CLASSES_NAMES


def _positive_class_thresholds(Y_train: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Return per-class positive-class prevalence in training data.

    Used as the default decision threshold: predict positive when RF probability
    exceeds the training base rate, which accounts for class imbalance without
    relying on the default 0.5 cutoff.
    """
    return {cls: float(y.ravel().mean()) for cls, y in Y_train.items()}


def _apply_threshold(
    proba: np.ndarray, threshold: float
) -> np.ndarray:
    return (proba[:, 1] >= threshold).astype(int)


def _score_row(
    cls: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    **hparams,
) -> dict:
    """Compute all metrics for one class and return as a flat dict."""
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "class": cls,
        **hparams,
        "threshold": round(threshold, 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "binary_f1": round(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba[:, 1]), 4) if has_both_classes else float("nan"),
        "pr_auc": round(average_precision_score(y_true, proba[:, 1]), 4) if has_both_classes else float("nan"),
    }


def train_rf_binary_per_class(
    X_train: np.ndarray,
    X_val: np.ndarray,
    Y_train: Dict[str, np.ndarray],
    Y_val: Dict[str, np.ndarray],
    class_names: List[str],
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    thresholds: Optional[Dict[str, float]] = None,
    models_dir: str = "rf_models",
    metrics_dir: str = "rf_metrics",
    experiment_log: str = "rf_experiment_log.csv",
) -> tuple[dict[str, RandomForestClassifier], pd.DataFrame]:
    """Train one binary RandomForestClassifier per class on raw features.

    No preprocessing is applied; X_train and X_val are used as-is.
    Decision thresholds default to per-class positive-class prevalence in the
    training set (majority-class-aware baseline). Pass a custom ``thresholds``
    dict to override per class.

    Fixed hyperparameters across all runs:
        class_weight = 'balanced'
        max_features = 'sqrt'
        n_jobs       = -1
        random_state = 42

    Returns
    -------
    models     : dict mapping class name to fitted RandomForestClassifier
    metrics_df : DataFrame with per-class validation metrics
    """
    md_tag = "None" if max_depth is None else str(max_depth)
    run_tag = f"n{n_estimators}_md{md_tag}_msl{min_samples_leaf}"
    run_dir = os.path.join(models_dir, f"rf_{run_tag}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_tag}_{timestamp}"

    default_thresholds = _positive_class_thresholds(Y_train)
    if thresholds is None:
        thresholds = default_thresholds

    models: dict[str, RandomForestClassifier] = {}
    rows = []

    for cls in class_names:
        y_tr = Y_train[cls].ravel()
        y_v = Y_val[cls].ravel()

        if len(np.unique(y_tr)) < 2:
            print(f"[{cls}] skipped — only one label present in training data.")
            continue

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_train, y_tr)
        models[cls] = clf

        model_path = os.path.join(run_dir, f"rf_{cls}_{run_id}.joblib")
        joblib.dump(clf, model_path)

        thr = thresholds.get(cls, default_thresholds[cls])

        val_proba = clf.predict_proba(X_val)
        val_pred = _apply_threshold(val_proba, thr)

        row = _score_row(
            cls, y_v, val_pred, val_proba, thr,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        rows.append(row)
        print(
            f"[{cls}] saved → {model_path} | "
            f"bal_acc={row['balanced_accuracy']:.4f}  "
            f"f1={row['binary_f1']:.4f}  "
            f"prec={row['precision']:.4f}  "
            f"rec={row['recall']:.4f}  "
            f"roc_auc={row['roc_auc']:.4f}  "
            f"pr_auc={row['pr_auc']:.4f}  "
            f"thr={thr:.4f}"
        )

    metrics_df = pd.DataFrame(rows).set_index("class").sort_index()

    metrics_path = os.path.join(metrics_dir, f"rf_val_metrics_{run_id}.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\nPer-class validation metrics saved → {metrics_path}")
    print(metrics_df)

    numeric_cols = ["balanced_accuracy", "binary_f1", "precision", "recall", "roc_auc", "pr_auc"]
    avg_row = {
        "timestamp": timestamp,
        "run_id": run_id,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        **{f"avg_val_{col}": round(metrics_df[col].mean(), 4) for col in numeric_cols},
        "n_classes_trained": len(metrics_df),
        "metrics_file": metrics_path,
    }

    write_header = not os.path.exists(experiment_log)
    pd.DataFrame([avg_row]).to_csv(
        experiment_log, mode="a", header=write_header, index=False
    )
    print(f"Experiment log updated → {experiment_log}")
    print(pd.DataFrame([avg_row]).to_string(index=False))

    return models, metrics_df


def evaluate_rf_on_test(
    X_test: np.ndarray,
    Y_test: Dict[str, np.ndarray],
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_leaf: int,
    thresholds: Optional[Dict[str, float]] = None,
    models_dir: str = "rf_models",
    metrics_dir: str = "rf_metrics",
) -> pd.DataFrame:
    """Load saved RF models and evaluate them on the held-out test set.

    No preprocessing is applied; X_test is used as-is.
    Pass ``thresholds`` to override per-class decision thresholds
    (defaults to 0.5 if not supplied, since training-set prevalence is unavailable here).

    Returns
    -------
    metrics_df : DataFrame with per-class test metrics
    """
    md_tag = "None" if max_depth is None else str(max_depth)
    run_tag = f"n{n_estimators}_md{md_tag}_msl{min_samples_leaf}"
    run_dir = os.path.join(models_dir, f"rf_{run_tag}")

    if thresholds is None:
        thresholds = {}

    rows = []
    for cls in CLASSES_NAMES:
        pattern = os.path.join(run_dir, f"rf_{cls}_{run_tag}_*.joblib")
        matches = glob.glob(pattern)

        if not matches:
            print(f"[{cls}] no saved model found at: {pattern}")
            continue

        clf = joblib.load(sorted(matches)[-1])

        y_te = Y_test[cls].ravel()
        thr = thresholds.get(cls, 0.5)
        proba = clf.predict_proba(X_test)
        pred = _apply_threshold(proba, thr)

        row = _score_row(
            cls, y_te, pred, proba, thr,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        rows.append(row)
        print(
            f"[{cls}] "
            f"bal_acc={row['balanced_accuracy']:.4f}  "
            f"f1={row['binary_f1']:.4f}  "
            f"prec={row['precision']:.4f}  "
            f"rec={row['recall']:.4f}  "
            f"roc_auc={row['roc_auc']:.4f}  "
            f"pr_auc={row['pr_auc']:.4f}  "
            f"thr={thr:.4f}"
        )

    metrics_df = pd.DataFrame(rows).set_index("class").sort_index()

    numeric_cols = ["balanced_accuracy", "binary_f1", "precision", "recall", "roc_auc", "pr_auc"]
    print("\n── Mean over classes ─────────────────────────────────────────────")
    print(metrics_df[numeric_cols].mean().round(4))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        metrics_dir, f"rf_test_metrics_{run_tag}_{ts}.csv"
    )
    metrics_df.to_csv(output_path)
    print(f"\nTest metrics saved → {output_path}")

    return metrics_df
