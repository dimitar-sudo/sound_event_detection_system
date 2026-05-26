import glob
import os
from datetime import datetime
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ..config import CLASSES_NAMES
from ..features.selection import (
    apply_mi_selector,
    apply_scaler,
    apply_variance_selector,
    create_top_k_masks,
)


def train_knn_binary_per_class(
    X_train_knn: Dict[str, np.ndarray],
    X_val_knn: Dict[str, np.ndarray],
    Y_train: Dict[str, np.ndarray],
    Y_val: Dict[str, np.ndarray],
    class_names: List[str],
    n_neighbors: int = 5,
    models_dir: str = "knn_models",
    metrics_dir: str = "knn_metrics",
    experiment_log: str = "knn_experiment_log.csv",
) -> tuple[dict[str, KNeighborsClassifier], pd.DataFrame]:
    """Train one distance-weighted KNN binary classifier per class.

    Models and per-run metrics are persisted to disk. Summary statistics are
    appended to the experiment log CSV for hyperparameter comparison.

    Returns
    -------
    models      : dict mapping class name to fitted KNeighborsClassifier
    metrics_df  : DataFrame with per-class validation balanced accuracy and F1
    """
    k_features = next(iter(X_train_knn.values())).shape[1]
    run_dir = os.path.join(models_dir, f"knn_k{k_features}_n{n_neighbors}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"k{k_features}_n{n_neighbors}_{timestamp}"

    models: dict[str, KNeighborsClassifier] = {}
    rows = []

    for cls in class_names:
        X_tr = X_train_knn[cls]
        X_v = X_val_knn[cls]
        y_tr = Y_train[cls].ravel()
        y_v = Y_val[cls].ravel()

        if len(np.unique(y_tr)) < 2:
            print(f"[{cls}] skipped (only one label in training).")
            continue

        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="minkowski",
        )
        clf.fit(X_tr, y_tr)
        models[cls] = clf

        model_path = os.path.join(run_dir, f"knn_{cls}_{run_id}.joblib")
        joblib.dump(clf, model_path)

        y_tr_pred = clf.predict(X_tr)
        test_bal_acc = balanced_accuracy_score(y_tr, y_tr_pred)
        test_macro_f1 = f1_score(y_tr, y_tr_pred)

        y_v_pred = clf.predict(X_v)
        val_bal_acc = balanced_accuracy_score(y_v, y_v_pred)
        val_macro_f1 = f1_score(y_v, y_v_pred, average="macro")

        rows.append(
            {
                "class": cls,
                "k_features": X_tr.shape[1],
                "n_neighbours": n_neighbors,
                "val_balanced_accuracy": round(val_bal_acc, 4),
                "val_macro_f1": round(val_macro_f1, 4),
            }
        )
        print(f"[{cls}] fitted and saved → {model_path}")

    metrics_df = pd.DataFrame(rows).set_index("class").sort_index()

    metrics_path = os.path.join(metrics_dir, f"knn_metrics_{run_id}.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\nPer-class metrics saved → {metrics_path}")
    print(metrics_df)

    avg_row = {
        "timestamp": timestamp,
        "run_id": run_id,
        "k_features": k_features,
        "n_neighbours": n_neighbors,
        "avg_val_balanced_accuracy": round(
            metrics_df["val_balanced_accuracy"].mean(), 4
        ),
        "avg_val_macro_f1": round(metrics_df["val_macro_f1"].mean(), 4),
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


def evaluate_knn_on_test(
    X_test: np.ndarray,
    Y_test: Dict[str, np.ndarray],
    vt_selector: VarianceThreshold,
    scaler: StandardScaler,
    class_mi_scores: Dict[str, np.ndarray],
    k_features: int,
    n_neighbours: int,
    models_dir: str = "knn_models",
    metrics_dir: str = "knn_metrics",
) -> pd.DataFrame:
    """Load saved KNN models and evaluate them on the held-out test set.

    Applies the same preprocessing pipeline used during training:
    VarianceThreshold → StandardScaler → per-class MI top-k mask.

    Returns
    -------
    metrics_df : DataFrame with per-class test balanced accuracy and macro F1
    """
    X_test_vt = apply_variance_selector(X_test, vt_selector)
    X_test_scaled = apply_scaler(scaler, X_test_vt)[0]
    class_masks = create_top_k_masks(class_mi_scores, k_features)

    run_dir = os.path.join(models_dir, f"knn_k{k_features}_n{n_neighbours}")
    rows = []

    for cls in CLASSES_NAMES:
        pattern = os.path.join(
            run_dir, f"knn_{cls}_k{k_features}_n{n_neighbours}_*.joblib"
        )
        matches = glob.glob(pattern)

        if not matches:
            print(f"[{cls}] No saved model found matching: {pattern}")
            continue

        clf = joblib.load(sorted(matches)[-1])

        X_te_cls = apply_mi_selector(X_test_scaled, class_masks, cls)
        y_te = Y_test[cls].ravel()
        y_te_pred = clf.predict(X_te_cls)

        test_bal_acc = balanced_accuracy_score(y_te, y_te_pred)
        test_macro_f1 = f1_score(y_te, y_te_pred, average="macro")

        rows.append(
            {
                "class": cls,
                "k_features": k_features,
                "n_neighbours": n_neighbours,
                "test_balanced_accuracy": round(test_bal_acc, 4),
                "test_macro_f1": round(test_macro_f1, 4),
                "model_file": os.path.basename(sorted(matches)[-1]),
            }
        )
        print(f"[{cls}] bal_acc={test_bal_acc:.4f}  macro_f1={test_macro_f1:.4f}")

    metrics_df = pd.DataFrame(rows).set_index("class").sort_index()

    print("\n── Mean over classes ─────────────────────────────────")
    print(metrics_df[["test_balanced_accuracy", "test_macro_f1"]].mean())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        metrics_dir, f"knn_TEST_metrics_k{k_features}_n{n_neighbours}_{ts}.csv"
    )
    metrics_df.to_csv(output_path)
    print(f"\nTest metrics saved → {output_path}")

    return metrics_df
