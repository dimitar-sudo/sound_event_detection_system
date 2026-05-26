import json
from typing import Dict, List

import joblib
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from ..config import CLASSES_NAMES


# ── Scaler ────────────────────────────────────────────────────────────────────

def fit_and_save_scaler(
    X_train: np.ndarray,
    path: str = "sed/data/scaler.joblib",
) -> StandardScaler:
    """Fit a StandardScaler on training data only and persist it to disk."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, path)
    print(f"Scaler fitted and saved to '{path}'")
    return scaler


def apply_scaler(
    scaler: StandardScaler,
    *arrays: np.ndarray,
) -> list[np.ndarray]:
    """Apply a pre-fitted scaler to one or more feature matrices."""
    return [scaler.transform(arr) for arr in arrays]


def load_scaler(path: str = "sed/data/scaler.joblib") -> StandardScaler:
    """Load a persisted scaler from disk for inference. Never refit."""
    return joblib.load(path)


# ── Variance threshold ────────────────────────────────────────────────────────

def fit_and_save_variance_selector(
    X_train: np.ndarray,
    variance_threshold: float = 0.01,
    vt_path: str = "sed/data/variance_selector.joblib",
) -> VarianceThreshold:
    """Fit a VarianceThreshold on training data and persist it to disk."""
    vt_selector = VarianceThreshold(threshold=variance_threshold)
    X_vt = vt_selector.fit_transform(X_train)
    joblib.dump(vt_selector, vt_path)
    print(f"VarianceThreshold: {X_train.shape[1]} → {X_vt.shape[1]} features")
    return vt_selector


def apply_variance_selector(
    X: np.ndarray,
    vt_selector: VarianceThreshold,
) -> np.ndarray:
    """Apply a pre-fitted VarianceThreshold selector."""
    return vt_selector.transform(X)


# ── Mutual information selector ───────────────────────────────────────────────

def fit_and_save_mi_selector(
    X_train_scaled: np.ndarray,
    Y_train: Dict[str, np.ndarray],
    class_names: List[str],
    mi_path: str = "sed/data/mi_scores.json",
) -> Dict[str, np.ndarray]:
    """Compute and save per-class mutual information scores.

    Scores are computed on training data only.
    """
    class_mi_scores: Dict[str, np.ndarray] = {}

    for class_name in class_names:
        y_cls = Y_train[class_name].ravel()

        if len(np.unique(y_cls)) < 2:
            print(f"[{class_name}] Skipped — only one label in training set")
            class_mi_scores[class_name] = np.zeros(X_train_scaled.shape[1])
            continue

        mi_scores = mutual_info_classif(X_train_scaled, y_cls, random_state=42)
        class_mi_scores[class_name] = mi_scores
        print(f"[{class_name}] Max MI: {mi_scores.max():.4f}")

    serializable = {cls: scores.tolist() for cls, scores in class_mi_scores.items()}
    with open(mi_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"MI scores saved to '{mi_path}'")

    return class_mi_scores


def load_feature_selectors(
    mi_path: str = "sed/data/mi_scores.json",
    vt_path: str = "sed/data/variance_selector.joblib",
) -> tuple[VarianceThreshold, Dict[str, np.ndarray]]:
    """Load persisted feature selectors from disk. Never refit on new data."""
    vt_selector = joblib.load(vt_path)

    with open(mi_path, "r") as f:
        raw = json.load(f)

    class_mi_scores = {
        cls: np.array(scores, dtype=np.float32) for cls, scores in raw.items()
    }

    return vt_selector, class_mi_scores


def create_top_k_masks(
    class_mi_scores: Dict[str, np.ndarray],
    k: int,
) -> Dict[str, np.ndarray]:
    """Create per-class boolean masks selecting the top-k MI features."""
    class_masks: Dict[str, np.ndarray] = {}
    for class_name, scores in class_mi_scores.items():
        top_k_idx = np.argsort(scores)[::-1][:k]
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_k_idx] = True
        class_masks[class_name] = mask
    return class_masks


def apply_mi_selector(
    X: np.ndarray,
    class_masks: Dict[str, np.ndarray],
    class_name: str,
) -> np.ndarray:
    """Apply the pre-fitted MI mask for a given class."""
    return X[:, class_masks[class_name]]


# ── Full KNN preprocessing pipeline ──────────────────────────────────────────

def knn_standardisation(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    vt_selector: VarianceThreshold,
    scaler: StandardScaler,
    class_mi_scores: Dict[str, np.ndarray],
    k: int,
    class_names: List[str],
) -> tuple[dict, dict, dict]:
    """Apply the full feature selection pipeline for KNN: VT → scaler → MI top-k."""
    X_train_vt = apply_variance_selector(X_train, vt_selector)
    X_val_vt = apply_variance_selector(X_val, vt_selector)
    X_test_vt = apply_variance_selector(X_test, vt_selector)

    X_train_scaled, X_val_scaled, X_test_scaled = apply_scaler(
        scaler, X_train_vt, X_val_vt, X_test_vt
    )

    class_masks = create_top_k_masks(class_mi_scores, k)

    X_train_knn: dict[str, np.ndarray] = {}
    X_val_knn: dict[str, np.ndarray] = {}
    X_test_knn: dict[str, np.ndarray] = {}

    for cls in class_names:
        X_train_knn[cls] = apply_mi_selector(X_train_scaled, class_masks, cls)
        X_val_knn[cls] = apply_mi_selector(X_val_scaled, class_masks, cls)
        X_test_knn[cls] = apply_mi_selector(X_test_scaled, class_masks, cls)

    return X_train_knn, X_val_knn, X_test_knn
