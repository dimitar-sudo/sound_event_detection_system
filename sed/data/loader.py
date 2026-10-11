"""Reading pre-extracted features and aggregated frame labels from ``.npz`` files.

Each file holds T frames (1 s windows, 0.5 s hop). Features are the 56 groups in
``config.FEATURE_NAMES`` (960 dimensions in total); ``annotations`` has shape
(T, C, A) with one score per frame, class and annotator.
"""
from typing import Tuple

import numpy as np

from ..config import ANNOTATION_THRESHOLD, FEATURE_NAMES


def aggregate_labels(annotations: np.ndarray) -> np.ndarray:
    """Majority vote over annotators: (T, C, A) scores -> (T, C) binary labels.

    A score above ``ANNOTATION_THRESHOLD`` counts as a positive vote; a frame is
    positive for a class when strictly more than half of the annotators agree.
    """
    votes = (annotations > ANNOTATION_THRESHOLD).sum(axis=2)
    return (votes > annotations.shape[-1] // 2).astype(np.uint8)


def load_features(npz) -> np.ndarray:
    """Stack the configured feature groups into a (T, 960) float32 matrix."""
    cols = [npz[f] if npz[f].ndim > 1 else npz[f][:, None] for f in FEATURE_NAMES]
    return np.hstack(cols).astype(np.float32)


def get_features_and_targets(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(X, Y)`` for one file: (T, 960) features and (T, C) labels."""
    npz = np.load(path, allow_pickle=True)
    return load_features(npz), aggregate_labels(npz["annotations"])
