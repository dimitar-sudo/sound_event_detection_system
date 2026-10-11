"""One-vs-rest detector: an independent binary classifier per sound class."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone


class Identity:
    """No-op preprocessor (tree models use the raw features)."""

    def fit(self, X, Y):
        return self

    def transform(self, X):
        return X

    def select(self, Z, c):
        return Z


class PerClassDetector:
    """Fits ``clone(estimator)`` once per class and returns (N, C) probabilities.

    ``preprocessor`` exposes ``fit(X, Y)``, ``transform(X)`` (shared work done
    once per batch) and ``select(Z, c)`` (the columns class ``c`` uses).
    """

    def __init__(self, estimator, class_names: list[str], preprocessor=None):
        self.estimator = estimator
        self.class_names = list(class_names)
        self.preprocessor = preprocessor if preprocessor is not None else Identity()

    def fit(self, X: np.ndarray, Y: np.ndarray, fit_preprocessor: bool = True):
        if fit_preprocessor:
            self.preprocessor.fit(X, Y)
        Z = self.preprocessor.transform(X)
        self.estimators_ = [
            clone(self.estimator).fit(self.preprocessor.select(Z, c), Y[:, c])
            for c in range(len(self.class_names))
        ]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.preprocessor.transform(X)
        return np.column_stack([
            est.predict_proba(self.preprocessor.select(Z, c))[:, 1]
            for c, est in enumerate(self.estimators_)
        ]).astype(np.float32)
