"""KNN preprocessing: variance filter → standardisation → per-class top-k by MI."""
from __future__ import annotations

import copy

import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from ..config import SEED


class MutualInfoSelector:
    """Keeps, for each class, the ``k`` features with the highest mutual
    information with that class's labels. Everything is fitted on training data.

    Mutual information is estimated on a random subsample of ``mi_sample``
    training frames to keep fitting tractable.
    """

    def __init__(self, k: int = 120, variance_threshold: float = 0.01,
                 mi_sample: int = 20_000, seed: int = SEED):
        self.k = k
        self.variance_threshold = variance_threshold
        self.mi_sample = mi_sample
        self.seed = seed

    def fit(self, X, Y):
        self.variance_ = VarianceThreshold(self.variance_threshold).fit(X)
        Xv = self.variance_.transform(X)
        self.scaler_ = StandardScaler().fit(Xv)
        Xs = self.scaler_.transform(Xv)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(Xs), size=min(self.mi_sample, len(Xs)), replace=False)
        self.mi_scores_ = np.stack([
            mutual_info_classif(Xs[idx], Y[idx, c], random_state=self.seed, n_jobs=-1)
            for c in range(Y.shape[1])
        ])
        self.ranking_ = np.argsort(-self.mi_scores_, axis=1)
        return self

    def with_k(self, k: int) -> "MutualInfoSelector":
        """Same fitted scores, different ``k`` (avoids recomputing MI in sweeps)."""
        other = copy.copy(self)
        other.k = k
        return other

    def transform(self, X):
        return self.scaler_.transform(self.variance_.transform(X)).astype(np.float32)

    def select(self, Z, c):
        if self.k >= Z.shape[1]:
            return Z  # all features: share one matrix instead of 15 copies
        return Z[:, self.ranking_[c, : self.k]]
