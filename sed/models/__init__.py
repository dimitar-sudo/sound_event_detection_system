"""Model registry: name -> hyperparameter grid and a builder for each config.

All models are wrapped in :class:`PerClassDetector` (one binary classifier per
class) so they are trained, thresholded and evaluated identically.
"""
from __future__ import annotations

from itertools import product

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ..config import CLASSES_NAMES, SEED
from .base import PerClassDetector
from .knn import MutualInfoSelector


def _grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, values)) for values in product(*axes.values())]


GRIDS: dict[str, list[dict]] = {
    # Scores every frame with the training prevalence: the "no skill" reference.
    "baseline": [{}],
    # The optimum kept landing on the grid's edge, so points beyond it were
    # added until it no longer improved (k_features=960 means no MI selection).
    "knn": _grid(k_features=[30, 60, 120, 240], n_neighbors=[5, 15, 31, 63, 127])
           + [{"k_features": 480, "n_neighbors": 127}, {"k_features": 240, "n_neighbors": 255},
              {"k_features": 960, "n_neighbors": 127}],
    "random_forest": _grid(min_samples_leaf=[1, 4, 16, 32]),
    "boosting": _grid(learning_rate=[0.1], max_leaf_nodes=[15, 31, 63, 127]),
}

LABELS = {
    "baseline": "Prevalence baseline",
    "knn": "KNN",
    "random_forest": "Random forest",
    "boosting": "Gradient boosting",
}


def build(name: str, params: dict, shared: dict | None = None) -> PerClassDetector:
    """Build an unfitted detector. ``shared`` may carry a pre-fitted MI selector."""
    if name == "baseline":
        return PerClassDetector(DummyClassifier(strategy="prior"), CLASSES_NAMES)
    if name == "knn":
        selector = (shared or {}).get("mi_selector")
        selector = selector.with_k(params["k_features"]) if selector else MutualInfoSelector(params["k_features"])
        est = KNeighborsClassifier(n_neighbors=params["n_neighbors"], weights="distance", n_jobs=-1)
        return PerClassDetector(est, CLASSES_NAMES, selector)
    if name == "random_forest":
        est = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=SEED)
        return PerClassDetector(est, CLASSES_NAMES)
    if name == "boosting":
        est = HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"], max_leaf_nodes=params["max_leaf_nodes"],
            max_iter=1000, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=SEED)
        return PerClassDetector(est, CLASSES_NAMES)
    raise ValueError(f"unknown model {name!r}; choose from {list(GRIDS)}")
