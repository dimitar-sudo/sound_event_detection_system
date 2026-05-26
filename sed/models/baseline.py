import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    """Always predicts the majority class from the training set."""

    def __init__(self) -> None:
        self.majority_class: int | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "BaselineClassifier":
        self.classes_ = np.unique(y_train)
        self.majority_class = 1 if y_train.sum() > len(y_train) / 2 else 0
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.majority_class)
