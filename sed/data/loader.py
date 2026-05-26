from typing import Dict, List, Literal, Tuple, Union

import numpy as np

from ..config import CLASSES_NAMES, FEATURE_NAMES

# Majority-vote threshold: annotator confidence score must exceed this value
# to count as a positive vote for a class.
_ANNOTATION_THRESHOLD = 0.25


def get_file_labels(filepath: str) -> np.ndarray:
    """Return a (C,) binary vector: 1 if the class is present anywhere in the file."""
    data = np.load(filepath, allow_pickle=True)
    annotations = data["annotations"]
    annotations = (annotations > _ANNOTATION_THRESHOLD).astype(int)
    votes = annotations.sum(axis=2)
    multilabel = (votes >= annotations.shape[-1] // 2).astype(int)
    return multilabel.max(axis=0)


def make_binary_targets(Y: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert a multilabel indicator matrix (N, C) into per-class binary arrays.

    Returns
    -------
    dict mapping each class name to a binary array of shape (N, 1).
    """
    return {
        class_name: Y[:, idx].reshape(-1, 1).astype(int)
        for idx, class_name in enumerate(CLASSES_NAMES)
    }


def get_features_and_targets(
    filename: str,
    label_type: Literal["binary", "multiclass", "multilabel"] = "multilabel",
) -> Tuple[np.ndarray, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """Load audio features and aggregated target labels from a .npz file.

    Parameters
    ----------
    filename:
        Path to the .npz file containing audio features and annotations.
    label_type:
        How to encode the targets:
        - "multilabel": binary matrix (T, C), one column per class.
        - "multiclass": integer class index (T, 1), argmax over classes.
        - "binary": dict mapping each class name to a binary array (T, 1).

    Returns
    -------
    x : np.ndarray of shape (T, D)
    y : np.ndarray of shape (T, C) or (T, 1), or Dict[str, np.ndarray]
    """
    audio_features = dict(np.load(filename, allow_pickle=True))
    annotations = audio_features["annotations"]
    T, C, _ = annotations.shape

    x = np.empty(
        (
            T,
            sum(
                audio_features[f].shape[1] if audio_features[f].ndim > 1 else 1
                for f in FEATURE_NAMES
            ),
        )
    )
    col = 0
    for feat_name in FEATURE_NAMES:
        feat = audio_features[feat_name]
        width = feat.shape[1] if feat.ndim > 1 else 1
        x[:, col : col + width] = feat if feat.ndim > 1 else feat[:, None]
        col += width

    annotations = (annotations > _ANNOTATION_THRESHOLD).astype(int)
    votes = annotations.sum(axis=2)

    if label_type == "multilabel":
        y = (votes > (annotations.shape[-1] // 2)).astype(int)
    elif label_type == "multiclass":
        y = np.argmax(votes, axis=1).astype(int).reshape(T, 1)
    else:  # "binary"
        y = make_binary_targets((votes > (annotations.shape[-1] // 2)).astype(int))

    return x, y


def read_files(
    file_list: List[str],
    label_type: Literal["binary", "multiclass", "multilabel"] = "multilabel",
) -> Tuple[np.ndarray, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """Load and stack features and labels from a list of .npz files.

    Parameters
    ----------
    file_list:
        Paths to .npz audio feature files to load.
    label_type:
        Passed through to ``get_features_and_targets``.

    Returns
    -------
    X : np.ndarray of shape (N, D)
    Y : np.ndarray of shape (N, C) or (N, 1), or Dict[str, np.ndarray]
    """
    if label_type not in ("binary", "multiclass", "multilabel"):
        raise ValueError(
            f"`label_type` must be 'binary', 'multiclass', or 'multilabel', "
            f"got {label_type!r}"
        )

    X, Y = [], []
    for filename in file_list:
        x, y = get_features_and_targets(filename, label_type=label_type)
        X.append(x)
        Y.append(y)

    X = np.vstack(X)

    if label_type == "binary":
        return X, {cls: np.vstack([y[cls] for y in Y]) for cls in CLASSES_NAMES}
    return X, np.vstack(Y)
