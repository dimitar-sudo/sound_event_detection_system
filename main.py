import glob
from collections import defaultdict

import numpy as np

from sed.config import CLASSES_NAMES
from sed.data import get_file_labels, read_files
from sed.models import train_rf_binary_per_class
from sed.models.random_forest import _positive_class_thresholds

# from sed.features import knn_standardisation, load_feature_selectors, load_scaler
# from sed.models import evaluate_knn_on_test, train_knn_binary_per_class
# from sed.visualization import run_qualitative_evaluation

PATH_TO_DATASET = "MLPC2026_dataset_development"
# PATH_TO_SCALER = "sed/data/scaler_reduced_features.joblib"
# PATH_TO_VT_SELECTOR = "sed/data/variance_selector.joblib"
# PATH_TO_MI_SCORES = "sed/data/mi_scores.json"


def build_stratified_split(
    all_paths: list[str],
    rng: np.random.Generator,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> tuple[list[str], list[str], list[str]]:
    """Split file paths into train/val/test using stratified sampling.

    Files are grouped by their label fingerprint so that every label
    combination is proportionally represented in each split.
    """
    file_labels = np.array([get_file_labels(p) for p in all_paths])
    strata: dict[tuple, list[int]] = defaultdict(list)
    for i, label_vec in enumerate(file_labels):
        strata[tuple(label_vec)].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for indices in strata.values():
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val :])

    paths = np.array(all_paths)
    return paths[train_idx].tolist(), paths[val_idx].tolist(), paths[test_idx].tolist()


def main() -> None:
    rng = np.random.default_rng(seed=42)

    all_audio_features_paths = glob.glob(
        f"{PATH_TO_DATASET}/audio_features/*.npz"
    )

    train_files, val_files, test_files = build_stratified_split(
        all_audio_features_paths, rng
    )
    print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

    # --- Load raw features (no preprocessing for RF) ---
    print("Loading training data...")
    X_train, Y_train = read_files(train_files, label_type="binary")
    print("Loading validation data...")
    X_val, Y_val = read_files(val_files, label_type="binary")
    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}\n")

    # --- Train RF (baseline experiment) ---
    models, val_metrics = train_rf_binary_per_class(
        X_train=X_train,
        X_val=X_val,
        Y_train=Y_train,
        Y_val=Y_val,
        class_names=CLASSES_NAMES,
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
    )

    # --- Test set evaluation held out until best hyperparameters are chosen ---
    # evaluate_rf_on_test(
    #     X_test=X_test,
    #     Y_test=Y_test,
    #     n_estimators=...,
    #     max_depth=...,
    #     min_samples_leaf=...,
    #     thresholds=_positive_class_thresholds(Y_train),
    # )


if __name__ == "__main__":
    main()
