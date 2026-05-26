import glob
from collections import defaultdict

import numpy as np

from sed.config import CLASSES_NAMES
from sed.data import get_file_labels, read_files
from sed.features import knn_standardisation, load_feature_selectors, load_scaler
from sed.models import evaluate_knn_on_test, train_knn_binary_per_class
from sed.visualization import run_qualitative_evaluation

PATH_TO_DATASET = "MLPC2026_dataset_development"
PATH_TO_SCALER = "sed/data/scaler_reduced_features.joblib"
PATH_TO_VT_SELECTOR = "sed/data/variance_selector.joblib"
PATH_TO_MI_SCORES = "sed/data/mi_scores.json"


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

    _train_files, _val_files, test_files = build_stratified_split(
        all_audio_features_paths, rng
    )

    # --- Fitting (run once, then comment out and use load_* below) ---
    # from sed.features import (
    #     fit_and_save_scaler,
    #     fit_and_save_variance_selector,
    #     fit_and_save_mi_selector,
    #     apply_variance_selector,
    # )
    # vt_selector = fit_and_save_variance_selector(X_train, vt_path=PATH_TO_VT_SELECTOR)
    # X_train_vt = apply_variance_selector(X_train, vt_selector)
    # scaler = fit_and_save_scaler(X_train_vt, path=PATH_TO_SCALER)
    # X_train_scaled = scaler.transform(X_train_vt)
    # fit_and_save_mi_selector(X_train_scaled, Y_train, CLASSES_NAMES, mi_path=PATH_TO_MI_SCORES)

    # --- Inference (normal entry point) ---
    vt_selector, class_mi_scores = load_feature_selectors(
        mi_path=PATH_TO_MI_SCORES,
        vt_path=PATH_TO_VT_SELECTOR,
    )
    scaler = load_scaler(PATH_TO_SCALER)

    #--- Training new KNN models ---
    # X_train_knn, X_val_knn, _ = knn_standardisation(
    #     X_train, X_val, X_test,
    #     vt_selector=vt_selector,
    #     scaler=scaler,
    #     class_mi_scores=class_mi_scores,
    #     k=120,
    #     class_names=CLASSES_NAMES,
    # )
    # train_knn_binary_per_class(
    #     X_train_knn, X_val_knn,
    #     Y_train, Y_val,
    #     class_names=CLASSES_NAMES,
    #     n_neighbors=13,
    # )

    # --- Evaluating on test set ---
    #evaluate_knn_on_test(
    #    X_test=X_test,
    #    Y_test=Y_test,
    #    vt_selector=vt_selector,
    #    scaler=scaler,
    #    class_mi_scores=class_mi_scores,
    #    k_features=120,
    #    n_neighbours=3,
    #)

    # --- Qualitative evaluation (k=120, n=3) ---
    # Two test files chosen for contrasting content:
    #  003125 — kitchen/hallway scene with vacuum_cleaner, bell_ringing,
    #           footsteps, door_open_close, light_switch
    #  001467 — bedroom scene with phone_ringing, footsteps, light_switch,
    #           window_open_close
    selected = [
        f"{PATH_TO_DATASET}/audio_features/003125.npz",
        f"{PATH_TO_DATASET}/audio_features/001467.npz",
    ]
    run_qualitative_evaluation(
        test_files=test_files,
        vt_selector=vt_selector,
        scaler=scaler,
        class_mi_scores=class_mi_scores,
        k_features=120,
        n_neighbours=3,
        selected_files=selected,
    )


if __name__ == "__main__":
    main()
