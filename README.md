# Sound Event Detection System

**Frame-level multi-label detection of 15 domestic sound events using per-class KNN with mutual-information feature selection.**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.7%2B-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-MLPC2026-blueviolet?style=flat-square)
![Classes](https://img.shields.io/badge/Classes-15-orange?style=flat-square)

---

## 🎧 Demo / Preview

> **Qualitative evaluation** — mel spectrogram with frame-level ground truth and predictions overlaid.  
> Green = true positive · Red = false positive · Orange = false negative

```
File: 003125.npz  — target: vacuum_cleaner; bell_ringing; footsteps; door_open_close; light_switch
┌────────────────────────────────────────────────────────────────────────┐
│  Mel Spectrogram (log)                                                 │
│  ░░░░░▓▓▓▓▒▒▒▒░░░░░░▒▒▒▒▒▓▓▓▓▓▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░         │
├────────────────────────────────────────────────────────────────────────┤
│  Expected (ground truth)              Predicted (KNN k=120, n=3)       │
│  bell_ringing    ██████░░░░░░░░░░     ██████░░░░░░░░░░ (TP)            │
│  vacuum_cleaner  ████████████████     ████████████████ (TP)            │
│  footsteps       ░░░░░░███░░░█░░░     ░░░░░░██░░░░░░░░ (FN)            │
│  door_open_close ░░░░░░░░░░░░███░     ░░░░░░░░░░░░███░ (TP)            │
│  light_switch    ░░░░░░░░░░░░░░█░     ░░░░░░░░░░░░░░█░ (TP)            │
└────────────────────────────────────────────────────────────────────────┘
```

Interactive HTML reports are written to `knn_qualitative/` when `run_qualitative_evaluation()` executes.

---

## 📋 Overview

This system detects the presence of 15 domestic and office sound events (bell ringing, footsteps, keyboard typing, etc.) at the **frame level** across audio recordings from the MLPC2026 challenge dataset. Each recording is represented as a sequence of pre-extracted feature vectors; the task is multi-label classification — multiple events can be active simultaneously.

The core model is a bank of **15 independent binary KNN classifiers** — one per sound class — each operating on a class-specific feature subspace selected by mutual information scoring. This one-vs-rest decomposition keeps each classifier focused on the discriminative structure of its own class, avoiding interference from the high class imbalance present across the full label set. The preprocessing pipeline (VarianceThreshold → StandardScaler → per-class MI top-k) is fitted exclusively on training data and serialised to disk to guarantee no leakage at inference time.

---

## ✨ Key Features

- **One-vs-rest binary KNN bank** — 15 distance-weighted classifiers, each trained on its own class-optimal feature subset.
- **Per-class mutual information feature selection** — selects the top-120 most informative features out of 960 dimensions independently for each class.
- **960-dimensional acoustic feature vector** — covers ZCR, mel-spectrogram statistics (512-d), MFCCs + Δ + ΔΔ (384-d), spectral flux, flatness, centroid, bandwidth, contrast, rolloff, energy, and power.
- **Stratified multilabel split** — train/val/test partitioning is stratified by label fingerprint (the full 15-bit label combination), preserving co-occurrence distributions across all splits.
- **Experiment tracking** — every training run appends balanced accuracy and macro F1 to `knn_experiment_log.csv`, enabling reproducible hyperparameter comparison across `k_features × n_neighbours` sweeps.
- **Interactive Plotly visualisations** — mel-spectrogram + TP/FP/FN frame overlays, per-class confusion matrix grid, and hyperparameter performance curves exported as self-contained HTML.
- **Artefact-safe inference** — scaler, variance selector, and MI scores are loaded from disk; the fit functions are intentionally commented out in `main.py` to prevent accidental refit.

---

## 🏗️ Architecture

```
MLPC2026 Dataset (.npz audio feature files, 3 666 recordings)
            │
            ▼
┌───────────────────────────────────┐
│   Stratified Train / Val / Test   │
│        70%  /  20%  /  10%        │
│  split by 15-bit label fingerprint│
└───────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│       Preprocessing (fit on train)│
│                                   │
│  1. VarianceThreshold (σ²≥0.01)   │   960-d → pruned
│  2. StandardScaler (μ=0, σ=1)     │   normalise
│  3. Per-class MI top-120 mask     │   → 120-d per class
└───────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│              Per-Class Binary KNN Bank (×15)            │
│                                                         │
│  class_i ──► X[:, MI_mask_i]  ──►  KNN(n=3, w=dist)   │
│                                       │                 │
│                              ŷ_i ∈ {0, 1}  (binary)    │
└─────────────────────────────────────────────────────────┘
            │
            ▼
  Frame-level multi-label prediction  (N × 15)
  Evaluated with balanced accuracy + macro F1
```

**Label aggregation:** Each annotation file stores scores from multiple annotators per frame and class. Scores above 0.25 are thresholded to binary votes; a class is considered active if strictly more than half the annotators agree.

---

## 📊 Results

Test-set evaluation — best configuration: **k_features = 120, n_neighbours = 3**.

| Class | Balanced Acc. | Macro F1 |
|---|---|---|
| bell_ringing | 0.7624 | 0.8312 |
| running_water | 0.7691 | 0.7890 |
| keyboard_typing | 0.7430 | 0.7591 |
| microwave | 0.7280 | 0.7594 |
| phone_ringing | 0.7312 | 0.7868 |
| vacuum_cleaner | 0.7153 | 0.7463 |
| keychain | 0.7037 | 0.7407 |
| toilet_flushing | 0.6345 | 0.6359 |
| coffee_machine | 0.6323 | 0.6913 |
| footsteps | 0.6239 | 0.6294 |
| light_switch | 0.5741 | 0.6109 |
| cutlery_dishes | 0.5671 | 0.5936 |
| door_open_close | 0.5609 | 0.5705 |
| window_open_close | 0.5395 | 0.5461 |
| wardrobe_drawer_open_close | 0.5377 | 0.5494 |
| **Mean** | **0.6548** | **0.6826** |

Hyperparameter sweep at k=120 (validation set):

| n\_neighbours | Avg Bal. Acc. | Avg Macro F1 |
|---|---|---|
| **3** | **0.6433** | **0.6670** |
| 5 | 0.6388 | 0.6677 |
| 7 | 0.6348 | 0.6648 |
| 9 | 0.6322 | 0.6627 |
| 11 | 0.6299 | 0.6605 |
| 13 | 0.6285 | 0.6596 |

Performance degrades monotonically as `n_neighbours` increases, confirming that the local neighbourhood structure in the MI-selected subspace is tight.

---

## 📁 Project Structure

```
sound_event_detection_system/
├── main.py                          # Entry point — train, evaluate, visualise
├── requirements.txt                 # Python dependencies
├── knn_experiment_log.csv           # Append-only hyperparameter sweep log
│
├── sed/                             # Core library package
│   ├── config.py                    # Feature names, class names (15 classes)
│   ├── data/
│   │   ├── loader.py                # .npz reader, label aggregation, split utils
│   │   ├── mi_scores.json           # Persisted per-class MI scores (tracked)
│   │   ├── scaler_reduced_features.joblib  # Fitted StandardScaler (gitignored)
│   │   └── variance_selector.joblib        # Fitted VarianceThreshold (gitignored)
│   ├── features/
│   │   └── selection.py             # VT, scaler, MI fitting/loading/application
│   ├── models/
│   │   ├── baseline.py              # Majority-class baseline classifier
│   │   └── knn.py                   # Per-class KNN train + test evaluation
│   └── visualization/
│       ├── plots.py                 # Class distribution + hyperparameter charts
│       └── qualitative.py           # Frame-level prediction overlays + CM grid
│
├── MLPC2026_dataset_development/    # Dataset root (gitignored, ~750 MB)
│   ├── audio_features/              # Pre-extracted .npz files (3 666 recordings)
│   ├── annotations.csv              # Raw annotator-level event annotations
│   └── metadata.csv                 # Recording metadata (device, scene, etc.)
│
├── knn_models/                      # Trained model artefacts (gitignored, ~14 GB)
│   └── knn_k120_n3/                 # One .joblib per class per run
│
└── knn_metrics/                     # Per-run metric CSVs (gitignored)
    └── knn_TEST_metrics_k120_n3_*.csv
```

---

## ⚡ Installation

**Requirements:** Python 3.11+

```bash
git clone https://github.com/<your-username>/sound_event_detection_system.git
cd sound_event_detection_system

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Place the `MLPC2026_dataset_development/` folder at the project root before running any scripts.

---

## 🚀 Usage

### Run qualitative evaluation (default entry point)

Loads pre-fitted preprocessors and models, runs frame-level predictions on two curated test files, and saves interactive HTML plots to `knn_qualitative/`.

```bash
python main.py
```

### Fit preprocessors from scratch

Uncomment the fitting block in `main.py` after loading train data:

```python
from sed.features import (
    fit_and_save_scaler,
    fit_and_save_variance_selector,
    fit_and_save_mi_selector,
    apply_variance_selector,
)

vt_selector = fit_and_save_variance_selector(X_train, vt_path=PATH_TO_VT_SELECTOR)
X_train_vt  = apply_variance_selector(X_train, vt_selector)
scaler       = fit_and_save_scaler(X_train_vt, path=PATH_TO_SCALER)
X_train_sc   = scaler.transform(X_train_vt)
fit_and_save_mi_selector(X_train_sc, Y_train, CLASSES_NAMES, mi_path=PATH_TO_MI_SCORES)
```

### Train KNN models

```python
from sed.features import knn_standardisation
from sed.models import train_knn_binary_per_class

X_train_knn, X_val_knn, _ = knn_standardisation(
    X_train, X_val, X_test,
    vt_selector=vt_selector,
    scaler=scaler,
    class_mi_scores=class_mi_scores,
    k=120,
    class_names=CLASSES_NAMES,
)

models, metrics_df = train_knn_binary_per_class(
    X_train_knn, X_val_knn,
    Y_train, Y_val,
    class_names=CLASSES_NAMES,
    n_neighbors=3,
)
```

### Evaluate on the held-out test set

```python
from sed.models import evaluate_knn_on_test

metrics_df = evaluate_knn_on_test(
    X_test=X_test,
    Y_test=Y_test,
    vt_selector=vt_selector,
    scaler=scaler,
    class_mi_scores=class_mi_scores,
    k_features=120,
    n_neighbours=3,
)
```

### Generate visualisations

```python
from sed.visualization import visualize_class_distribution, visualize_knn_n_neighbours

# Class frequency across train / val / test splits
visualize_class_distribution(Y_train, Y_val, Y_test, CLASSES_NAMES, save=True)

# Hyperparameter sweep results
visualize_knn_n_neighbours(
    experiment_log="knn_experiment_log.csv",
    k_features_filter=120,
    save_path="knn_n_neighbours_performance.png",
)
```

---

## ⚙️ Configuration

All key constants live in [sed/config.py](sed/config.py).

| Parameter | Location | Default | Description |
|---|---|---|---|
| `CLASSES_NAMES` | `config.py` | 15 classes | Target sound event classes |
| `FEATURE_NAMES` | `config.py` | 56 feature keys | Acoustic feature groups to load |
| `variance_threshold` | `selection.py` | `0.01` | Minimum per-feature variance to retain |
| `k` (MI top-k) | `knn_standardisation()` | `120` | Features selected per class by MI rank |
| `n_neighbors` | `train_knn_binary_per_class()` | `5` (best found: `3`) | KNN neighbourhood size |
| `weights` | `knn.py` | `"distance"` | KNN vote weighting scheme |
| `train_ratio` | `build_stratified_split()` | `0.7` | Fraction of data used for training |
| `val_ratio` | `build_stratified_split()` | `0.2` | Fraction of data used for validation |
| `_ANNOTATION_THRESHOLD` | `loader.py` | `0.25` | Minimum annotator score to count as a positive vote |
| `seed` | `main.py` | `42` | Global RNG seed for reproducibility |

To sweep `n_neighbours`, modify the call in `main.py` and re-run; results are automatically appended to `knn_experiment_log.csv`.

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| Python 3.11+ | Runtime |
| NumPy 2.4+ | Feature matrices, array ops |
| pandas 3.0+ | Metrics DataFrames, CSV I/O |
| scikit-learn 1.8+ | KNN, StandardScaler, VarianceThreshold, MI scoring, metrics |
| joblib 1.5+ | Model and preprocessor serialisation |
| Plotly 6.7+ | Interactive visualisation (HTML export) |
| kaleido 1.3+ | Static image export from Plotly figures |

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

The MLPC2026 dataset is distributed under **CC0 (Public Domain Dedication)** by its respective collectors.
