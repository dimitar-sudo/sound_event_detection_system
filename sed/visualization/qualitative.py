"""Qualitative evaluation plots for the per-class KNN classifier.

Provides:
- ``predict_file_framewise``: load one .npz file, run all per-class KNN models,
  return (T, C) predicted and ground-truth label matrices plus the mel spectrogram.
- ``plot_file_predictions``: render mel-spectrogram + predicted-vs-expected
  framewise labels into a single Plotly figure.
- ``plot_per_class_confusion_matrices``: a 3x5 grid of per-class confusion matrices.
"""

import glob
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from ..config import CLASSES_NAMES
from ..data.loader import get_features_and_targets
from ..features.selection import (
    apply_mi_selector,
    apply_scaler,
    apply_variance_selector,
    create_top_k_masks,
)


def _load_models(
    models_dir: str,
    k_features: int,
    n_neighbours: int,
) -> Dict[str, object]:
    """Load the most recent KNN classifier per class for a given (k, n) run."""
    run_dir = os.path.join(models_dir, f"knn_k{k_features}_n{n_neighbours}")
    models: Dict[str, object] = {}
    for cls in CLASSES_NAMES:
        pattern = os.path.join(
            run_dir, f"knn_{cls}_k{k_features}_n{n_neighbours}_*.joblib"
        )
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"[{cls}] no saved model found")
            continue
        models[cls] = joblib.load(matches[-1])
    return models


def predict_file_framewise(
    npz_path: str,
    models: Dict[str, object],
    vt_selector,
    scaler,
    class_masks: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full preprocessing + per-class KNN inference on one file.

    Returns
    -------
    y_pred  : (T, C) binary predictions
    y_true  : (T, C) binary ground truth labels
    mel     : (T, n_mels) mel-spectrogram (mean per frame)
    times   : (T,) frame start times in seconds
    """
    X, Y = get_features_and_targets(npz_path, label_type="multilabel")
    X_vt = apply_variance_selector(X, vt_selector)
    X_scaled = apply_scaler(scaler, X_vt)[0]

    T = X.shape[0]
    C = len(CLASSES_NAMES)
    y_pred = np.zeros((T, C), dtype=int)
    for i, cls in enumerate(CLASSES_NAMES):
        if cls not in models:
            continue
        X_cls = apply_mi_selector(X_scaled, class_masks, cls)
        y_pred[:, i] = models[cls].predict(X_cls)

    raw = np.load(npz_path, allow_pickle=True)
    mel = raw["melspect_mean"]
    times = raw["start_time"]

    return y_pred, Y.astype(int), mel, times


def _wrap_class_name(name: str) -> str:
    return name.replace("_", " ").title()


def plot_file_predictions(
    npz_path: str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mel: np.ndarray,
    times: np.ndarray,
    class_names: List[str],
    metadata_row: pd.Series | None = None,
    save_path: str | None = None,
) -> go.Figure:
    """Plot mel-spectrogram + per-frame ground-truth and predicted labels.

    The lower panel shows two heatmap rows per class: top = expected,
    bottom = predicted. Correct positives are coloured green, false positives
    red, false negatives orange, true negatives left blank.
    """
    fname = os.path.basename(npz_path)

    # 0 = nothing, 1 = TP (green), 2 = FP (red), 3 = FN (orange)
    T, C = y_true.shape
    pred_overlay = np.zeros_like(y_pred)
    pred_overlay[(y_pred == 1) & (y_true == 1)] = 1   # TP
    pred_overlay[(y_pred == 1) & (y_true == 0)] = 2   # FP
    pred_overlay[(y_pred == 0) & (y_true == 1)] = 3   # FN
    # ground truth: 0 = absent, 1 = present
    truth_overlay = y_true.copy()

    title = (
        f"File: {fname}"
        + (
            f" — target: {metadata_row['target_classes']}"
            if metadata_row is not None and pd.notna(metadata_row.get('target_classes'))
            else ""
        )
    )

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.42, 0.29, 0.29],
        subplot_titles=[
            "Mel Spectrogram (log)",
            "Expected (ground truth)",
            "Predicted (KNN k=120, n=3)",
        ],
    )

    # ── 1. Mel spectrogram ───────────────────────────────────────────────
    mel_db = 10.0 * np.log10(np.maximum(mel.T, 1e-10))
    fig.add_trace(
        go.Heatmap(
            z=mel_db,
            x=times,
            colorscale="Viridis",
            colorbar=dict(title="dB", thickness=12, x=1.02, y=0.80, len=0.35),
            zsmooth="best",
            showscale=True,
        ),
        row=1, col=1,
    )

    labels = [_wrap_class_name(c) for c in class_names]

    # ── 2. Ground truth heatmap ──────────────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            z=truth_overlay.T,
            x=times,
            y=labels,
            colorscale=[[0, "#F8F9FB"], [1, "#2C7FB8"]],
            zmin=0, zmax=1,
            showscale=False,
            xgap=1, ygap=1,
            hovertemplate="t=%{x:.1f}s<br>class=%{y}<br>label=%{z}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── 3. Prediction heatmap with TP/FP/FN colouring ────────────────────
    pred_colorscale = [
        [0.00, "#F8F9FB"],  # 0 = TN
        [0.25, "#F8F9FB"],
        [0.26, "#2CA02C"],  # 1 = TP
        [0.50, "#2CA02C"],
        [0.51, "#D62728"],  # 2 = FP
        [0.75, "#D62728"],
        [0.76, "#FF9F40"],  # 3 = FN
        [1.00, "#FF9F40"],
    ]
    fig.add_trace(
        go.Heatmap(
            z=pred_overlay.T,
            x=times,
            y=labels,
            colorscale=pred_colorscale,
            zmin=0, zmax=3,
            showscale=False,
            xgap=1, ygap=1,
            hovertemplate=(
                "t=%{x:.1f}s<br>class=%{y}<br>code=%{z}"
                "<extra>0=TN, 1=TP, 2=FP, 3=FN</extra>"
            ),
        ),
        row=3, col=1,
    )

    # Manual legend for prediction colours (one dummy scatter per state)
    legend_entries = [
        ("True positive", "#2CA02C"),
        ("False positive", "#D62728"),
        ("False negative", "#FF9F40"),
    ]
    for name, colour in legend_entries:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=colour, symbol="square"),
                name=name,
                showlegend=True,
            ),
            row=3, col=1,
        )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=16, color="#2C3E50", family="Inter, Arial, sans-serif"),
        ),
        height=900,
        width=1200,
        margin=dict(t=90, b=70, l=140, r=80),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial, sans-serif", color="#2C3E50"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    fig.update_yaxes(title_text="mel bin", row=1, col=1, autorange="reversed")
    fig.update_yaxes(title_text="class", row=2, col=1, autorange="reversed")
    fig.update_yaxes(title_text="class", row=3, col=1, autorange="reversed")
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)

    fig.show()

    if save_path:
        html_path = save_path.replace(".png", ".html")
        fig.write_html(html_path)
        print(f"Saved qualitative plot → {html_path}")

    return fig


def plot_per_class_confusion_matrices(
    y_pred_all: np.ndarray,
    y_true_all: np.ndarray,
    class_names: List[str],
    save_path: str | None = None,
) -> go.Figure:
    """Render a 3x5 grid of per-class binary confusion matrices.

    Each cell shows raw counts and the row-normalised proportion.
    """
    n_classes = len(class_names)
    n_cols = 5
    n_rows = int(np.ceil(n_classes / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[_wrap_class_name(c) for c in class_names],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    for i, cls in enumerate(class_names):
        r = i // n_cols + 1
        c = i % n_cols + 1
        cm = confusion_matrix(y_true_all[:, i], y_pred_all[:, i], labels=[0, 1])
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        annotations = np.array([
            [f"{cm[0, 0]}<br>({cm_norm[0, 0]:.2f})", f"{cm[0, 1]}<br>({cm_norm[0, 1]:.2f})"],
            [f"{cm[1, 0]}<br>({cm_norm[1, 0]:.2f})", f"{cm[1, 1]}<br>({cm_norm[1, 1]:.2f})"],
        ])

        fig.add_trace(
            go.Heatmap(
                z=cm_norm,
                x=["pred 0", "pred 1"],
                y=["true 0", "true 1"],
                colorscale="Blues",
                zmin=0, zmax=1,
                showscale=(i == 0),
                colorbar=dict(thickness=10, x=1.02, len=0.35, y=0.80, title="row %") if i == 0 else None,
                text=annotations,
                texttemplate="%{text}",
                textfont=dict(size=10, color="#2C3E50"),
                hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
            ),
            row=r, col=c,
        )
        fig.update_yaxes(autorange="reversed", row=r, col=c)

    fig.update_layout(
        title=dict(
            text="Per-Class Binary Confusion Matrices (KNN k=120, n=3, test set)",
            x=0.5, xanchor="center",
            font=dict(size=16, color="#2C3E50", family="Inter, Arial, sans-serif"),
        ),
        height=720,
        width=1500,
        margin=dict(t=90, b=40, l=60, r=80),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial, sans-serif", color="#2C3E50"),
    )

    fig.show()

    if save_path:
        html_path = save_path.replace(".png", ".html")
        fig.write_html(html_path)
        print(f"Saved confusion-matrix grid → {html_path}")

    return fig


def run_qualitative_evaluation(
    test_files: List[str],
    vt_selector,
    scaler,
    class_mi_scores: Dict[str, np.ndarray],
    k_features: int = 120,
    n_neighbours: int = 3,
    models_dir: str = "knn_models",
    metadata_csv: str = "MLPC2026_dataset_development/metadata.csv",
    selected_files: List[str] | None = None,
    out_dir: str = "knn_qualitative",
) -> None:
    """End-to-end driver: pick two test files, plot them, then a confusion grid."""
    os.makedirs(out_dir, exist_ok=True)

    models = _load_models(models_dir, k_features, n_neighbours)
    class_masks = create_top_k_masks(class_mi_scores, k_features)

    # Pick the two files. If user did not pre-select, choose by interestingness
    # (multiple target classes, decent length).
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata.set_index("filename")

    if selected_files is None:
        # Score each test file by number of target classes
        scored = []
        for path in test_files:
            wav_name = os.path.basename(path).replace(".npz", ".wav")
            if wav_name not in metadata.index:
                continue
            row = metadata.loc[wav_name]
            targets = str(row["target_classes"]).split(";") if pd.notna(row["target_classes"]) else []
            scored.append((len(targets), path))
        scored.sort(reverse=True)
        # take two files with different scene environments if possible
        selected_files = [scored[0][1], scored[5][1]] if len(scored) > 5 else [p for _, p in scored[:2]]

    # ── Qualitative plots ──────────────────────────────────────────────
    for path in selected_files:
        wav_name = os.path.basename(path).replace(".npz", ".wav")
        meta_row = metadata.loc[wav_name] if wav_name in metadata.index else None
        y_pred, y_true, mel, times = predict_file_framewise(
            path, models, vt_selector, scaler, class_masks
        )
        plot_file_predictions(
            path, y_pred, y_true, mel, times,
            class_names=CLASSES_NAMES,
            metadata_row=meta_row,
            save_path=os.path.join(out_dir, f"qualitative_{wav_name.replace('.wav', '')}.png"),
        )

    # ── Aggregate confusion matrices on full test set ──────────────────
    print("Aggregating predictions across the full test set …")
    y_pred_all, y_true_all = [], []
    for path in test_files:
        y_pred, y_true, _, _ = predict_file_framewise(
            path, models, vt_selector, scaler, class_masks
        )
        y_pred_all.append(y_pred)
        y_true_all.append(y_true)
    y_pred_all = np.vstack(y_pred_all)
    y_true_all = np.vstack(y_true_all)

    plot_per_class_confusion_matrices(
        y_pred_all, y_true_all,
        class_names=CLASSES_NAMES,
        save_path=os.path.join(out_dir, "confusion_matrices_k120_n3.png"),
    )
