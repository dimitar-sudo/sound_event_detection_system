"""Static (PNG) figures for the README and the ``predict --plot`` command."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..config import CLASSES_NAMES, FIGURES_DIR, RESULTS_DIR  # noqa: E402

TP, FP, FN = "#2CA02C", "#D62728", "#FF9F40"
MODEL_COLORS = {"baseline": "#B0B7C3", "knn": "#8DA0CB", "random_forest": "#66C2A5",
                "boosting": "#1F6FEB"}


def pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def plot_timeline(npz, pred: np.ndarray, truth: np.ndarray | None,
                  class_names: list[str], path: str, title: str | None = None) -> None:
    """Log-mel spectrogram above a per-class timeline of detections.

    Only classes that are annotated or predicted somewhere in the file are shown.
    With ground truth, predictions are coloured TP / FP / FN.
    """
    times = npz["start_time"]
    t_end = times[-1] + 1.0
    show = np.flatnonzero(pred.any(0) | (truth.any(0) if truth is not None else False))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 2.6 + 0.42 * max(len(show), 1)), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 0.42 * max(len(show), 1)]})

    ax1.imshow(npz["melspect_mean"].T, origin="lower", aspect="auto", cmap="magma",
               extent=[0, t_end, 0, npz["melspect_mean"].shape[1]])
    ax1.set_ylabel("Mel bin")
    ax1.set_title(title or "Log-mel spectrogram", loc="left", fontsize=12, fontweight="semibold")

    for row, c in enumerate(show):
        for i, t in enumerate(times):
            p = pred[i, c]
            g = truth[i, c] if truth is not None else None
            if g is None:
                color = TP if p else None
            else:
                color = TP if (p and g) else FP if p else FN if g else None
            if color:
                ax2.add_patch(plt.Rectangle((t, row - 0.38), 0.5, 0.76, color=color, lw=0))
    ax2.set_yticks(range(len(show)), [pretty(class_names[c]) for c in show])
    ax2.set_ylim(len(show) - 0.5, -0.5)
    ax2.set_xlim(0, t_end)
    ax2.set_xlabel("Time (s)")
    ax2.grid(axis="x", alpha=0.3)
    _style(ax2)
    if truth is not None:
        ax2.legend(handles=[Patch(color=TP, label="Correctly detected"),
                            Patch(color=FP, label="False alarm"),
                            Patch(color=FN, label="Missed")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.32 if len(show) > 2 else -0.6),
                   ncol=3, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, path: str) -> None:
    """Grouped bars: test macro PR-AUC and F1 for every model."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(2)
    width = 0.8 / len(summary)
    for i, r in enumerate(summary.itertuples()):
        vals = [r.test_pr_auc, r.test_f1]
        bars = ax.bar(x + (i - (len(summary) - 1) / 2) * width, vals, width * 0.92,
                      color=MODEL_COLORS.get(r.key, "grey"), label=r.model)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, ["PR-AUC", "F1 (positive class)"])
    ax.set_ylabel("Test score (mean over 15 classes)")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison — test set (people unseen in training)", loc="left",
                 fontsize=12, fontweight="semibold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(metrics: pd.DataFrame, path: str, model_label: str) -> None:
    """Per-class test PR-AUC (sorted) against the chance level (prevalence)."""
    m = metrics.sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    y = np.arange(len(m))
    ax.barh(y, m["pr_auc"], color=MODEL_COLORS["boosting"], label=f"{model_label} PR-AUC")
    ax.scatter(m["prevalence"], y, color="black", marker="|", s=120, zorder=3,
               label="Chance level (prevalence)")
    for yi, v in zip(y, m["pr_auc"]):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, [pretty(c) for c in m.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test PR-AUC")
    ax.set_title("Per-class detection quality", loc="left", fontsize=12, fontweight="semibold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_sweeps(path: str) -> None:
    """Validation PR-AUC for every config tried (selected config marked)."""
    from ..models import LABELS
    read = lambda n: pd.read_csv(os.path.join(RESULTS_DIR, "sweeps", f"{n}.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8),
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # KNN: heatmap over (features kept, neighbours); blank = not tried.
    knn = read("knn")
    grid = knn.pivot_table(index="k_features", columns="n_neighbors", values="val_pr_auc")
    ax = axes[0]
    im = ax.imshow(grid.values, cmap="Blues", vmin=0.3, vmax=0.65, aspect="auto")
    best = knn.loc[knn.val_pr_auc.idxmax()]
    for i, k in enumerate(grid.index):
        for j, n in enumerate(grid.columns):
            v = grid.values[i, j]
            if not np.isnan(v):
                is_best = k == best.k_features and n == best.n_neighbors
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.5 else "black",
                        fontweight="bold" if is_best else "normal",
                        bbox=dict(boxstyle="round,pad=0.2", fc="none", ec="black") if is_best else None)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("Neighbours")
    ax.set_ylabel("Features kept (960 = all)")
    ax.set_title(LABELS["knn"], loc="left", fontsize=11, fontweight="semibold")

    for ax, n, col, xlabel in [(axes[1], "random_forest", "min_samples_leaf", "Min. samples per leaf"),
                               (axes[2], "boosting", "max_leaf_nodes", "Max. leaves per tree")]:
        df = read(n).sort_values(col)
        b = df.loc[df.val_pr_auc.idxmax()]
        ax.plot(df[col], df.val_pr_auc, "o-", color=MODEL_COLORS[n], lw=2)
        ax.plot(b[col], b.val_pr_auc, "o", ms=12, mfc="none", mec="black", mew=1.5,
                label=f"Selected ({b.val_pr_auc:.3f})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(df[col], df[col].astype(int))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation PR-AUC")
        ax.set_title(LABELS[n], loc="left", fontsize=11, fontweight="semibold")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3)
        _style(ax)
    fig.suptitle("Hyperparameter search — validation PR-AUC", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def pick_demo_file(ds, pred: np.ndarray, min_classes: int = 2, min_frames: int = 40) -> int:
    """Index of the best-detected test file among those with >= ``min_classes``
    annotated event classes: highest frame-level F1 over all its classes."""
    best, best_f1 = None, -1.0
    for f in np.unique(ds.test.file_idx):
        rows = ds.test.file_idx == f
        y, p = ds.test.Y[rows], pred[rows]
        if rows.sum() < min_frames or (y.any(0)).sum() < min_classes:
            continue
        tp = (y & p).sum()
        f1 = 2 * tp / max(y.sum() + p.sum(), 1)
        if f1 > best_f1:
            best, best_f1 = f, f1
    return int(best)


def make_all() -> None:
    """Regenerate every README figure from ``results/`` and the cached probabilities."""
    from ..config import DATASET_DIR, EXAMPLES_DIR
    from ..data import load_dataset
    from ..evaluation import binarize
    from ..experiment import _proba_path, best_config

    os.makedirs(FIGURES_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    plot_model_comparison(summary, os.path.join(FIGURES_DIR, "model_comparison.png"))
    best = summary[summary.key != "baseline"].sort_values("val_pr_auc").iloc[-1]
    metrics = pd.read_csv(os.path.join(RESULTS_DIR, f"test_metrics_{best.key}.csv"), index_col=0)
    plot_per_class(metrics, os.path.join(FIGURES_DIR, "per_class_pr_auc.png"), best.model)
    plot_sweeps(os.path.join(FIGURES_DIR, "hyperparameter_search.png"))

    # Demo: a well-detected recording from the held-out test collectors.
    ds = load_dataset()
    with open(os.path.join(RESULTS_DIR, "selected_configs.json")) as fh:
        thr = np.array(list(json.load(fh)[best.key]["thresholds"].values()))
    pred = binarize(np.load(_proba_path(best.key, best_config(best.key)))["test"], thr)
    f = pick_demo_file(ds, pred)
    name = str(ds.files[f])
    rows = ds.test.file_idx == f
    npz = np.load(os.path.join(DATASET_DIR, "audio_features", name), allow_pickle=True)
    plot_timeline(npz, pred[rows], ds.test.Y[rows], CLASSES_NAMES,
                  os.path.join(FIGURES_DIR, "demo_timeline.png"),
                  title=f"Test recording {name[:-4]} — log-mel spectrogram")
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(DATASET_DIR, "audio_features", name), os.path.join(EXAMPLES_DIR, name))
    print(f"Demo file: {name} (copied to {EXAMPLES_DIR})")
    print(f"Figures written to {FIGURES_DIR}")
