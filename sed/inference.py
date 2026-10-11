"""Run the shipped detector on one feature file and list detected events."""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .data.loader import aggregate_labels, load_features
from .evaluation import binarize

FRAME_SECONDS = 1.0  # each frame covers a 1 s window (0.5 s hop)


def load_detector(path: str = os.path.join(MODEL_DIR, "detector.joblib")) -> dict:
    return joblib.load(path)


def frames_to_events(pred: np.ndarray, start_times: np.ndarray,
                     class_names: list[str]) -> pd.DataFrame:
    """Merge runs of consecutive positive frames into (class, onset, offset) events."""
    rows = []
    for c, name in enumerate(class_names):
        active = np.flatnonzero(pred[:, c])
        if not len(active):
            continue
        breaks = np.flatnonzero(np.diff(active) > 1)
        for seg in np.split(active, breaks + 1):
            rows.append({"event": name, "onset_s": float(start_times[seg[0]]),
                         "offset_s": float(start_times[seg[-1]] + FRAME_SECONDS)})
    return pd.DataFrame(rows, columns=["event", "onset_s", "offset_s"]).sort_values(
        ["onset_s", "event"], ignore_index=True)


def predict_file(path: str, plot_path: str | None = None, bundle: dict | None = None):
    """Print detected events for ``path``; optionally save a timeline figure."""
    bundle = bundle or load_detector()
    npz = np.load(path, allow_pickle=True)
    proba = bundle["detector"].predict_proba(load_features(npz))
    pred = binarize(proba, bundle["thresholds"])
    events = frames_to_events(pred, npz["start_time"], bundle["class_names"])
    print(f"{os.path.basename(path)}: {len(npz['start_time'])} frames, "
          f"{len(events)} detected events")
    if len(events):
        print(events.to_string(index=False))
    if plot_path:
        from .visualization.figures import plot_timeline
        truth = aggregate_labels(npz["annotations"]) if "annotations" in npz.files else None
        plot_timeline(npz, pred, truth, bundle["class_names"], plot_path)
        print(f"Timeline saved → {plot_path}")
    return proba, pred, events
