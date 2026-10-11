"""Hyperparameter sweeps, model selection and the final test evaluation.

Protocol (identical for every model):

1. ``sweep``: fit each config of the model's grid on the *train* split and
   score it on *validation* by mean per-class PR-AUC. Validation and test
   probabilities are cached per config so nothing has to be refitted.
2. ``evaluate``: per model, pick the config with the best validation PR-AUC,
   choose per-class thresholds that maximise validation F1, and apply those
   exact thresholds to the *test* split. Test data is never used for choices.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from .config import CACHE_PATH, CLASSES_NAMES, MODEL_DIR, RESULTS_DIR
from .data import load_dataset
from .evaluation import best_f1_thresholds, macro, per_class_metrics, pr_auc_per_class
from .models import GRIDS, LABELS, build
from .models.knn import MutualInfoSelector

PROBA_DIR = os.path.join(os.path.dirname(CACHE_PATH), "probas")
SWEEP_DIR = os.path.join(RESULTS_DIR, "sweeps")


def config_id(params: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in params.items()) or "default"


def _proba_path(name: str, params: dict) -> str:
    return os.path.join(PROBA_DIR, f"{name}__{config_id(params)}.npz")


def sweep(name: str) -> pd.DataFrame:
    """Train and validate every config of ``name``; resumable after interruption."""
    os.makedirs(PROBA_DIR, exist_ok=True)
    os.makedirs(SWEEP_DIR, exist_ok=True)
    out_csv = os.path.join(SWEEP_DIR, f"{name}.csv")
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    ds = load_dataset()
    shared: dict = {}

    for params in GRIDS[name]:
        cid = config_id(params)
        if len(done) and cid in set(done["config"]):
            print(f"[{name}] {cid}: already done, skipping")
            continue
        if name == "knn" and "mi_selector" not in shared:
            print("[knn] fitting variance filter, scaler and mutual-information scores …")
            shared["mi_selector"] = MutualInfoSelector().fit(ds.train.X, ds.train.Y)

        t0 = time.time()
        det = build(name, params, shared)
        det.fit(ds.train.X, ds.train.Y, fit_preprocessor=(name != "knn"))
        val_proba = det.predict_proba(ds.val.X)
        test_proba = det.predict_proba(ds.test.X)
        np.savez_compressed(_proba_path(name, params), val=val_proba, test=test_proba)
        if name in ("baseline", "boosting"):  # small enough to keep for shipping
            joblib.dump(det, _proba_path(name, params).replace(".npz", ".joblib"))

        scores = pr_auc_per_class(ds.val.Y, val_proba)
        row = {"config": cid, **params, "val_pr_auc": scores.mean(),
               "fit_minutes": round((time.time() - t0) / 60, 1),
               **{f"pr_auc_{c}": s for c, s in zip(CLASSES_NAMES, scores)}}
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(out_csv, index=False)
        print(f"[{name}] {cid}: val PR-AUC {scores.mean():.4f} ({row['fit_minutes']} min)")
    return done


def best_config(name: str) -> dict:
    df = pd.read_csv(os.path.join(SWEEP_DIR, f"{name}.csv"))
    best = df.loc[df["val_pr_auc"].idxmax()]
    return {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in GRIDS[name][0]}


def evaluate(names: list[str] | None = None) -> pd.DataFrame:
    """Select, threshold on validation and score on test; writes ``results/``."""
    names = names or [n for n in GRIDS if os.path.exists(os.path.join(SWEEP_DIR, f"{n}.csv"))]
    ds = load_dataset()
    summary, thresholds_out = [], {}
    for name in names:
        params = best_config(name)
        probas = np.load(_proba_path(name, params))
        thr = best_f1_thresholds(ds.val.Y, probas["val"])
        val_df = per_class_metrics(ds.val.Y, probas["val"], thr, CLASSES_NAMES)
        test_df = per_class_metrics(ds.test.Y, probas["test"], thr, CLASSES_NAMES)
        test_df.round(4).to_csv(os.path.join(RESULTS_DIR, f"test_metrics_{name}.csv"))
        thresholds_out[name] = {"config": params, "thresholds": dict(zip(CLASSES_NAMES, thr.round(6).tolist()))}
        v, t = macro(val_df), macro(test_df)
        summary.append({"model": LABELS[name], "key": name, "config": config_id(params),
                        **{f"val_{m}": v[m] for m in ("pr_auc", "f1")},
                        **{f"test_{m}": t[m] for m in ("pr_auc", "f1", "precision", "recall")}})
    out = pd.DataFrame(summary).round(4)
    out.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "selected_configs.json"), "w") as fh:
        json.dump(thresholds_out, fh, indent=2)
    print(out.to_string(index=False))
    return out


def export_model(name: str = "boosting") -> str:
    """Copy the selected model plus its validation thresholds to ``models/``."""
    params = best_config(name)
    det = joblib.load(_proba_path(name, params).replace(".npz", ".joblib"))
    with open(os.path.join(RESULTS_DIR, "selected_configs.json")) as fh:
        thr = json.load(fh)[name]["thresholds"]
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "detector.joblib")
    joblib.dump({"detector": det, "thresholds": np.array([thr[c] for c in CLASSES_NAMES]),
                 "class_names": CLASSES_NAMES, "model": name, "config": params},
                path, compress=3)
    print(f"Exported {name} {config_id(params)} → {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path
