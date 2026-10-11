"""Frame-level dataset: one cached matrix for all recordings, split by collector.

Reading 3,656 ``.npz`` files takes a while, so ``build_cache`` stacks every
frame once into ``cache/frames.npz``. ``load_dataset`` then returns the
train / validation / test splits from that cache.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    CACHE_PATH, CLASSES_NAMES, DATASET_DIR, SEED, TRAIN_FRACTION, VAL_FRACTION,
)
from .loader import get_features_and_targets


@dataclass
class Split:
    X: np.ndarray          # (N, 960) float32 features
    Y: np.ndarray          # (N, C) uint8 labels
    file_idx: np.ndarray   # (N,) index into ``Dataset.files``

    @property
    def prevalence(self) -> np.ndarray:
        return self.Y.mean(axis=0)


@dataclass
class Dataset:
    train: Split
    val: Split
    test: Split
    files: np.ndarray      # file names (e.g. "003075.npz"), indexed by file_idx


def build_cache(dataset_dir: str = DATASET_DIR, cache_path: str = CACHE_PATH) -> None:
    """Read every recording once and store all frames in a single ``.npz``."""
    paths = sorted(glob.glob(os.path.join(dataset_dir, "audio_features", "*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No .npz files under {dataset_dir}/audio_features — see the README "
            "for how to obtain the MLPC2026 dataset."
        )
    meta = pd.read_csv(os.path.join(dataset_dir, "metadata.csv")).set_index("filename")

    X, Y, file_idx, collectors = [], [], [], []
    for i, path in enumerate(paths):
        x, y = get_features_and_targets(path)
        X.append(x)
        Y.append(y)
        file_idx.append(np.full(len(x), i, dtype=np.int32))
        collectors.append(meta.loc[os.path.basename(path)[:-4] + ".wav", "collector_id"])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X=np.vstack(X), Y=np.vstack(Y), file_idx=np.concatenate(file_idx),
        files=np.array([os.path.basename(p) for p in paths]),
        collectors=np.array(collectors),
    )
    print(f"Cached {sum(len(x) for x in X):,} frames from {len(paths):,} files → {cache_path}")


def split_collectors(collectors: np.ndarray, seed: int = SEED):
    """Assign each collector to train / val / test (by fraction of collectors)."""
    unique = np.array(sorted(set(collectors)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = round(len(unique) * TRAIN_FRACTION)
    n_val = round(len(unique) * VAL_FRACTION)
    return (set(unique[:n_train]), set(unique[n_train:n_train + n_val]),
            set(unique[n_train + n_val:]))


def load_dataset(cache_path: str = CACHE_PATH, seed: int = SEED) -> Dataset:
    """Load the frame cache and split it so no collector spans two splits."""
    if not os.path.exists(cache_path):
        build_cache(cache_path=cache_path)
    d = np.load(cache_path, allow_pickle=True)
    file_collector = d["collectors"]
    groups = split_collectors(file_collector, seed)

    def subset(group: set) -> Split:
        files_in = np.isin(file_collector, list(group))
        rows = files_in[d["file_idx"]]
        return Split(d["X"][rows], d["Y"][rows], d["file_idx"][rows])

    return Dataset(*(subset(g) for g in groups), files=d["files"])


def describe(ds: Dataset) -> pd.DataFrame:
    """Frames, files and per-class prevalence for each split."""
    rows = {}
    for name in ("train", "val", "test"):
        s = getattr(ds, name)
        rows[name] = {"frames": len(s.Y), "files": len(np.unique(s.file_idx)),
                      **dict(zip(CLASSES_NAMES, s.prevalence.round(4)))}
    return pd.DataFrame(rows)
