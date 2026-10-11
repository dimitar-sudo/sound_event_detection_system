import os

import numpy as np
import pytest

from sed.config import CLASSES_NAMES, EXAMPLES_DIR, MODEL_DIR
from sed.inference import frames_to_events
from sed.models import build


def test_detector_returns_one_probability_per_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 960)).astype(np.float32)
    Y = (rng.random((200, len(CLASSES_NAMES))) < 0.3).astype(np.uint8)
    proba = build("baseline", {}).fit(X, Y).predict_proba(X[:7])
    assert proba.shape == (7, len(CLASSES_NAMES))
    np.testing.assert_allclose(proba[0], Y.mean(0), rtol=1e-5)


def test_consecutive_frames_merge_into_one_event():
    pred = np.array([[1], [1], [0], [1]])
    events = frames_to_events(pred, np.array([0.0, 0.5, 1.0, 1.5]), ["bell"])
    assert events[["onset_s", "offset_s"]].values.tolist() == [[0.0, 1.5], [1.5, 2.5]]


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "detector.joblib")),
                    reason="shipped model not present")
def test_shipped_model_runs_on_example_file():
    from sed.inference import predict_file
    example = sorted(f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".npz"))[0]
    proba, pred, _ = predict_file(os.path.join(EXAMPLES_DIR, example))
    assert proba.shape == pred.shape and proba.shape[1] == len(CLASSES_NAMES)
