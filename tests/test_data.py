import numpy as np

from sed.data import aggregate_labels, split_collectors


def test_majority_vote_needs_more_than_half_of_annotators():
    # 1 frame, 3 classes, 2 annotators: both agree / one agrees / score too low
    ann = np.array([[[0.9, 0.8], [0.9, 0.0], [0.2, 0.2]]])
    assert aggregate_labels(ann).tolist() == [[1, 0, 0]]


def test_collector_split_is_disjoint_and_complete():
    collectors = np.array([f"c{i}" for i in range(100)] * 3)
    train, val, test = split_collectors(collectors, seed=0)
    assert not (train & val or train & test or val & test)
    assert len(train | val | test) == 100
    assert (len(train), len(val), len(test)) == (70, 15, 15)
