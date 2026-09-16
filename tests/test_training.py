from __future__ import annotations

import numpy as np

from scripts.train_retake_baseline import grouped_folds, metrics, preprocess


def test_grouped_folds_do_not_leak_orders_and_stay_balanced():
    groups = [f"order-{index}" for index in range(10) for _ in range(3 + index % 3)]
    labels = np.asarray([(index // 4) % 2 for index in range(len(groups))], dtype=np.int32)
    folds = grouped_folds(groups, labels, folds=5, seed=123)

    assert set.union(*folds) == set(groups)
    for index, fold in enumerate(folds):
        assert all(fold.isdisjoint(other) for other in folds[index + 1:])
    sizes = [sum(group in fold for group in groups) for fold in folds]
    assert max(sizes) - min(sizes) <= 5


def test_preprocess_and_binary_metrics():
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    blob = preprocess(image)
    assert blob.shape == (1, 3, 224, 224)
    assert blob.dtype == np.float32
    assert np.isfinite(blob).all()

    result = metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        np.asarray([0, 1, 1, 1], dtype=np.int32),
    )
    assert result["accuracy"] == 0.75
    assert result["retake_recall"] == 1.0
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
