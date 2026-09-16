#!/usr/bin/env python3
"""Train a six-view classifier from human-reviewed inspection photos."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.db import SessionLocal  # noqa: E402
from backend.app.models import DatasetAnnotation  # noqa: E402
from scripts.train_retake_baseline import (  # noqa: E402
    DEFAULT_ONNX,
    FEATURE_LAYER,
    order_key,
    preprocess,
)


LABELS = (
    "left_front",
    "right_front",
    "left_rear",
    "right_rear",
    "interior_front",
    "interior_rear",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
DEFAULT_SOURCE = ROOT_DIR / "downloaded_images"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "angle_classifier"


def load_rows() -> list[DatasetAnnotation]:
    with SessionLocal() as db:
        return list(db.scalars(
            select(DatasetAnnotation).where(
                DatasetAnnotation.review_status.in_(("approved", "relabelled")),
                DatasetAnnotation.acceptable.is_(True),
                DatasetAnnotation.actual_label.in_(LABELS),
            ).order_by(DatasetAnnotation.source_path)
        ).all())


def extract_features(
    rows: list[DatasetAnnotation], source_dir: Path, onnx_path: Path
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    paths: list[str] = []
    for index, item in enumerate(rows, 1):
        image_path = source_dir / item.source_path
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"略過無法讀取的圖片：{item.source_path}")
            continue
        net.setInput(preprocess(image))
        features.append(net.forward(FEATURE_LAYER).reshape(-1))
        labels.append(LABEL_TO_ID[item.actual_label])
        groups.append(order_key(item.source_path))
        paths.append(item.source_path)
        if index % 25 == 0 or index == len(rows):
            print(f"擷取特徵 {index}/{len(rows)}", flush=True)
    return np.asarray(features, np.float32), np.asarray(labels, np.int32), groups, paths


def grouped_folds(
    groups: list[str], labels: np.ndarray, folds: int, seed: int
) -> list[set[str]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    if len(grouped) < folds:
        raise ValueError(f"只有 {len(grouped)} 個訂單，無法進行 {folds} 折驗證")

    rng = random.Random(seed)
    entries = list(grouped.items())
    rng.shuffle(entries)
    entries.sort(key=lambda pair: len(pair[1]), reverse=True)
    fold_groups = [set() for _ in range(folds)]
    fold_counts = [np.zeros(len(LABELS), dtype=np.int32) for _ in range(folds)]
    target = np.bincount(labels, minlength=len(LABELS)) / folds
    for group, indices in entries:
        counts = np.bincount(labels[indices], minlength=len(LABELS))
        smallest_size = min(int(values.sum()) for values in fold_counts)
        candidates = [
            candidate
            for candidate in range(folds)
            if int(fold_counts[candidate].sum()) == smallest_size
        ]
        fold = min(
            candidates,
            key=lambda candidate: (
                float(np.square((fold_counts[candidate] + counts) - target).sum()),
                candidate,
            ),
        )
        fold_groups[fold].add(group)
        fold_counts[fold] += counts
    return fold_groups


def normalize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((features - mean) / std).astype(np.float32), mean, std


def fit_svm(features: np.ndarray, labels: np.ndarray) -> tuple[cv2.ml_SVM, np.ndarray, np.ndarray]:
    normalized, mean, std = normalize_fit(features)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 3000, 1e-6))
    svm.train(normalized, cv2.ml.ROW_SAMPLE, labels)
    return svm, mean, std


def report(labels: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int32)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    per_class = {}
    recalls = []
    for index, label in enumerate(LABELS):
        support = int(matrix[index].sum())
        recall = float(matrix[index, index] / support) if support else 0.0
        recalls.append(recall)
        per_class[label] = {"recall": round(recall, 4), "support": support}
    return {
        "accuracy": round(float(np.mean(labels == predictions)), 4),
        "balanced_accuracy": round(float(np.mean(recalls)), 4),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "label_order": list(LABELS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="使用人工審核照片訓練六角度分類器")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    rows = load_rows()
    if len(rows) < 60:
        raise SystemExit(f"六角度人工標註只有 {len(rows)} 張，至少需要 60 張")
    features, labels, groups, paths = extract_features(rows, args.source_dir, args.onnx)
    missing = [label for label, class_id in LABEL_TO_ID.items() if class_id not in labels]
    if missing:
        raise SystemExit(f"缺少類別：{', '.join(missing)}")

    folds = grouped_folds(groups, labels, args.folds, args.seed)
    predictions = np.full(len(labels), -1, dtype=np.int32)
    group_array = np.asarray(groups)
    fold_reports = []
    for fold_index, test_groups in enumerate(folds, 1):
        test_mask = np.isin(group_array, list(test_groups))
        train_mask = ~test_mask
        svm, mean, std = fit_svm(features[train_mask], labels[train_mask])
        _, values = svm.predict(((features[test_mask] - mean) / std).astype(np.float32))
        predictions[test_mask] = values.reshape(-1).astype(np.int32)
        fold_report = report(labels[test_mask], predictions[test_mask])
        fold_report.update({
            "fold": fold_index,
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "test_orders": len(test_groups),
        })
        fold_reports.append(fold_report)

    overall = report(labels, predictions)
    final_svm, final_mean, final_std = fit_svm(features, labels)
    normalized = ((features - final_mean) / final_std).astype(np.float32)
    centroids = np.stack([
        normalized[labels == class_id].mean(axis=0)
        for class_id in range(len(LABELS))
    ]).astype(np.float32)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_svm.save(str(args.output_dir / "angle_svm.xml"))
    np.savez_compressed(
        args.output_dir / "angle_metadata.npz",
        mean=final_mean,
        std=final_std,
        centroids=centroids,
        labels=np.asarray(LABELS),
    )
    np.savez_compressed(
        args.output_dir / "training_features.npz",
        features=features,
        labels=labels,
        groups=group_array,
        paths=np.asarray(paths),
    )
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "six_view_production_candidate",
        "samples": len(labels),
        "orders": len(set(groups)),
        "class_counts": dict(Counter(LABELS[value] for value in labels)),
        "folds": args.folds,
        "split_strategy": "grouped_by_vehicle_and_order",
        "feature_extractor": str(args.onnx),
        "feature_layer": FEATURE_LAYER,
        "classifier": "mobilenetv2_features_plus_opencv_linear_svm",
        "overall": overall,
        "fold_reports": fold_reports,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
