from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sqlalchemy import select

from backend.app.db import SessionLocal
from backend.app.models import DatasetAnnotation


DEFAULT_SOURCE = ROOT_DIR / "downloaded_images"
DEFAULT_ONNX = ROOT_DIR / "models" / "mobilenetv2-12.onnx"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "retake_baseline"
FEATURE_LAYER = "onnx_node!Reshape_103"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def order_key(source_path: str) -> str:
    return "/".join(source_path.split("/")[:2])


def preprocess(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = 256.0 / min(height, width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    top = max(0, (resized.shape[0] - 224) // 2)
    left = max(0, (resized.shape[1] - 224) // 2)
    crop = resized[top:top + 224, left:left + 224]
    if crop.shape[:2] != (224, 224):
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    return np.transpose(normalized, (2, 0, 1))[None, ...]


def load_rows() -> list[DatasetAnnotation]:
    with SessionLocal() as db:
        return list(db.scalars(
            select(DatasetAnnotation).where(
                DatasetAnnotation.review_status.in_(("approved", "relabelled", "invalid"))
            ).order_by(DatasetAnnotation.source_path)
        ).all())


def extract_features(rows: list[DatasetAnnotation], source_dir: Path, onnx_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    paths: list[str] = []
    skipped = 0
    for index, item in enumerate(rows, 1):
        image_path = source_dir / item.source_path
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            print(f"略過無法讀取的圖片：{item.source_path}")
            continue
        net.setInput(preprocess(image))
        vector = net.forward(FEATURE_LAYER).reshape(-1)
        features.append(vector)
        labels.append(1 if item.review_status == "invalid" else 0)
        groups.append(order_key(item.source_path))
        paths.append(item.source_path)
        if index % 25 == 0 or index == len(rows):
            print(f"擷取特徵 {index}/{len(rows)}")
    if skipped:
        print(f"共略過 {skipped} 張無法讀取的圖片")
    return np.asarray(features, np.float32), np.asarray(labels, np.int32), groups, paths


def grouped_folds(groups: list[str], labels: np.ndarray, folds: int, seed: int) -> list[set[str]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    if len(grouped) < folds:
        raise ValueError(f"只有 {len(grouped)} 個訂單，無法進行 {folds} 折分組驗證")

    rng = random.Random(seed)
    entries = list(grouped.items())
    rng.shuffle(entries)
    entries.sort(key=lambda pair: len(pair[1]), reverse=True)
    result = [set() for _ in range(folds)]
    fold_sizes = [0] * folds
    fold_positive = [0] * folds
    positive_rate = float(labels.mean())

    for group, indices in entries:
        group_size = len(indices)
        group_positive = int(labels[indices].sum())
        smallest_size = min(fold_sizes)
        candidates = [fold for fold in range(folds) if fold_sizes[fold] == smallest_size]
        target_fold = min(
            candidates,
            key=lambda fold: abs(
                (fold_positive[fold] + group_positive) / (fold_sizes[fold] + group_size) - positive_rate
            ),
        )
        result[target_fold].add(group)
        fold_sizes[target_fold] += group_size
        fold_positive[target_fold] += group_positive
    return result


def fit_svm(features: np.ndarray, labels: np.ndarray) -> tuple[cv2.ml_SVM, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = ((features - mean) / std).astype(np.float32)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 2000, 1e-6))
    svm.train(normalized, cv2.ml.ROW_SAMPLE, labels.astype(np.int32))
    return svm, mean.astype(np.float32), std.astype(np.float32)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "accuracy": round((tp + tn) / max(len(labels), 1), 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "retake_precision": round(precision, 4),
        "retake_recall": round(recall, 4),
        "acceptable_recall": round(specificity, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="以 MobileNetV2 特徵建立本機接受／重拍輕量基準")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    rows = load_rows()
    if len(rows) < 40:
        raise SystemExit("可用人工標註少於 40 張，暫不執行訓練")
    if not args.onnx.is_file():
        raise SystemExit(f"找不到 MobileNetV2 ONNX：{args.onnx}")

    features, labels, groups, paths = extract_features(rows, args.source_dir, args.onnx)
    if len(np.unique(labels)) != 2:
        raise SystemExit("資料必須同時包含可接受與需要重拍照片")

    fold_groups = grouped_folds(groups, labels, args.folds, args.seed)
    predictions = np.full(len(labels), -1, dtype=np.int32)
    fold_reports = []
    group_array = np.asarray(groups)
    for fold_index, test_groups in enumerate(fold_groups, 1):
        test_mask = np.isin(group_array, list(test_groups))
        train_mask = ~test_mask
        svm, mean, std = fit_svm(features[train_mask], labels[train_mask])
        _, predicted = svm.predict(((features[test_mask] - mean) / std).astype(np.float32))
        predictions[test_mask] = predicted.reshape(-1).astype(np.int32)
        report = metrics(labels[test_mask], predictions[test_mask])
        report.update({
            "fold": fold_index,
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "test_orders": len(test_groups),
        })
        fold_reports.append(report)

    overall = metrics(labels, predictions)
    final_svm, final_mean, final_std = fit_svm(features, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_svm.save(str(args.output_dir / "retake_svm.xml"))
    np.savez_compressed(args.output_dir / "feature_scaler.npz", mean=final_mean, std=final_std)
    np.savez_compressed(
        args.output_dir / "training_features.npz",
        features=features, labels=labels, groups=group_array, paths=np.asarray(paths),
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "baseline_only_not_production",
        "positive_class": "retake",
        "samples": len(labels),
        "acceptable": int(np.sum(labels == 0)),
        "retake": int(np.sum(labels == 1)),
        "orders": len(set(groups)),
        "folds": args.folds,
        "split_strategy": "grouped_by_vehicle_and_order",
        "feature_extractor": str(args.onnx),
        "feature_layer": FEATURE_LAYER,
        "classifier": "opencv_linear_svm",
        "preprocessing": "resize_shorter_256_center_crop_224_imagenet_normalization",
        "overall": overall,
        "fold_reports": fold_reports,
        "deployment_recommendation": "do_not_enable_until_reviewed",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n基準模型與報告已寫入：{args.output_dir}")


if __name__ == "__main__":
    main()
