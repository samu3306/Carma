#!/usr/bin/env python3
"""Evaluate frozen DINOv2 ViT-S/14 features with an OpenCV linear SVM."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.train_angle_classifier import (  # noqa: E402
    LABELS,
    LABEL_TO_ID,
    DEFAULT_SOURCE,
    fit_svm,
    grouped_folds,
    load_rows,
    report,
)
from scripts.train_retake_baseline import order_key  # noqa: E402


DEFAULT_MODEL = "facebook/dinov2-small"
DEFAULT_CACHE = ROOT_DIR / "models" / "huggingface"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "dinov2_angle_classifier"


def resolve_device(torch_module, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("指定使用 CUDA，但目前 .venv 的 PyTorch 無法使用 CUDA")
    return requested


def extract_features(
    rows,
    source_dir: Path,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    requested_device: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], str]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = resolve_device(torch, requested_device)
    processor = AutoImageProcessor.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=False
    )
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    model.eval()

    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    paths: list[str] = []
    valid_items: list[tuple[object, Image.Image]] = []

    def flush_batch() -> None:
        if not valid_items:
            return
        images = [image for _, image in valid_items]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        vectors = pooled.detach().float().cpu().numpy()
        for (item, image), vector in zip(valid_items, vectors):
            features.append(vector)
            labels.append(LABEL_TO_ID[item.actual_label])
            groups.append(order_key(item.source_path))
            paths.append(item.source_path)
            image.close()
        valid_items.clear()

    for index, item in enumerate(rows, 1):
        image_path = source_dir / item.source_path
        try:
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB").copy()
        except (OSError, UnidentifiedImageError):
            print(f"略過無法讀取的圖片：{item.source_path}")
            continue
        valid_items.append((item, image))
        if len(valid_items) >= batch_size:
            flush_batch()
        if index % 25 == 0 or index == len(rows):
            print(f"DINOv2 特徵 {index}/{len(rows)}", flush=True)
    flush_batch()
    return (
        np.asarray(features, np.float32),
        np.asarray(labels, np.int32),
        groups,
        paths,
        device,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="以 frozen DINOv2 ViT-S/14 特徵建立六角度 Linear SVM 實驗"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size 必須大於 0")

    rows = load_rows()
    if len(rows) < 60:
        raise SystemExit(f"六角度人工標註只有 {len(rows)} 張，至少需要 60 張")
    features, labels, groups, paths, device = extract_features(
        rows,
        args.source_dir,
        args.model,
        args.cache_dir,
        args.batch_size,
        args.device,
    )
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
        model_name=np.asarray(args.model),
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
        "purpose": "dinov2_s14_linear_svm_experiment_not_deployed",
        "samples": len(labels),
        "orders": len(set(groups)),
        "class_counts": dict(Counter(LABELS[value] for value in labels)),
        "folds": args.folds,
        "split_strategy": "grouped_by_vehicle_and_order",
        "feature_extractor": args.model,
        "feature_dimension": int(features.shape[1]),
        "classifier": "frozen_dinov2_vits14_features_plus_opencv_linear_svm",
        "device": device,
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
