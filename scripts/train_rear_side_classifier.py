#!/usr/bin/env python3
"""Train and cross-validate a rear-side classifier on frozen Qwen3-VL image features."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vlm.qwen_vl_client import QwenVLClient  # noqa: E402


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DATABASE = ROOT / "carma.db"
IMAGE_ROOT = ROOT / "downloaded_images"
OUTPUT_DIR = ROOT / "outputs" / "rear_side_classifier"
LABEL_TO_VALUE = {"left_rear": -1.0, "right_rear": 1.0}
RIDGE_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def load_pairs() -> list[dict[str, Path]]:
    connection = sqlite3.connect(DATABASE)
    rows = connection.execute(
        """
        SELECT source_path, actual_label
        FROM dataset_annotations
        WHERE acceptable = 1
          AND review_status = ?
          AND actual_label = original_label
          AND actual_label IN (?, ?)
        ORDER BY source_path
        """,
        ("approved", "left_rear", "right_rear"),
    ).fetchall()
    connection.close()
    grouped: dict[str, dict[str, Path]] = {}
    for source_path, label in rows:
        parts = Path(source_path).parts
        path = (IMAGE_ROOT / source_path).resolve()
        if len(parts) >= 3 and path.is_file():
            grouped.setdefault("/".join(parts[:2]), {})[label] = path
    return [
        {"group": group, **images}
        for group, images in grouped.items()
        if LABEL_TO_VALUE.keys() <= images.keys()
    ]


def extract_feature(client: QwenVLClient, path: Path) -> torch.Tensor:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    inputs = client.processor.image_processor(images=[image], return_tensors="pt")
    device = next(client.model.parameters()).device
    pixel_values = inputs["pixel_values"].to(device)
    image_grid_thw = inputs["image_grid_thw"].to(device)
    with torch.inference_mode():
        image_embeds, _ = client.model.get_image_features(pixel_values, image_grid_thw)
    feature = image_embeds[0].float().mean(dim=0).cpu()
    return torch.nn.functional.normalize(feature, dim=0)


def fit_ridge(features: torch.Tensor, targets: torch.Tensor, ridge: float) -> torch.Tensor:
    design = torch.cat([features, torch.ones(features.shape[0], 1)], dim=1)
    gram = design @ design.T
    dual = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0]),
        targets,
    )
    return design.T @ dual


def predict(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    design = torch.cat([features, torch.ones(features.shape[0], 1)], dim=1)
    return torch.where(design @ weights >= 0, 1.0, -1.0)


def main() -> int:
    pairs = load_pairs()
    if len(pairs) < 4:
        raise RuntimeError(f"成對人工核准資料不足：{len(pairs)}")
    client = QwenVLClient(MODEL_NAME)
    client.load()

    feature_rows: list[torch.Tensor] = []
    targets: list[float] = []
    groups: list[str] = []
    paths: list[str] = []
    for pair in pairs:
        for label in ("left_rear", "right_rear"):
            path = pair[label]
            print(f"extract {label}: {path}", flush=True)
            feature_rows.append(extract_feature(client, path))
            targets.append(LABEL_TO_VALUE[label])
            groups.append(str(pair["group"]))
            paths.append(str(path))

    features = torch.stack(feature_rows)
    target_tensor = torch.tensor(targets)
    unique_groups = sorted(set(groups))
    scores: dict[float, dict[str, object]] = {}
    for ridge in RIDGE_VALUES:
        predictions = torch.empty_like(target_tensor)
        for held_out in unique_groups:
            train_mask = torch.tensor([group != held_out for group in groups])
            test_mask = ~train_mask
            weights = fit_ridge(features[train_mask], target_tensor[train_mask], ridge)
            predictions[test_mask] = predict(features[test_mask], weights)
        correct = predictions == target_tensor
        left_mask = target_tensor < 0
        right_mask = target_tensor > 0
        scores[ridge] = {
            "accuracy": float(correct.float().mean()),
            "left_recall": float(correct[left_mask].float().mean()),
            "right_recall": float(correct[right_mask].float().mean()),
            "predictions": predictions.tolist(),
        }
        print(
            f"ridge={ridge:g} accuracy={scores[ridge]['accuracy']:.3f} "
            f"left={scores[ridge]['left_recall']:.3f} right={scores[ridge]['right_recall']:.3f}"
        )

    best_ridge = max(
        RIDGE_VALUES,
        key=lambda value: (
            scores[value]["accuracy"],
            min(scores[value]["left_recall"], scores[value]["right_recall"]),
        ),
    )
    final_weights = fit_ridge(features, target_tensor, best_ridge)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "weights": final_weights,
            "ridge": best_ridge,
            "feature_dimension": features.shape[1],
            "training_groups": unique_groups,
        },
        OUTPUT_DIR / "classifier.pt",
    )
    metrics = {
        "pairs": len(pairs),
        "samples": len(targets),
        "class_counts": Counter("left_rear" if value < 0 else "right_rear" for value in targets),
        "validation": {str(key): value for key, value in scores.items()},
        "best_ridge": best_ridge,
        "paths": paths,
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"saved: {OUTPUT_DIR / 'classifier.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
