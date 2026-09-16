#!/usr/bin/env python3
"""Resumable one-by-one Qwen3-VL angle and quality evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm.qwen_vl_client import (  # noqa: E402
    ImageLoadError,
    InferenceError,
    ModelLoadError,
    QwenVLClient,
)
from src.vlm.schemas import DETECTED_VIEWS  # noqa: E402


PREDICTION_FIELDS = (
    "image_path", "order_number", "car_no", "image_type", "ground_truth_view",
    "original_filename", "detected_view", "vehicle_visible", "plate_visible",
    "quality_passed", "quality_issues", "confidence", "explanation",
    "retake_instruction", "json_valid", "inference_success", "inference_seconds",
    "gpu_peak_memory_mb", "error", "raw_model_output",
)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "order_number", "car_no", "image_type", "ground_truth_view", "original_filename"}
    if not rows:
        raise ValueError("manifest 沒有資料")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"manifest 缺少欄位：{', '.join(sorted(missing))}")
    return rows


def load_completed(jsonl_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    results: list[dict[str, Any]] = []
    if not jsonl_path.is_file():
        return results, set()
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"警告：忽略 predictions.jsonl 第 {line_number} 行：{error}", file=sys.stderr)
                continue
            results.append(row)
    return results, {str(row.get("image_path")) for row in results if row.get("image_path")}


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def calculate_metrics(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], list[list[int]]]:
    all_rows = list(rows)
    labels = list(DETECTED_VIEWS)
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    classified = []
    for row in all_rows:
        truth, predicted = row.get("ground_truth_view"), row.get("detected_view")
        if row.get("json_valid") and truth in index and predicted in index:
            matrix[index[truth]][index[predicted]] += 1
            classified.append(row)

    per_class: dict[str, dict[str, float | int]] = {}
    for label, position in index.items():
        tp = matrix[position][position]
        fp = sum(matrix[row][position] for row in range(len(labels)) if row != position)
        fn = sum(matrix[position][column] for column in range(len(labels)) if column != position)
        support = sum(matrix[position])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    total = len(all_rows)
    inference_times = [float(row["inference_seconds"]) for row in all_rows if row.get("inference_seconds") is not None]
    memories = [float(row["gpu_peak_memory_mb"]) for row in all_rows if row.get("gpu_peak_memory_mb") is not None]
    classified_count = len(classified)
    metrics = {
        "samples_attempted": total,
        "samples_classified": classified_count,
        "accuracy": sum(matrix[i][i] for i in range(len(labels))) / classified_count if classified_count else 0.0,
        "macro_precision": statistics.fmean(item["precision"] for item in per_class.values()),
        "macro_recall": statistics.fmean(item["recall"] for item in per_class.values()),
        "macro_f1": statistics.fmean(item["f1"] for item in per_class.values()),
        "per_class": per_class,
        "valid_json_rate": sum(bool(row.get("json_valid")) for row in all_rows) / total if total else 0.0,
        "inference_success_rate": sum(bool(row.get("inference_success")) for row in all_rows) / total if total else 0.0,
        "average_inference_seconds": statistics.fmean(inference_times) if inference_times else None,
        "p50_inference_seconds": percentile(inference_times, 0.50),
        "p95_inference_seconds": percentile(inference_times, 0.95),
        "average_gpu_peak_memory_mb": statistics.fmean(memories) if memories else None,
        "peak_gpu_memory_mb": max(memories) if memories else None,
    }
    return metrics, matrix


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    fields = tuple(fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(row)


def write_derived_outputs(output: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output / "predictions.csv", rows, PREDICTION_FIELDS)
    metrics, matrix = calculate_metrics(rows)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    labels = list(DETECTED_VIEWS)
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ground_truth\\predicted", *labels])
        for label, counts in zip(labels, matrix):
            writer.writerow([label, *counts])
    review_rows = [
        row for row in rows
        if not row.get("inference_success") or row.get("detected_view") != row.get("ground_truth_view")
    ]
    write_csv(output / "failure_cases.csv", review_rows, PREDICTION_FIELDS)


def evaluate_rows(
    manifest_rows: list[dict[str, str]],
    output: Path,
    client: Any,
    resume: bool,
    limit: int | None,
    on_saved: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    jsonl_path = output / "predictions.jsonl"
    existing, completed = load_completed(jsonl_path) if resume else ([], set())
    if not resume and jsonl_path.exists():
        raise FileExistsError(f"結果已存在：{jsonl_path}；請改用新輸出目錄或加 --resume")
    selected = manifest_rows[:limit] if limit is not None else manifest_rows
    results = existing
    for position, manifest_row in enumerate(selected, 1):
        if manifest_row["image_path"] in completed:
            continue
        row: dict[str, Any] = dict(manifest_row)
        try:
            result = client.infer(Path(manifest_row["image_path"]))
            row.update(result.prediction or {})
            row.update({
                "json_valid": result.json_valid,
                "inference_success": result.json_valid and result.error is None,
                "inference_seconds": result.inference_seconds,
                "gpu_peak_memory_mb": result.gpu_peak_memory_mb,
                "error": result.error,
                "raw_model_output": result.raw_output,
            })
        except (ModelLoadError, ImageLoadError, InferenceError, OSError) as error:
            row.update({
                "json_valid": False, "inference_success": False,
                "inference_seconds": None, "gpu_peak_memory_mb": None,
                "error": f"{type(error).__name__}: {error}", "raw_model_output": None,
            })
        append_jsonl(jsonl_path, row)
        results.append(row)
        if on_saved:
            on_saved(row)
        print(f"[{position}/{len(selected)}] {manifest_row['image_path']} success={row['inference_success']}")
    write_derived_outputs(output, results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="執行可續跑的 Qwen3-VL 角度 zero-shot 評估")
    parser.add_argument("--manifest", type=Path, default=Path("data/eval/angle_eval_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen3_vl_angle_baseline"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必須大於 0")
    try:
        rows = load_manifest(args.manifest)
    except (OSError, ValueError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model,
        "manifest": str(args.manifest.resolve()),
        "output": str(args.output.resolve()),
        "limit": args.limit,
        "resume": args.resume,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "attention": "sdpa",
        "do_sample": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        results = evaluate_rows(
            rows, args.output, QwenVLClient(args.model, args.max_new_tokens), args.resume, args.limit
        )
    except FileExistsError as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 2
    successes = sum(bool(row.get("inference_success")) for row in results)
    print(f"完成：{len(results)} 筆，成功 {successes} 筆；輸出：{args.output}")
    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
