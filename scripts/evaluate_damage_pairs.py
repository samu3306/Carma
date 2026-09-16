#!/usr/bin/env python3
"""Evaluate Qwen damage-pair few-shot classification on official weak labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm.damage_pair import (  # noqa: E402
    QwenDamagePairAnalyzer,
    load_damage_references,
)
from src.vlm.qwen_vl_client import (  # noqa: E402
    ImageLoadError,
    InferenceError,
    QwenVLClient,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"pair_id", "plate", "before_path", "after_path", "label"}
    if not rows or required.difference(rows[0]):
        raise ValueError(f"manifest 欄位不完整：{path}")
    return rows


def select_balanced(
    rows: list[dict[str, str]], per_label: int | None, seed: int
) -> list[dict[str, str]]:
    if per_label is None:
        return rows
    rng = random.Random(seed)
    selected = []
    for label in ("new_damage", "no_new_damage"):
        candidates = [row for row in rows if row["label"] == label]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_label])
    rng.shuffle(selected)
    return selected


def metrics(rows: list[dict]) -> dict:
    successful = [row for row in rows if row.get("inference_success")]
    tp = sum(
        row["label"] == "new_damage" and row.get("decision") == "new_damage"
        for row in successful
    )
    fn = sum(
        row["label"] == "new_damage" and row.get("decision") != "new_damage"
        for row in successful
    )
    fp = sum(
        row["label"] == "no_new_damage" and row.get("decision") == "new_damage"
        for row in successful
    )
    tn = sum(
        row["label"] == "no_new_damage" and row.get("decision") == "no_new_damage"
        for row in successful
    )
    times = [
        float(row["inference_seconds"])
        for row in successful
        if row.get("inference_seconds") is not None
    ]
    return {
        "label_type": "official_folder_weak_label",
        "samples": len(rows),
        "successful": len(successful),
        "valid_json_rate": len(successful) / len(rows) if rows else 0,
        "decision_counts": dict(Counter(row.get("decision") for row in successful)),
        "true_positive": tp,
        "false_negative_or_uncertain": fn,
        "false_positive": fp,
        "true_negative": tn,
        "new_damage_recall": tp / (tp + fn) if tp + fn else 0,
        "new_damage_precision": tp / (tp + fp) if tp + fp else 0,
        "no_claim_false_alert_rate": fp / sum(
            row["label"] == "no_new_damage" for row in successful
        ) if successful else 0,
        "average_inference_seconds": statistics.fmean(times) if times else None,
    }


def write_outputs(output: Path, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = sorted({key for row in rows for key in row})
    with (output / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(row)
    summary = metrics(rows)
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failures = [
        row
        for row in rows
        if not row.get("inference_success") or row.get("decision") != row["label"]
    ]
    with (output / "review_queue.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in failures:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/damage/test_pairs.csv")
    )
    parser.add_argument(
        "--few-shot-manifest",
        type=Path,
        default=Path("data/damage/few_shot_pairs.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/qwen_damage_few_shot")
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        test_rows = read_rows(args.manifest)
        references = load_damage_references(args.few_shot_manifest)
    except (OSError, ValueError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 2
    selected = select_balanced(test_rows, args.per_label, args.seed)
    output_rows = []
    analyzer = QwenDamagePairAnalyzer(QwenVLClient(args.model), references)
    for index, source in enumerate(selected, 1):
        row: dict = dict(source)
        try:
            result = analyzer.infer_pair(
                Path(source["before_path"]), Path(source["after_path"])
            )
            row.update(result.prediction or {})
            row.update({
                "json_valid": result.json_valid,
                "inference_success": result.json_valid and result.error is None,
                "inference_seconds": result.inference_seconds,
                "gpu_peak_memory_mb": result.gpu_peak_memory_mb,
                "error": result.error,
                "raw_model_output": result.raw_output,
            })
        except (ImageLoadError, InferenceError, OSError) as error:
            row.update({
                "json_valid": False,
                "inference_success": False,
                "inference_seconds": None,
                "gpu_peak_memory_mb": None,
                "error": f"{type(error).__name__}: {error}",
                "raw_model_output": None,
            })
        output_rows.append(row)
        print(
            f"[{index}/{len(selected)}] {source['pair_id']} "
            f"label={source['label']} decision={row.get('decision')} "
            f"success={row['inference_success']}",
            flush=True,
        )
    write_outputs(args.output, output_rows)
    print(json.dumps(metrics(output_rows), ensure_ascii=False, indent=2))
    print(f"輸出：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
