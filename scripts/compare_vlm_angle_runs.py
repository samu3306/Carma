#!/usr/bin/env python3
"""Create paired comparison artifacts for two angle-evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from src.vlm.schemas import DETECTED_VIEWS


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    total = improved + regressed
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(improved, regressed) + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def main() -> int:
    parser = argparse.ArgumentParser(description="比較兩組 VLM 角度評估")
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--few", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    zero_rows = load_jsonl(args.zero / "predictions.jsonl")
    few_rows = load_jsonl(args.few / "predictions.jsonl")
    zero_by_path = {row["image_path"]: row for row in zero_rows}
    few_by_path = {row["image_path"]: row for row in few_rows}
    common = sorted(set(zero_by_path) & set(few_by_path))
    paired = []
    transitions = Counter()
    class_rows = []
    for path in common:
        zero = zero_by_path[path]
        few = few_by_path[path]
        truth = zero["ground_truth_view"]
        zero_correct = bool(zero.get("json_valid") and zero.get("detected_view") == truth)
        few_correct = bool(few.get("json_valid") and few.get("detected_view") == truth)
        state = (
            "both_correct" if zero_correct and few_correct else
            "improved" if not zero_correct and few_correct else
            "regressed" if zero_correct and not few_correct else
            "both_wrong"
        )
        transitions[state] += 1
        paired.append({
            "image_path": path, "ground_truth_view": truth,
            "zero_detected_view": zero.get("detected_view"),
            "few_detected_view": few.get("detected_view"),
            "zero_json_valid": zero.get("json_valid"), "few_json_valid": few.get("json_valid"),
            "zero_correct": zero_correct, "few_correct": few_correct,
            "transition": state, "zero_confidence": zero.get("confidence"),
            "few_confidence": few.get("confidence"),
        })
    for label in DETECTED_VIEWS:
        subset = [row for row in paired if row["ground_truth_view"] == label]
        class_rows.append({
            "class": label,
            "support": len(subset),
            "zero_accuracy": sum(row["zero_correct"] for row in subset) / len(subset) if subset else None,
            "few_accuracy": sum(row["few_correct"] for row in subset) / len(subset) if subset else None,
            "improved": sum(row["transition"] == "improved" for row in subset),
            "regressed": sum(row["transition"] == "regressed" for row in subset),
        })
    zero_metrics = json.loads((args.zero / "metrics.json").read_text(encoding="utf-8"))
    few_metrics = json.loads((args.few / "metrics.json").read_text(encoding="utf-8"))
    report = {
        "paired_samples": len(paired),
        "zero_shot": zero_metrics,
        "few_shot": few_metrics,
        "accuracy_delta": few_metrics["accuracy"] - zero_metrics["accuracy"],
        "macro_f1_delta": few_metrics["macro_f1"] - zero_metrics["macro_f1"],
        "paired_transitions": dict(transitions),
        "mcnemar_exact_p": exact_mcnemar_p(transitions["improved"], transitions["regressed"]),
        "warning": "other 類別測試樣本只有 4 張，該類指標不具穩定代表性",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for filename, rows in (("comparison_by_class.csv", class_rows), ("paired_predictions.csv", paired)):
        with (args.output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
