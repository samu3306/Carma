#!/usr/bin/env python3
"""Build order-isolated few-shot references and a human-verified angle test set."""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm.schemas import DETECTED_VIEWS  # noqa: E402


REFERENCE_FIELDS = (
    "image_path", "human_verified_view", "order_number", "car_no",
    "review_status", "reviewer", "source_path", "original_label", "notes",
)
EVAL_FIELDS = (
    "image_path", "order_number", "car_no", "image_type", "ground_truth_view",
    "original_filename", "human_review_status", "reviewer", "notes",
)


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def load_candidates(database: Path, image_root: Path) -> list[dict[str, str]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    records = connection.execute(
        """
        SELECT source_path, image_type, original_label, actual_label,
               review_status, reviewer, notes
        FROM dataset_annotations
        WHERE review_status IN ('approved', 'relabelled')
          AND acceptable = 1
          AND actual_label IS NOT NULL
        ORDER BY source_path
        """
    ).fetchall()
    connection.close()
    candidates = []
    for record in records:
        parts = Path(record["source_path"]).parts
        if len(parts) < 3 or record["actual_label"] not in DETECTED_VIEWS:
            continue
        path = (image_root / record["source_path"]).resolve()
        if not path.is_file() or not valid_image(path):
            continue
        candidates.append({
            "image_path": str(path),
            "human_verified_view": record["actual_label"],
            "order_number": parts[-2],
            "car_no": parts[-3],
            "image_type": "" if record["image_type"] is None else str(record["image_type"]),
            "review_status": record["review_status"],
            "reviewer": record["reviewer"] or "",
            "source_path": record["source_path"],
            "original_label": record["original_label"],
            "original_filename": Path(record["source_path"]).name,
            "notes": record["notes"] or "",
        })
    return candidates


def choose_references(
    candidates: list[dict[str, str]], per_class: int, seed: int
) -> tuple[list[dict[str, str]], set[str]]:
    rng = random.Random(seed)
    by_order_label: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        by_order_label[row["order_number"]][row["human_verified_view"]].append(row)
    remaining = {label: per_class for label in DETECTED_VIEWS}
    selected: list[dict[str, str]] = []
    selected_orders: set[str] = set()
    order_tiebreak = {order: rng.random() for order in by_order_label}
    while any(value > 0 for value in remaining.values()):
        choices = []
        for order, labels in by_order_label.items():
            if order in selected_orders:
                continue
            score = sum(1 for label in labels if remaining.get(label, 0) > 0)
            if score:
                choices.append((-score, order_tiebreak[order], order))
        if not choices:
            break
        _, _, order = min(choices)
        selected_orders.add(order)
        for label in DETECTED_VIEWS:
            if remaining[label] <= 0 or label not in by_order_label[order]:
                continue
            rows = sorted(
                by_order_label[order][label],
                key=lambda row: (row["review_status"] != "approved", row["image_path"]),
            )
            selected.append(rows[0])
            remaining[label] -= 1
    missing = {label: count for label, count in remaining.items() if count}
    if missing:
        raise ValueError(f"無法建立足量 few-shot 參考：{missing}")
    return selected, selected_orders


def choose_eval(
    candidates: list[dict[str, str]], excluded_orders: set[str], per_class: int, seed: int
) -> list[dict[str, str]]:
    rng = random.Random(seed + 1)
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        if row["order_number"] not in excluded_orders:
            grouped[row["human_verified_view"]][row["order_number"]].append(row)
    selected = []
    for label in DETECTED_VIEWS:
        orders = sorted(grouped[label])
        rng.shuffle(orders)
        target = min(per_class, len(orders))
        if target < per_class:
            print(f"警告：{label} 只有 {target} 個不重疊人工驗證訂單", file=sys.stderr)
        for order in orders[:target]:
            choices = sorted(grouped[label][order], key=lambda row: row["image_path"])
            selected.append(rng.choice(choices))
    return selected


def write_manifests(
    references: list[dict[str, str]], evaluation: list[dict[str, str]],
    reference_output: Path, eval_output: Path,
) -> None:
    reference_output.parent.mkdir(parents=True, exist_ok=True)
    with reference_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(references)
    eval_rows = []
    for row in evaluation:
        eval_rows.append({
            "image_path": row["image_path"], "order_number": row["order_number"],
            "car_no": row["car_no"], "image_type": row["image_type"],
            "ground_truth_view": row["human_verified_view"],
            "original_filename": row["original_filename"],
            "human_review_status": row["review_status"], "reviewer": row["reviewer"],
            "notes": row["notes"],
        })
    with eval_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(eval_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="建立按訂單隔離的人工驗證角度測試集")
    parser.add_argument("--database", type=Path, default=Path("carma.db"))
    parser.add_argument("--image-root", type=Path, default=Path("downloaded_images"))
    parser.add_argument("--reference-output", type=Path, default=Path("data/eval/human_few_shot_reference_manifest.csv"))
    parser.add_argument("--eval-output", type=Path, default=Path("data/eval/human_verified_angle_eval_manifest.csv"))
    parser.add_argument("--references-per-class", type=int, default=3)
    parser.add_argument("--eval-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        candidates = load_candidates(args.database, args.image_root)
        references, reference_orders = choose_references(candidates, args.references_per_class, args.seed)
        evaluation = choose_eval(candidates, reference_orders, args.eval_per_class, args.seed)
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 1
    write_manifests(references, evaluation, args.reference_output, args.eval_output)
    reference_counts = Counter(row["human_verified_view"] for row in references)
    eval_counts = Counter(row["human_verified_view"] for row in evaluation)
    eval_orders = {row["order_number"] for row in evaluation}
    print(f"參考集：{len(references)} 張 / {len(reference_orders)} 筆訂單")
    print(f"測試集：{len(evaluation)} 張 / {len(eval_orders)} 筆訂單")
    print(f"訂單重疊：{len(reference_orders & eval_orders)}")
    for label in DETECTED_VIEWS:
        print(f"{label}: reference={reference_counts[label]} eval={eval_counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
