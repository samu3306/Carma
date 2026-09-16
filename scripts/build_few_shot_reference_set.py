#!/usr/bin/env python3
"""Build a leakage-free few-shot manifest from yesterday's human reviews."""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

LABELS = ("left_front", "right_front", "left_rear", "right_rear", "interior_front", "interior_rear", "other")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("carma.db"))
    parser.add_argument("--image-root", type=Path, default=Path("downloaded_images"))
    parser.add_argument("--exclude-manifest", type=Path, default=Path("data/eval/angle_eval_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/few_shot_reference_manifest.csv"))
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.exclude_manifest.open(encoding="utf-8-sig", newline="") as handle:
        excluded = {str(Path(row["image_path"]).resolve()) for row in csv.DictReader(handle)}
    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    records = connection.execute("""
        SELECT source_path, original_label, actual_label, review_status, reviewer, notes
        FROM dataset_annotations
        WHERE review_status IN ('approved', 'relabelled') AND acceptable = 1
        ORDER BY source_path
    """).fetchall()
    connection.close()
    grouped = defaultdict(list)
    for record in records:
        path = (args.image_root / record["source_path"]).resolve()
        if record["actual_label"] not in LABELS or str(path) in excluded or not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except OSError:
            continue
        grouped[record["actual_label"]].append({
            "image_path": str(path), "human_verified_view": record["actual_label"],
            "review_status": record["review_status"], "reviewer": record["reviewer"] or "",
            "source_path": record["source_path"], "original_label": record["original_label"],
            "notes": record["notes"] or "",
        })
    rng = random.Random(args.seed)
    selected = []
    for label in LABELS:
        approved = [row for row in grouped[label] if row["review_status"] == "approved"]
        relabelled = [row for row in grouped[label] if row["review_status"] == "relabelled"]
        rng.shuffle(approved)
        rng.shuffle(relabelled)
        selected.extend((approved + relabelled)[:args.per_class])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("image_path", "human_verified_view", "review_status", "reviewer", "source_path", "original_label", "notes")
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    counts = Counter(row["human_verified_view"] for row in selected)
    print(f"Few-shot manifest: {args.output} ({len(selected)} images)")
    for label in LABELS:
        print(f"{label}: {counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
