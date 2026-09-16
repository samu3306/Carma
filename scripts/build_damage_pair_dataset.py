#!/usr/bin/env python3
"""Build leakage-free Qwen damage-pair few-shot and test manifests."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}
VIEWS = ("左前", "右前", "左後", "右後")
VIEW_NAMES = {
    "左前": "left_front",
    "右前": "right_front",
    "左後": "left_rear",
    "右後": "right_rear",
}


def collect_pairs(folder: Path, label: str, label_source: str) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(
        lambda: {"借": [], "還": []}
    )
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        compact = path.stem.replace(" ", "")
        view = next((item for item in VIEWS if item in compact), None)
        if view is None:
            continue
        stage = "還" if "還" in compact else "借" if "借" in compact else None
        if stage is None:
            continue
        plate = compact[: compact.find(view)].rstrip("-_")
        grouped[(plate, view)][stage].append(path.resolve())

    rows: list[dict[str, str]] = []
    for (plate, view), stages in sorted(grouped.items()):
        if not stages["借"] or not stages["還"]:
            continue
        for return_index, current in enumerate(stages["還"], 1):
            suffix = f"-{return_index}" if len(stages["還"]) > 1 else ""
            rows.append({
                "pair_id": f"{plate}-{VIEW_NAMES[view]}{suffix}",
                "plate": plate,
                "view": VIEW_NAMES[view],
                "before_path": str(stages["借"][0]),
                "after_path": str(current),
                "label": label,
                "label_source": label_source,
            })
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "pair_id", "plate", "view", "before_path", "after_path",
        "label", "label_source", "split",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-dir", type=Path, default=Path("進行索賠資料"))
    parser.add_argument("--no-claim-dir", type=Path, default=Path("沒有進行索賠"))
    parser.add_argument(
        "--few-shot-output",
        type=Path,
        default=Path("data/damage/few_shot_pairs.csv"),
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/damage/test_pairs.csv"),
    )
    parser.add_argument("--reference-cases-per-label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    positive = collect_pairs(
        args.claim_dir,
        "new_damage",
        "weak_label_from_claim_folder",
    )
    negative = collect_pairs(
        args.no_claim_dir,
        "no_new_damage",
        "weak_label_from_no_claim_folder",
    )
    if not positive or not negative:
        raise SystemExit("錯誤：兩個資料夾都必須包含可配對的借／還照片")

    rng = random.Random(args.seed)
    reference_plates: dict[str, set[str]] = {}
    references: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for label, rows in (("new_damage", positive), ("no_new_damage", negative)):
        plates = sorted({row["plate"] for row in rows})
        rng.shuffle(plates)
        selected = set(plates[: args.reference_cases_per_label])
        reference_plates[label] = selected
        selected_once: set[str] = set()
        for row in rows:
            if row["plate"] in selected:
                if row["plate"] not in selected_once:
                    references.append({**row, "split": "few_shot"})
                    selected_once.add(row["plate"])
                continue
            test_rows.append({**row, "split": "test"})

    write_manifest(args.few_shot_output, references)
    write_manifest(args.test_output, test_rows)
    overlap = {row["plate"] for row in references} & {row["plate"] for row in test_rows}
    if overlap:
        raise RuntimeError(f"車牌資料洩漏：{sorted(overlap)}")

    print(f"Few-shot: {args.few_shot_output} ({len(references)} pairs)")
    print(f"Test: {args.test_output} ({len(test_rows)} pairs)")
    for label in ("new_damage", "no_new_damage"):
        ref_count = sum(row["label"] == label for row in references)
        test_count = sum(row["label"] == label for row in test_rows)
        print(
            f"{label}: reference={ref_count}, test={test_count}, "
            f"reference_plates={sorted(reference_plates[label])}"
        )
    print("標籤屬於資料夾層級弱標籤，正式評估前仍需人工確認照片中是否可見新車損。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
