from __future__ import annotations

import argparse
import json
from collections import Counter
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from backend.app.db import SessionLocal
from backend.app.models import DatasetAnnotation


DEFAULT_OUTPUT = ROOT_DIR / "datasets" / "reports" / "annotation_summary.json"


def effective_label(item: DatasetAnnotation) -> str:
    return item.actual_label or item.original_label


def build_summary() -> dict:
    with SessionLocal() as db:
        rows = list(db.scalars(select(DatasetAnnotation)).all())

    reviewed = [item for item in rows if item.review_status != "unreviewed"]
    accepted = [item for item in reviewed if item.acceptable is True]
    invalid = [item for item in reviewed if item.review_status == "invalid"]
    excluded = [item for item in reviewed if item.review_status == "excluded"]
    trainable = accepted + invalid
    orders = {"/".join(item.source_path.split("/")[:2]) for item in trainable}

    status_counts = Counter(item.review_status for item in rows)
    accepted_labels = Counter(effective_label(item) for item in accepted)
    reason_counts = Counter(item.retake_reason or "unspecified" for item in invalid)
    original_counts = Counter(item.original_label for item in reviewed)

    warnings: list[str] = []
    for label in ("left_front", "right_front", "left_rear", "right_rear", "interior_front", "interior_rear"):
        count = accepted_labels[label]
        if count < 20:
            warnings.append(f"{label} 只有 {count} 張合格人工標註，暫不適合訓練可靠角度分類器")
    if len(orders) < 30:
        warnings.append("可訓練資料涵蓋少於 30 個訂單，跨車輛泛化風險偏高")
    if len(trainable) < 500:
        warnings.append("二元重拍資料少於 500 張，目前模型只能視為基準實驗")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
        "reviewed": len(reviewed),
        "remaining": status_counts["unreviewed"],
        "trainable_binary": len(trainable),
        "trainable_orders": len(orders),
        "accepted": len(accepted),
        "invalid": len(invalid),
        "excluded": len(excluded),
        "by_status": dict(sorted(status_counts.items())),
        "accepted_by_effective_label": dict(sorted(accepted_labels.items())),
        "invalid_by_reason": dict(sorted(reason_counts.items())),
        "reviewed_by_original_label": dict(sorted(original_counts.items())),
        "warnings": warnings,
        "recommended_task": "binary_retake_baseline",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 Carma 人工標註的數量、平衡與模型可用性")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = build_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n報告已寫入：{args.output}")


if __name__ == "__main__":
    main()
