from __future__ import annotations

import csv
import io
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import DatasetAnnotation


LABELS = (
    "left_front",
    "right_front",
    "left_rear",
    "right_rear",
    "interior_front",
    "interior_rear",
    "other",
)
STATUSES = ("unreviewed", "approved", "relabelled", "invalid", "excluded")


def sync_manifest(db: Session, manifest_path: Path) -> int:
    if not manifest_path.is_file():
        return 0
    existing = set(db.scalars(select(DatasetAnnotation.source_path)).all())
    created = 0
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_path = (row.get("source_path") or "").strip()
            original_label = (row.get("category") or "other").strip()
            if not source_path or source_path in existing:
                continue
            image_type_text = (row.get("image_type") or "").strip()
            db.add(DatasetAnnotation(
                source_path=source_path,
                image_type=int(image_type_text) if image_type_text else None,
                original_label=original_label if original_label in LABELS else "other",
                review_status="unreviewed",
            ))
            existing.add(source_path)
            created += 1
    if created:
        db.commit()
    return created


def annotation_dict(item: DatasetAnnotation) -> dict:
    return {
        "id": item.id,
        "source_path": item.source_path,
        "image_url": f"/api/labeling/items/{item.id}/image",
        "image_type": item.image_type,
        "original_label": item.original_label,
        "actual_label": item.actual_label,
        "acceptable": item.acceptable,
        "retake_reason": item.retake_reason,
        "review_status": item.review_status,
        "notes": item.notes,
        "reviewer": item.reviewer,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def labeling_stats(db: Session) -> dict:
    status_counts = {status: 0 for status in STATUSES}
    category_counts = {label: 0 for label in LABELS}
    for status, count in db.execute(
        select(DatasetAnnotation.review_status, func.count()).group_by(DatasetAnnotation.review_status)
    ):
        status_counts[status] = count
    for label, count in db.execute(
        select(DatasetAnnotation.original_label, func.count()).group_by(DatasetAnnotation.original_label)
    ):
        category_counts[label] = count
    total = sum(status_counts.values())
    reviewed = total - status_counts["unreviewed"]
    return {
        "total": total,
        "reviewed": reviewed,
        "remaining": status_counts["unreviewed"],
        "progress_percent": round(reviewed / total * 100, 1) if total else 0,
        "by_status": status_counts,
        "by_original_label": category_counts,
    }


def export_annotations_csv(db: Session) -> str:
    output = io.StringIO()
    fields = [
        "source_path", "image_type", "original_label", "actual_label",
        "acceptable", "retake_reason", "review_status", "notes", "reviewer", "updated_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in db.scalars(select(DatasetAnnotation).order_by(DatasetAnnotation.source_path)):
        writer.writerow({field: getattr(item, field) for field in fields})
    return "\ufeff" + output.getvalue()
