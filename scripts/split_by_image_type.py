#!/usr/bin/env python3
"""Build a non-destructive ImageType folder view for labeling and training."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import Counter
from pathlib import Path


CATEGORY_BY_TYPE = {
    1: ("left_front", "左前"),
    2: ("right_front", "右前"),
    3: ("left_rear", "左後"),
    4: ("right_rear", "右後"),
    10: ("interior_front", "車前座"),
    11: ("interior_rear", "車後座"),
}
OTHER_CATEGORY = ("other", "其他")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TYPE_PATTERN = re.compile(r"^(\d{1,2})_")


def image_type_from_name(filename: str) -> int | None:
    match = TYPE_PATTERN.match(filename)
    return int(match.group(1)) if match else None


def category_for_type(image_type: int | None) -> tuple[str, str]:
    return CATEGORY_BY_TYPE.get(image_type, OTHER_CATEGORY)


def iter_images(source: Path):
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def materialize(source: Path, target: Path, mode: str, dry_run: bool) -> str:
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return "existing"
        return "collision"
    if dry_run:
        return "planned"
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))
    elif mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)
    return "created"


def build_dataset(
    source: Path,
    destination: Path,
    mode: str = "symlink",
    dry_run: bool = False,
    flat: bool = False,
) -> Counter:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"找不到來源資料夾：{source}")

    rows: list[dict[str, str | int]] = []
    counts: Counter = Counter()
    for image_path in iter_images(source):
        relative = image_path.relative_to(source)
        image_type = image_type_from_name(image_path.name)
        category, category_zh = category_for_type(image_type)
        target_name = "__".join(relative.parts) if flat else relative
        target = destination / category / target_name
        status = materialize(image_path, target, mode, dry_run)
        counts[category] += 1
        counts[f"status:{status}"] += 1
        rows.append({
            "source_path": relative.as_posix(),
            "image_type": "" if image_type is None else image_type,
            "category": category,
            "category_zh": category_zh,
            "target_path": target.relative_to(destination).as_posix(),
            "status": status,
            "weak_label": "true",
        })

    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / "manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "source_path", "image_type", "category", "category_zh",
                "target_path", "status", "weak_label",
            ])
            writer.writeheader()
            writer.writerows(rows)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="依 ImageType 建立七類影像資料夾，不修改原始圖片")
    parser.add_argument("--source", type=Path, default=Path("downloaded_images"))
    parser.add_argument("--destination", type=Path, default=Path("datasets/by_image_type"))
    parser.add_argument("--mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--flat", action="store_true", help="每個類別直接放圖片，將來源路徑編入檔名")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    counts = build_dataset(args.source, args.destination, args.mode, args.dry_run, args.flat)
    print("ImageType 分類完成" if not args.dry_run else "ImageType 分類預覽")
    ordered_categories = [CATEGORY_BY_TYPE[value] for value in (1, 2, 3, 4, 10, 11)] + [OTHER_CATEGORY]
    for category, category_zh in ordered_categories:
        print(f"{category_zh} ({category}): {counts[category]}")
    print(f"created={counts['status:created']} existing={counts['status:existing']} planned={counts['status:planned']} collision={counts['status:collision']}")
    return 1 if counts["status:collision"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
