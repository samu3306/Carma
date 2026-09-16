#!/usr/bin/env python3
"""Download iRent images listed in the official Excel workbook.

Default output layout:
    downloads/<CarNo>/<order_number>/<ImageType>_<angle>__<original filename>
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openpyxl import load_workbook


ANGLE_NAMES = {
    1: "left_front",
    2: "right_front",
    3: "left_rear",
    4: "right_rear",
    5: "other_1",
    6: "other_2",
    7: "other_3",
    8: "other_4",
    9: "other_5",
    10: "interior_front",
    11: "interior_rear",
}

REQUIRED_COLUMNS = {"order_number", "CarNo", "ImageType", "Image", "下載網址"}
INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class ImageRecord:
    excel_row: int
    order_number: str
    car_no: str
    image_type: int
    original_name: str
    url: str


@dataclass(frozen=True)
class DownloadResult:
    record: ImageRecord
    status: str
    output_path: str
    message: str = ""


def clean_part(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    text = INVALID_PATH_CHARS.sub("_", text).strip(" .")
    return text or fallback


def normalize_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def parse_image_type(value: Any, row_number: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Excel row {row_number}: invalid ImageType {value!r}") from exc


def is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_records(excel_path: Path, sheet_name: str | None) -> list[ImageRecord]:
    workbook = load_workbook(excel_path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet_name] if sheet_name else workbook[workbook.sheetnames[0]]
        rows = worksheet.iter_rows(values_only=True)
        try:
            headers = next(rows)
        except StopIteration as exc:
            raise ValueError("The Excel workbook is empty.") from exc

        column_map = {str(value).strip(): index for index, value in enumerate(headers) if value is not None}
        missing = REQUIRED_COLUMNS - column_map.keys()
        if missing:
            raise ValueError(f"Missing required Excel columns: {', '.join(sorted(missing))}")

        records: list[ImageRecord] = []
        for excel_row, row in enumerate(rows, start=2):
            url_value = row[column_map["下載網址"]]
            if url_value is None or not str(url_value).strip():
                continue
            url = str(url_value).strip()
            if not is_http_url(url):
                print(f"Warning: skip Excel row {excel_row}, invalid URL", file=sys.stderr)
                continue

            image_type = parse_image_type(row[column_map["ImageType"]], excel_row)
            original_name = clean_part(row[column_map["Image"]], f"row_{excel_row}.jpg")
            records.append(
                ImageRecord(
                    excel_row=excel_row,
                    order_number=clean_part(normalize_number(row[column_map["order_number"]]), "unknown_order"),
                    car_no=clean_part(row[column_map["CarNo"]], "unknown_car"),
                    image_type=image_type,
                    original_name=original_name,
                    url=url,
                )
            )
        return records
    finally:
        workbook.close()


def output_path_for(record: ImageRecord, output_dir: Path, flat: bool) -> Path:
    angle = ANGLE_NAMES.get(record.image_type, f"type_{record.image_type}")
    filename = f"{record.image_type:02d}_{angle}__{record.original_name}"
    if flat:
        filename = f"{record.order_number}__{filename}"
        return output_dir / record.car_no / filename
    return output_dir / record.car_no / record.order_number / filename


def download_one(
    record: ImageRecord,
    output_dir: Path,
    flat: bool,
    overwrite: bool,
    timeout: float,
    dry_run: bool,
) -> DownloadResult:
    destination = output_path_for(record, output_dir, flat)
    if dry_run:
        return DownloadResult(record, "dry_run", str(destination))
    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return DownloadResult(record, "skipped", str(destination), "already exists")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                request = Request(record.url, headers={"User-Agent": "iRent-Hackathon-Image-Downloader/1.0"})
                with urlopen(request, timeout=timeout) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    allowed_content_types = {
                        ".png",
                        "image/png",
                        "image/jpeg",
                        "image/jpg",
                        "application/octet-stream",
                    }

                    if (
                        content_type
                        and content_type not in allowed_content_types
                        and not content_type.startswith("image/")
                    ):
                        raise ValueError(
                            f"unexpected Content-Type: {content_type}"
                        )

                    with temporary.open("wb") as output_file:
                        while chunk := response.read(128 * 1024):
                            output_file.write(chunk)
                last_error = None
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt < 3:
                    time.sleep(0.8 * (2 ** (attempt - 1)))
        if last_error is not None:
            raise last_error
        if temporary.stat().st_size == 0:
            raise ValueError("downloaded file is empty")
        temporary.replace(destination)
        return DownloadResult(record, "downloaded", str(destination))
    except Exception as exc:  # keep batch running and record the precise failure
        temporary.unlink(missing_ok=True)
        return DownloadResult(record, "failed", str(destination), str(exc))


def write_log(results: list[DownloadResult], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8-sig") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["excel_row", "order_number", "CarNo", "ImageType", "status", "output_path", "message", "url"])
        for result in results:
            record = result.record
            writer.writerow([
                record.excel_row,
                record.order_number,
                record.car_no,
                record.image_type,
                result.status,
                result.output_path,
                result.message,
                record.url,
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download iRent images from the official Excel list.")
    parser.add_argument("excel", type=Path, help="Path to 車外車內照片上傳清單.xlsx")
    parser.add_argument("-o", "--output", type=Path, default=Path("downloaded_images"), help="Output directory")
    parser.add_argument("--sheet", help="Worksheet name; defaults to the first worksheet")
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads (default: 8)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Read timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty files")
    parser.add_argument("--flat", action="store_true", help="Store images directly under each car folder")
    parser.add_argument("--dry-run", action="store_true", help="Only preview paths; do not download")
    parser.add_argument("--limit", type=int, help="Only process the first N rows, useful for testing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.excel.is_file():
        print(f"Error: Excel file not found: {args.excel}", file=sys.stderr)
        return 2
    if args.workers < 1 or args.workers > 32:
        print("Error: --workers must be between 1 and 32", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be at least 1", file=sys.stderr)
        return 2

    try:
        records = load_records(args.excel, args.sheet)
    except Exception as exc:
        print(f"Error reading Excel: {exc}", file=sys.stderr)
        return 2
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("No downloadable image rows found.")
        return 0

    output_dir = args.output.resolve()
    print(f"Found {len(records):,} image rows. Output: {output_dir}")
    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                record,
                output_dir,
                args.flat,
                args.overwrite,
                args.timeout,
                args.dry_run,
            ): record
            for record in records
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 100 == 0 or completed == len(records):
                print(f"Progress: {completed:,}/{len(records):,}")

    results.sort(key=lambda item: item.record.excel_row)
    log_path = output_dir / "download_log.csv"
    write_log(results, log_path)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items()))
    print(f"Finished: {summary}")
    print(f"Log: {log_path}")
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
