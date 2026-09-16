#!/usr/bin/env python3
"""Evaluate Qwen3-VL with human-reviewed in-context angle examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_angle_vlm import evaluate_rows, load_manifest  # noqa: E402
from src.vlm.few_shot_qwen_vl_client import FewShotQwenVLClient  # noqa: E402
from src.vlm.schemas import DETECTED_VIEWS  # noqa: E402


def load_references(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "human_verified_view"}
    if not rows or required.difference(rows[0]):
        raise ValueError("few-shot manifest 必須包含 image_path 與 human_verified_view")
    for row in rows:
        if row["human_verified_view"] not in DETECTED_VIEWS:
            raise ValueError(f"不允許的人工角度標籤：{row['human_verified_view']}")
        if not Path(row["image_path"]).is_file():
            raise ValueError(f"參考圖片不存在：{row['image_path']}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="使用人工審核照片執行 Qwen3-VL few-shot 評估")
    parser.add_argument("--manifest", type=Path, default=Path("data/eval/angle_eval_manifest.csv"))
    parser.add_argument("--few-shot-manifest", type=Path, default=Path("data/eval/few_shot_reference_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen3_vl_angle_few_shot"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必須大於 0")
    try:
        rows = load_manifest(args.manifest)
        references = load_references(args.few_shot_manifest)
    except (OSError, ValueError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 2
    eval_paths = {str(Path(row["image_path"]).resolve()) for row in rows}
    overlap = [row["image_path"] for row in references if str(Path(row["image_path"]).resolve()) in eval_paths]
    if overlap:
        print(f"錯誤：few-shot 參考集與評估集重疊：{overlap[0]}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": "few_shot_in_context",
        "model": args.model,
        "manifest": str(args.manifest.resolve()),
        "few_shot_manifest": str(args.few_shot_manifest.resolve()),
        "few_shot_reference_count": len(references),
        "output": str(args.output.resolve()),
        "limit": args.limit,
        "resume": args.resume,
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
            rows,
            args.output,
            FewShotQwenVLClient(args.model, references, args.max_new_tokens),
            args.resume,
            args.limit,
        )
    except FileExistsError as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 2
    successes = sum(bool(row.get("inference_success")) for row in results)
    print(f"Few-shot 完成：{len(results)} 筆，成功 {successes} 筆；參考圖 {len(references)} 張")
    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
