#!/usr/bin/env python3
"""Run one photo through Qwen3-VL and deterministic retake guidance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm.photo_guidance import build_photo_guidance  # noqa: E402
from src.vlm.qwen_vl_client import (  # noqa: E402
    ImageLoadError,
    InferenceError,
    ModelLoadError,
    QwenVLClient,
)
from src.vlm.schemas import DETECTED_VIEWS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="測試單張 iRent 照片並產生固定重拍建議")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--expected-view", choices=DETECTED_VIEWS, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        print(f"錯誤：找不到圖片：{args.image}", file=sys.stderr)
        return 2
    try:
        result = QwenVLClient(args.model).infer(args.image)
    except (ModelLoadError, ImageLoadError, InferenceError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 1
    payload = {
        "image_path": str(args.image.resolve()),
        "json_valid": result.json_valid,
        "inference_seconds": result.inference_seconds,
        "gpu_peak_memory_mb": result.gpu_peak_memory_mb,
        "error": result.error,
        "raw_model_output": result.raw_output,
    }
    if result.prediction is not None:
        payload["model_prediction"] = result.prediction
        payload["guidance"] = build_photo_guidance(result.prediction, args.expected_view)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"結果已寫入：{args.output}", file=sys.stderr)
    print(rendered)
    return 0 if result.json_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
