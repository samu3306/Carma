#!/usr/bin/env python3
"""Run one image through Qwen3-VL and print the validated JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm.qwen_vl_client import (  # noqa: E402
    ImageLoadError,
    InferenceError,
    ModelLoadError,
    QwenVLClient,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="以 Qwen3-VL 評估單張 iRent 取還車照片")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--raw-output", type=Path, default=Path("outputs/single_image_raw_output.txt"))
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"錯誤：找不到圖片：{args.image}", file=sys.stderr)
        return 2
    try:
        result = QwenVLClient(args.model, args.max_new_tokens).infer(args.image)
    except (ModelLoadError, ImageLoadError, InferenceError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 1

    print(f"推論時間：{result.inference_seconds:.3f} 秒", file=sys.stderr)
    print(f"GPU 峰值顯存：{result.gpu_peak_memory_mb:.1f} MiB", file=sys.stderr)
    if not result.json_valid:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(result.raw_output or "", encoding="utf-8")
        print(f"模型輸出 JSON 無效：{result.error}", file=sys.stderr)
        print(f"原始輸出已保存：{args.raw_output}", file=sys.stderr)
        return 1
    print(json.dumps(result.prediction, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
