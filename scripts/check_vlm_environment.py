#!/usr/bin/env python3
"""Report whether the active interpreter can run the Qwen3-VL baseline."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {package_version('torch')}")
    print(f"Transformers: {package_version('transformers')}")
    print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV') or sys.prefix}")
    errors: list[str] = []
    try:
        import torch

        print(f"CUDA (PyTorch build): {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            print(f"GPU: {properties.name}")
            print(f"GPU memory: {properties.total_memory / (1024 ** 3):.2f} GiB")
            print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
            if not torch.cuda.is_bf16_supported():
                errors.append("目前 GPU/PyTorch 組合未回報 BF16 支援")
        else:
            print("GPU: unavailable through PyTorch")
            print("GPU memory: unavailable")
            print("BF16 supported: False")
            errors.append("PyTorch 無法使用 CUDA")
    except Exception as error:
        print("CUDA (PyTorch build): unavailable")
        print("CUDA available: False")
        print("GPU: unavailable through PyTorch")
        print("GPU memory: unavailable")
        print("BF16 supported: unavailable")
        errors.append(f"無法匯入 PyTorch：{type(error).__name__}: {error}")

    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
        print("Qwen3VLForConditionalGeneration import: OK")
    except Exception as error:
        print(f"Qwen3VLForConditionalGeneration import: FAILED ({type(error).__name__}: {error})")
        errors.append("Transformers 必須提供 Qwen3VLForConditionalGeneration")

    if errors:
        print("\nEnvironment check: FAILED")
        for error in errors:
            print(f"- {error}")
        print("未自動修改 PyTorch、CUDA、driver 或套件版本。")
        return 1
    print("\nEnvironment check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
