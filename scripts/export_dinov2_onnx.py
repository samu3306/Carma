#!/usr/bin/env python3
"""Export the cached frozen DINOv2 ViT-S/14 feature extractor to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModel


ROOT_DIR = Path(__file__).resolve().parents[1]


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=pixel_values)
        pooled = getattr(output, "pooler_output", None)
        return pooled if pooled is not None else output.last_hidden_state[:, 0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--cache-dir", type=Path, default=ROOT_DIR / "models" / "huggingface")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "models" / "dinov2_angle_classifier" / "dinov2_vits14.onnx",
    )
    args = parser.parse_args()

    model = AutoModel.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=True,
    ).cpu().eval()
    wrapper = FeatureExtractor(model).eval()
    example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            args.output,
            input_names=["pixel_values"],
            output_names=["features"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"ONNX written to {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
