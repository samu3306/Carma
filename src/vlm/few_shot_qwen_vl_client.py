"""Qwen3-VL inference with human-reviewed in-context angle examples."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .few_shot_prompts import (
    FEW_SHOT_INTRO,
    REAR_SIDE_INTRO,
    REAR_SIDE_TARGET_PROMPT,
    TARGET_INTRO,
)
from .json_parser import ModelOutputError, parse_model_output
from .prompts import angle_quality_prompt
from .qwen_vl_client import (
    ImageLoadError,
    InferenceError,
    InferenceResult,
    ModelLoadError,
    QwenVLClient,
)


class FewShotQwenVLClient(QwenVLClient):
    def __init__(
        self,
        model_name: str,
        references: list[dict[str, str]],
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(model_name, max_new_tokens)
        if not references:
            raise ValueError("few-shot references 不可為空")
        self.references = references

    def infer(self, image_path: Path, rear_side_only: bool = False) -> InferenceResult:
        self.load()
        torch = self._torch
        raw_output: str | None = None
        started = time.perf_counter()
        try:
            with Image.open(image_path) as opened:
                target_image = opened.convert("RGB")
            reference_images = []
            for reference in self.references:
                with Image.open(reference["image_path"]) as opened:
                    reference_images.append(opened.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise ImageLoadError(f"圖片無法開啟：{type(error).__name__}: {error}") from error

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            intro = REAR_SIDE_INTRO if rear_side_only else FEW_SHOT_INTRO
            content: list[dict[str, Any]] = [{"type": "text", "text": intro}]
            for index, (reference, image) in enumerate(zip(self.references, reference_images), 1):
                content.extend([
                    {
                        "type": "text",
                        "text": f"人工審核範例 {index}，正確角度是 {reference['human_verified_view']}：",
                    },
                    {"type": "image", "image": image},
                ])
            content.extend([
                {"type": "text", "text": TARGET_INTRO},
                {"type": "image", "image": target_image},
                {
                    "type": "text",
                    "text": REAR_SIDE_TARGET_PROMPT if rear_side_only else angle_quality_prompt(),
                },
            ])
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[prompt],
                images=[*reference_images, target_image],
                padding=True,
                return_tensors="pt",
            )
            input_device = next(self.model.parameters()).device
            inputs = inputs.to(input_device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            raw_output = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            elapsed = time.perf_counter() - started
            peak_mb = (
                torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            )
            try:
                prediction = parse_model_output(raw_output)
                return InferenceResult(prediction, raw_output, True, elapsed, peak_mb)
            except ModelOutputError as error:
                return InferenceResult(None, raw_output, False, elapsed, peak_mb, str(error))
        except torch.cuda.OutOfMemoryError as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise InferenceError(f"CUDA OOM：{type(error).__name__}: {error}") from error
        except Exception as error:
            raise InferenceError(f"Few-shot 推論失敗：{type(error).__name__}: {error}") from error
