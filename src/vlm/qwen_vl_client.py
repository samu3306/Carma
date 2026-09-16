"""Single-load Qwen3-VL inference client."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .json_parser import ModelOutputError, parse_model_output
from .prompts import angle_quality_prompt


class ModelLoadError(RuntimeError):
    """The requested Qwen3-VL model could not be loaded."""


class ImageLoadError(RuntimeError):
    """The input image could not be opened."""


class InferenceError(RuntimeError):
    """Inference failed after model loading."""


@dataclass
class InferenceResult:
    prediction: dict[str, Any] | None
    raw_output: str | None
    json_valid: bool
    inference_seconds: float
    gpu_peak_memory_mb: float
    error: str | None = None


class QwenVLClient:
    def __init__(self, model_name: str, max_new_tokens: int = 512) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.model: Any = None
        self.processor: Any = None
        self._torch: Any = None
        self.inference_lock = threading.RLock()

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except Exception as error:
            raise ModelLoadError(
                "無法匯入 Qwen3-VL 相依套件；請先執行 scripts/check_vlm_environment.py。"
                f"完整錯誤：{type(error).__name__}: {error}"
            ) from error
        try:
            self._torch = torch
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
            )
            self.model.eval()
        except Exception as error:
            self.model = None
            self.processor = None
            raise ModelLoadError(
                f"載入模型 {self.model_name!r} 失敗；未改用其他模型。"
                f"完整錯誤：{type(error).__name__}: {error}"
            ) from error

    def infer(
        self,
        image_path: Path,
        expected_view: str | None = None,
        prompt_text: str | None = None,
    ) -> InferenceResult:
        self.load()
        torch = self._torch
        raw_output: str | None = None
        started = time.perf_counter()
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise ImageLoadError(f"圖片無法開啟：{image_path}；{type(error).__name__}: {error}") from error

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text or angle_quality_prompt(expected_view)},
            ]}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
            input_device = next(self.model.parameters()).device
            inputs = inputs.to(input_device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
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
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError(f"推論失敗：{type(error).__name__}: {error}") from error
