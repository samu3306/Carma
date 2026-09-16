"""Qwen3-VL few-shot comparison for pickup/return exterior photos."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .json_parser import ModelOutputError, extract_json_object
from .qwen_vl_client import (
    ImageLoadError,
    InferenceError,
    InferenceResult,
    QwenVLClient,
)

DECISIONS = {"new_damage", "no_new_damage", "uncertain"}
DAMAGE_TYPES = {
    "scratch", "dent", "crack", "paint_loss", "broken_part", "other", "none",
}
SEVERITIES = {"none", "low", "medium", "high"}

DAMAGE_PAIR_PROMPT = """你是租賃車輛新車損比對員。待判斷資料包含同一台車、同一方向的兩張照片：
第一張是借車前照片，第二張是還車後照片。

只判斷還車照相較借車照是否出現新的可見車身損傷。刮痕、凹陷、裂痕、掉漆或零件破損才算車損。光線、陰影、反射、水漬、污垢、拍攝角度、距離、背景與既有損傷都不算新車損。若兩張照片無法可靠對應相同車身區域，必須輸出 uncertain，不可猜測。

資料夾索賠標籤只是弱標籤；你仍必須根據待判斷照片的可見證據獨立判斷。不要辨識責任歸屬或是否應索賠。

只輸出一個合法 JSON object，不要輸出 Markdown：
{
  "decision": "new_damage",
  "damage_type": "scratch",
  "severity": "medium",
  "vehicle_area": "右前保險桿",
  "confidence": 0.86,
  "changed_regions": ["還車照右前保險桿新增線狀刮痕"],
  "explanation": "說明借車照與還車照相同區域的可見差異"
}

decision 只能是 new_damage、no_new_damage、uncertain。
damage_type 只能是 scratch、dent、crack、paint_loss、broken_part、other、none。
severity 只能是 none、low、medium、high。
若 decision 不是 new_damage，damage_type 與 severity 必須為 none，changed_regions 必須為空陣列。
confidence 必須介於 0 到 1。"""


def validate_damage_prediction(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelOutputError("車損輸出必須是 JSON object")
    required = {
        "decision", "damage_type", "severity", "vehicle_area",
        "confidence", "changed_regions", "explanation",
    }
    missing = required.difference(value)
    if missing:
        raise ModelOutputError(f"車損輸出缺少欄位：{', '.join(sorted(missing))}")
    decision = value["decision"]
    damage_type = value["damage_type"]
    severity = value["severity"]
    if decision not in DECISIONS:
        raise ModelOutputError(f"不允許的 decision：{decision!r}")
    if damage_type not in DAMAGE_TYPES:
        raise ModelOutputError(f"不允許的 damage_type：{damage_type!r}")
    if severity not in SEVERITIES:
        raise ModelOutputError(f"不允許的 severity：{severity!r}")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ModelOutputError("confidence 必須是數字")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ModelOutputError("confidence 必須介於 0 與 1")
    regions = value["changed_regions"]
    if not isinstance(regions, list) or any(not isinstance(item, str) for item in regions):
        raise ModelOutputError("changed_regions 必須是字串陣列")
    area = value["vehicle_area"]
    explanation = value["explanation"]
    if not isinstance(area, str) or not isinstance(explanation, str) or not explanation.strip():
        raise ModelOutputError("vehicle_area 與 explanation 必須是字串")
    if decision != "new_damage":
        damage_type = "none"
        severity = "none"
        regions = []
        area = ""
    elif damage_type == "none" or severity == "none":
        raise ModelOutputError("new_damage 必須提供車損類型與嚴重程度")
    return {
        "decision": decision,
        "damage_type": damage_type,
        "severity": severity,
        "vehicle_area": area.strip(),
        "confidence": confidence,
        "changed_regions": [item.strip() for item in regions if item.strip()],
        "explanation": explanation.strip(),
    }


def load_damage_references(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"before_path", "after_path", "label", "pair_id"}
    if not rows or required.difference(rows[0]):
        raise ValueError("車損 Few-shot manifest 欄位不完整")
    labels = {row["label"] for row in rows}
    if not {"new_damage", "no_new_damage"}.issubset(labels):
        raise ValueError("Few-shot 必須同時包含 new_damage 與 no_new_damage")
    for row in rows:
        if row["label"] not in {"new_damage", "no_new_damage"}:
            raise ValueError(f"Few-shot 標籤不允許：{row['label']}")
        for field in ("before_path", "after_path"):
            if not Path(row[field]).is_file():
                raise ValueError(f"Few-shot 圖片不存在：{row[field]}")
    return rows


def _open_image(path: Path, max_edge: int = 768) -> Image.Image:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ImageLoadError(f"圖片無法開啟：{path}；{type(error).__name__}: {error}") from error
    image.thumbnail((max_edge, max_edge))
    return image


class QwenDamagePairAnalyzer:
    """Compare one target pair with weakly labelled in-context reference pairs."""

    def __init__(
        self,
        client: QwenVLClient,
        references: list[dict[str, str]],
        max_new_tokens: int = 384,
    ) -> None:
        if not references:
            raise ValueError("車損 Few-shot references 不可為空")
        self.client = client
        self.references = references
        self.max_new_tokens = max_new_tokens
        self._lock = getattr(client, "inference_lock", threading.RLock())

    def infer_pair(self, before_path: Path, after_path: Path) -> InferenceResult:
        with self._lock:
            return self._infer_pair_locked(before_path, after_path)

    def _infer_pair_locked(self, before_path: Path, after_path: Path) -> InferenceResult:
        self.client.load()
        torch = self.client._torch
        started = time.perf_counter()
        raw_output: str | None = None
        reference_images: list[Image.Image] = []
        try:
            content: list[dict[str, Any]] = [{
                "type": "text",
                "text": "以下是官方資料夾弱標籤範例，只用來示範輸入順序與可能分類。",
            }]
            for index, reference in enumerate(self.references, 1):
                before = _open_image(Path(reference["before_path"]))
                after = _open_image(Path(reference["after_path"]))
                reference_images.extend([before, after])
                content.extend([
                    {
                        "type": "text",
                        "text": (
                            f"弱標籤範例 {index}：第一張借車、第二張還車；"
                            f"資料夾標籤為 {reference['label']}。"
                        ),
                    },
                    {"type": "image", "image": before},
                    {"type": "image", "image": after},
                ])
            target_before = _open_image(before_path)
            target_after = _open_image(after_path)
            all_images = [*reference_images, target_before, target_after]
            content.extend([
                {
                    "type": "text",
                    "text": "現在判斷待測照片對：下一張是借車照，再下一張是還車照。",
                },
                {"type": "image", "image": target_before},
                {"type": "image", "image": target_after},
                {"type": "text", "text": DAMAGE_PAIR_PROMPT},
            ])
            messages = [{"role": "user", "content": content}]
            prompt = self.client.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.client.processor(
                text=[prompt], images=all_images, padding=True, return_tensors="pt"
            )
            inputs = inputs.to(next(self.client.model.parameters()).device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                generated = self.client.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            trimmed = [
                output[len(source):]
                for source, output in zip(inputs.input_ids, generated)
            ]
            raw_output = self.client.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            elapsed = time.perf_counter() - started
            peak_mb = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if torch.cuda.is_available()
                else 0.0
            )
            try:
                prediction = validate_damage_prediction(extract_json_object(raw_output))
                return InferenceResult(
                    prediction, raw_output, True, elapsed, peak_mb
                )
            except ModelOutputError as error:
                return InferenceResult(
                    None, raw_output, False, elapsed, peak_mb, str(error)
                )
        except torch.cuda.OutOfMemoryError as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise InferenceError(f"CUDA OOM：{type(error).__name__}: {error}") from error
        except (ImageLoadError, InferenceError):
            raise
        except Exception as error:
            raise InferenceError(
                f"車損配對推論失敗：{type(error).__name__}: {error}"
            ) from error
