"""Qwen3-VL capture-quality adapter for the six-view inspection workflow."""

from __future__ import annotations

import threading
import re
from pathlib import Path
from typing import Any

from src.vlm.photo_guidance import EXTERIOR_VIEWS, GUIDANCE_TEMPLATES, build_photo_guidance
from src.vlm.prompts import plate_visibility_prompt
from src.vlm.qwen_vl_client import QwenVLClient


ISSUE_LABELS = {
    "blur": "清晰度",
    "too_dark": "曝光",
    "overexposed": "曝光",
    "too_far": "拍攝距離",
    "too_close": "拍攝距離",
    "vehicle_cropped": "構圖完整",
    "wrong_view": "六方向角度",
    "plate_not_visible": "目標車牌",
    "plate_mismatch": "訂單車牌",
    "obstruction": "畫面遮擋",
    "other": "照片內容",
}


SIDE_MIRROR_PAIRS = {
    frozenset({"left_front", "right_front"}),
    frozenset({"left_rear", "right_rear"}),
}


class QwenCaptureQualityAnalyzer:
    """Run the tuned Qwen assessment and an exterior target-plate second pass."""

    def __init__(self, model_name: str, client: QwenVLClient | None = None) -> None:
        self.model_name = model_name
        self.client = client or QwenVLClient(model_name)
        self._lock = getattr(self.client, "inference_lock", threading.RLock())

    def warmup(self) -> None:
        self.client.load()

    @staticmethod
    def normalize_plate(value: str | None) -> str:
        return re.sub(r"[^A-Z0-9]", "", (value or "").upper())

    def analyze(
        self,
        image_path: Path,
        expected_view: str,
        expected_plate: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            base_result = self.client.infer(image_path)
            if not base_result.json_valid or base_result.prediction is None:
                raise RuntimeError(base_result.error or "Qwen 品質輸出格式錯誤")
            plate_result = None
            if expected_view in EXTERIOR_VIEWS:
                plate_result = self.client.infer(
                    image_path,
                    prompt_text=plate_visibility_prompt(expected_view),
                )

        prediction = dict(base_result.prediction)
        inference_seconds = base_result.inference_seconds
        gpu_peak_memory_mb = base_result.gpu_peak_memory_mb
        plate_method = "not_applicable"
        plate_text: str | None = None
        plate_confidence: float | None = None
        plate_match: bool | None = None

        if expected_view in EXTERIOR_VIEWS:
            plate_method = "base_vlm"
            if (
                plate_result is not None
                and plate_result.json_valid
                and plate_result.prediction is not None
            ):
                inference_seconds += plate_result.inference_seconds
                gpu_peak_memory_mb = max(
                    gpu_peak_memory_mb, plate_result.gpu_peak_memory_mb
                )
                plate_prediction = plate_result.prediction
                plate_text = plate_prediction.get("plate_text")
                plate_confidence = plate_prediction.get("plate_confidence")
                original_issues = list(prediction.get("quality_issues", []))
                remaining_issues = [
                    issue for issue in original_issues if issue != "plate_not_visible"
                ]
                prediction["quality_issues"] = remaining_issues
                prediction["plate_visible"] = bool(
                    plate_prediction.get("plate_visible")
                )
                normalized_text = self.normalize_plate(plate_text)
                normalized_expected = self.normalize_plate(expected_plate)
                if prediction["plate_visible"] and normalized_expected and not normalized_text:
                    prediction["plate_visible"] = False
                if prediction["plate_visible"] and normalized_expected:
                    plate_match = normalized_text == normalized_expected
                    if not plate_match:
                        remaining_issues.append("plate_mismatch")
                        prediction["quality_passed"] = False
                elif normalized_expected:
                    plate_match = False
                if not prediction["plate_visible"]:
                    prediction["quality_passed"] = False
                elif original_issues == ["plate_not_visible"] and plate_match is not False:
                    prediction["quality_passed"] = True
                prediction["explanation"] = (
                    f"目標車牌檢查：{plate_prediction.get('explanation', '')} "
                    + (
                        f"辨識車牌 {plate_text or '無法讀取'}，"
                        f"訂單車牌 {expected_plate}，"
                        if expected_plate else f"辨識車牌 {plate_text or '無法讀取'}，"
                    )
                    + f"整體判斷：{prediction.get('explanation', '')}"
                )
                prediction["plate_text"] = plate_text
                prediction["plate_confidence"] = plate_confidence
                prediction["expected_plate"] = expected_plate
                prediction["plate_match"] = plate_match
                plate_method = "target_vehicle_second_pass"

        detected_view = prediction.get("detected_view")
        side_review_required = (
            frozenset({expected_view, detected_view}) in SIDE_MIRROR_PAIRS
            and expected_view != detected_view
        )
        guidance = build_photo_guidance(
            prediction, expected_view, side_review_required
        )
        checks = self._checks(
            guidance, prediction, expected_view, side_review_required
        )
        accepted_for_capture = guidance["passed"] or guidance["review_required"]
        return {
            "passed": accepted_for_capture,
            "review_required": guidance["review_required"],
            "detected_view": guidance["detected_view"],
            "plate_detected": bool(prediction.get("plate_visible"))
            if expected_view in EXTERIOR_VIEWS
            else False,
            "retake_instruction": (
                None if accepted_for_capture else guidance["retake_instruction"]
            ),
            "warnings": (
                [] if accepted_for_capture else [guidance["retake_instruction"]]
            ),
            "model_confidence": float(prediction.get("confidence") or 0),
            "checks": checks,
            "issues": guidance["issues"],
            "model_explanation": guidance["model_explanation"],
            "inference_seconds": inference_seconds,
            "gpu_peak_memory_mb": gpu_peak_memory_mb,
            "plate_method": plate_method,
            "plate_text": plate_text,
            "plate_confidence": plate_confidence,
            "expected_plate": expected_plate,
            "plate_match": plate_match,
            "raw_model_output": base_result.raw_output,
        }

    @staticmethod
    def _checks(
        guidance: dict[str, Any],
        prediction: dict[str, Any],
        expected_view: str,
        side_review_required: bool,
    ) -> list[dict[str, Any]]:
        issues = list(guidance["issues"])
        checks: list[dict[str, Any]] = []

        wrong_view = "wrong_view" in issues
        angle_status = (
            "review"
            if side_review_required and not wrong_view
            else "fail" if wrong_view else "pass"
        )
        checks.append({
            "code": "vlm_exact_view",
            "label": "六方向角度",
            "status": angle_status,
            "score": prediction.get("confidence"),
            "message": (
                "前後方向符合，但模型無法可靠區分左右；請依俯視站位圖確認拍攝側別"
                if angle_status == "review"
                else guidance["model_explanation"]
            ),
            "instruction": GUIDANCE_TEMPLATES["wrong_view"] if wrong_view else None,
            "mode": "qwen3_vl",
        })

        if expected_view in EXTERIOR_VIEWS:
            missing_plate = "plate_not_visible" in issues
            mismatched_plate = "plate_mismatch" in issues
            checks.append({
                "code": "vlm_target_plate",
                "label": "目標車牌",
                "status": "fail" if missing_plate else "pass",
                "score": 0 if missing_plate else 1,
                "message": (
                    "主要目標車的車牌未清楚入鏡"
                    if missing_plate
                    else "主要目標車的車牌清楚可見"
                ),
                "instruction": (
                    GUIDANCE_TEMPLATES["plate_not_visible"]
                    if missing_plate
                    else None
                ),
                "mode": "qwen3_vl_target_plate",
            })
            if not prediction.get("expected_plate"):
                checks.append({
                    "code": "vlm_plate_read",
                    "label": "車牌辨識",
                    "status": "pass" if prediction.get("plate_text") else "review",
                    "score": prediction.get("plate_confidence"),
                    "message": f"辨識結果：{prediction.get('plate_text') or '無法可靠讀取'}",
                    "instruction": None,
                    "mode": "qwen3_vl_plate_ocr",
                })
            if expected_plate := prediction.get("expected_plate"):
                checks.append({
                    "code": "vlm_plate_match",
                    "label": "訂單車牌",
                    "status": "fail" if mismatched_plate or missing_plate else "pass",
                    "score": prediction.get("plate_confidence"),
                    "message": (
                        f"辨識為 {prediction.get('plate_text') or '無法讀取'}；訂單為 {expected_plate}"
                    ),
                    "instruction": GUIDANCE_TEMPLATES["plate_mismatch"] if mismatched_plate else None,
                    "mode": "qwen3_vl_plate_ocr",
                })

        for issue in issues:
            if issue in {"wrong_view", "plate_not_visible", "plate_mismatch"}:
                continue
            checks.append({
                "code": f"vlm_{issue}",
                "label": ISSUE_LABELS.get(issue, issue),
                "status": "fail",
                "score": None,
                "message": guidance["model_explanation"],
                "instruction": GUIDANCE_TEMPLATES[issue],
                "mode": "qwen3_vl",
            })

        if not issues:
            checks.append({
                "code": "vlm_content",
                "label": "必要內容",
                "status": "pass",
                "score": prediction.get("confidence"),
                "message": guidance["model_explanation"],
                "instruction": None,
                "mode": "qwen3_vl",
            })
        return checks


TECHNICAL_CHECK_CODES = {"resolution", "sharpness", "exposure", "lens", "duplicate"}


def merge_capture_quality(cv_result: dict[str, Any], qwen_result: dict[str, Any]) -> dict[str, Any]:
    """Keep deterministic camera checks and use Qwen for semantic requirements."""

    technical_checks = [
        check
        for check in cv_result["metrics"]["checks"]
        if check["code"] in TECHNICAL_CHECK_CODES
    ]
    technical_failures = [
        check for check in technical_checks if check["status"] == "fail"
    ]
    passed = not technical_failures and bool(qwen_result["passed"])
    instructions = [
        check["instruction"]
        for check in technical_failures
        if check.get("instruction")
    ]
    if not qwen_result["passed"] and qwen_result.get("retake_instruction"):
        instructions.append(qwen_result["retake_instruction"])
    instructions = list(dict.fromkeys(instructions))

    result = dict(cv_result)
    result.update({
        "passed": passed,
        "detected_view": qwen_result["detected_view"],
        "plate_detected": qwen_result["plate_detected"],
        "retake_instruction": None if passed else " ".join(instructions),
        "warnings": [] if passed else instructions,
        "model_confidence": qwen_result["model_confidence"],
    })
    metrics = dict(cv_result["metrics"])
    metrics.update({
        "checks": [*technical_checks, *qwen_result["checks"]],
        "review_required": bool(qwen_result.get("review_required")) and passed,
        "capture_decision": (
            "retake"
            if not passed
            else "confirm" if qwen_result.get("review_required") else "accept"
        ),
        "inference_mode": "qwen3_vl_plus_cv",
        "quality_engine": "Qwen/Qwen3-VL-8B-Instruct",
        "vlm_issues": qwen_result["issues"],
        "vlm_explanation": qwen_result["model_explanation"],
        "vlm_inference_seconds": qwen_result["inference_seconds"],
        "vlm_gpu_peak_memory_mb": qwen_result["gpu_peak_memory_mb"],
        "plate_method": qwen_result["plate_method"],
        "plate_text": qwen_result.get("plate_text"),
        "plate_confidence": qwen_result.get("plate_confidence"),
        "expected_plate": qwen_result.get("expected_plate"),
        "plate_match": qwen_result.get("plate_match"),
        "angle_method": qwen_result.get("angle_method", "qwen3_vl_zero_shot"),
        "limitations": [
            "六角度模型判定位置必須與官方要求完全相同，否則一律重拍",
            "車外照片會辨識主要目標車牌並與訂單車牌比對；低畫質仍可能需要重拍",
        ],
    })
    result["metrics"] = metrics
    return result
