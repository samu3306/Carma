"""Schema and validation for VLM photo assessments."""

from __future__ import annotations

from typing import Any


DETECTED_VIEWS = (
    "left_front",
    "right_front",
    "left_rear",
    "right_rear",
    "interior_front",
    "interior_rear",
    "other",
)
QUALITY_ISSUES = (
    "blur",
    "too_dark",
    "overexposed",
    "too_far",
    "too_close",
    "vehicle_cropped",
    "wrong_view",
    "plate_not_visible",
    "plate_mismatch",
    "obstruction",
    "other",
)
REQUIRED_FIELDS = (
    "detected_view",
    "vehicle_visible",
    "plate_visible",
    "quality_passed",
    "quality_issues",
    "confidence",
    "explanation",
    "retake_instruction",
)


class PredictionValidationError(ValueError):
    """Raised when model JSON does not conform to the prediction schema."""


def validate_prediction(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PredictionValidationError("模型輸出必須是 JSON object")
    missing = [field for field in REQUIRED_FIELDS if field not in value]
    if missing:
        raise PredictionValidationError(f"缺少欄位：{', '.join(missing)}")

    view = value["detected_view"]
    if view not in DETECTED_VIEWS:
        raise PredictionValidationError(f"detected_view 不在允許清單：{view!r}")
    for field in ("vehicle_visible", "plate_visible", "quality_passed"):
        if type(value[field]) is not bool:
            raise PredictionValidationError(f"{field} 必須是 boolean")

    issues = value["quality_issues"]
    if not isinstance(issues, list) or any(issue not in QUALITY_ISSUES for issue in issues):
        raise PredictionValidationError("quality_issues 必須是允許值組成的 array")
    if len(set(issues)) != len(issues):
        raise PredictionValidationError("quality_issues 不可包含重複值")

    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise PredictionValidationError("confidence 必須是 0 到 1 的數字")
    if not 0.0 <= float(confidence) <= 1.0:
        raise PredictionValidationError("confidence 必須介於 0 與 1")
    if not isinstance(value["explanation"], str) or not value["explanation"].strip():
        raise PredictionValidationError("explanation 必須是非空字串")
    retake = value["retake_instruction"]
    if retake is not None and (not isinstance(retake, str) or not retake.strip()):
        raise PredictionValidationError("retake_instruction 必須是非空字串或 null")
    if value["quality_passed"] and retake is not None:
        raise PredictionValidationError("quality_passed=true 時 retake_instruction 必須是 null")

    result = {
        "detected_view": view,
        "vehicle_visible": value["vehicle_visible"],
        "plate_visible": value["plate_visible"],
        "quality_passed": value["quality_passed"],
        "quality_issues": list(issues),
        "confidence": float(confidence),
        "explanation": value["explanation"].strip(),
        "retake_instruction": retake.strip() if isinstance(retake, str) else None,
    }
    plate_text = value.get("plate_text")
    if plate_text is not None and not isinstance(plate_text, str):
        raise PredictionValidationError("plate_text 必須是字串或 null")
    plate_confidence = value.get("plate_confidence")
    if plate_confidence is not None:
        if isinstance(plate_confidence, bool) or not isinstance(plate_confidence, (int, float)):
            raise PredictionValidationError("plate_confidence 必須是 0 到 1 的數字或 null")
        if not 0 <= float(plate_confidence) <= 1:
            raise PredictionValidationError("plate_confidence 必須介於 0 與 1")
    result["plate_text"] = plate_text.strip().upper() if isinstance(plate_text, str) else None
    result["plate_confidence"] = float(plate_confidence) if plate_confidence is not None else None
    return result
