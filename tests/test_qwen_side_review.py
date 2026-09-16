from pathlib import Path

from backend.app.services.qwen_quality import (
    QwenCaptureQualityAnalyzer,
    merge_capture_quality,
)
from src.vlm.qwen_vl_client import InferenceResult


def model_result(prediction):
    return InferenceResult(prediction, "{}", True, 0.2, 100.0)


class FakeClient:
    def __init__(self, results):
        self.results = iter(results)

    def infer(self, image_path: Path, **kwargs):
        return next(self.results)


def prediction(view, *, issues=None, passed=True):
    return {
        "detected_view": view,
        "vehicle_visible": True,
        "plate_visible": True,
        "quality_passed": passed,
        "quality_issues": issues or [],
        "confidence": 0.9,
        "explanation": "測試判斷",
    }


def test_same_end_left_right_mismatch_requires_confirmation_not_retake():
    client = FakeClient([
        model_result(prediction("left_front", issues=["wrong_view"], passed=False)),
        model_result(prediction("other")),
    ])
    analyzer = QwenCaptureQualityAnalyzer("test-model", client=client)

    result = analyzer.analyze(Path("photo.jpg"), "right_front")

    assert result["passed"] is True
    assert result["review_required"] is True
    assert result["retake_instruction"] is None
    angle_check = next(c for c in result["checks"] if c["code"] == "vlm_exact_view")
    assert angle_check["status"] == "review"
    assert "確認拍攝側別" in angle_check["message"]


def test_merge_turns_side_review_into_confirmation():
    cv_result = {
        "passed": True,
        "metrics": {
            "checks": [
                {"code": "sharpness", "status": "pass", "instruction": None},
            ]
        },
    }
    qwen_result = {
        "passed": True,
        "review_required": True,
        "detected_view": "left_rear",
        "plate_detected": True,
        "retake_instruction": None,
        "model_confidence": 0.8,
        "checks": [{"code": "vlm_exact_view", "status": "review"}],
        "issues": [],
        "model_explanation": "左右待確認",
        "inference_seconds": 1.0,
        "gpu_peak_memory_mb": 100,
        "plate_method": "target_vehicle_second_pass",
    }

    merged = merge_capture_quality(cv_result, qwen_result)

    assert merged["passed"] is True
    assert merged["metrics"]["review_required"] is True
    assert merged["metrics"]["capture_decision"] == "confirm"
