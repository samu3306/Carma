from pathlib import Path

from backend.app.services.qwen_quality import (
    QwenCaptureQualityAnalyzer,
    merge_capture_quality,
)
from src.vlm.qwen_vl_client import InferenceResult


def inference(prediction: dict, seconds: float = 0.2) -> InferenceResult:
    return InferenceResult(
        prediction=prediction,
        raw_output="{}",
        json_valid=True,
        inference_seconds=seconds,
        gpu_peak_memory_mb=100.0,
    )


class FakeClient:
    def __init__(self, results: list[InferenceResult]):
        self.results = iter(results)
        self.calls = []
        self.loaded = False

    def load(self):
        self.loaded = True

    def infer(self, image_path: Path, **kwargs):
        self.calls.append((image_path, kwargs))
        return next(self.results)


def prediction(view: str, *, plate: bool = True, issues=None, passed=True, plate_text=None):
    result = {
        "detected_view": view,
        "vehicle_visible": True,
        "plate_visible": plate,
        "quality_passed": passed,
        "quality_issues": issues or [],
        "confidence": 0.9,
        "explanation": "測試判斷",
    }
    if plate_text is not None:
        result.update({"plate_text": plate_text, "plate_confidence": 0.94})
    return result


def test_exterior_uses_target_plate_second_pass():
    client = FakeClient([
        inference(prediction("left_rear", plate=True)),
        inference(prediction("left_rear", plate=False, passed=False)),
    ])
    analyzer = QwenCaptureQualityAnalyzer("test-model", client=client)

    result = analyzer.analyze(Path("photo.jpg"), "left_rear")

    assert result["passed"] is False
    assert result["plate_detected"] is False
    assert "plate_not_visible" in result["issues"]
    assert result["plate_method"] == "target_vehicle_second_pass"
    assert client.calls[1][1]["prompt_text"]


def test_interior_ignores_plate_and_skips_plate_second_pass():
    client = FakeClient([
        inference(
            prediction(
                "interior_front",
                plate=False,
                issues=["plate_not_visible"],
                passed=False,
            )
        )
    ])
    analyzer = QwenCaptureQualityAnalyzer("test-model", client=client)

    result = analyzer.analyze(Path("cabin.jpg"), "interior_front")

    assert result["passed"] is True
    assert result["plate_detected"] is False
    assert result["issues"] == []
    assert result["plate_method"] == "not_applicable"
    assert len(client.calls) == 1


def test_exterior_plate_text_matches_normalized_order_plate():
    client = FakeClient([
        inference(prediction("left_front")),
        inference(prediction("other", plate_text="RFY-6613")),
    ])
    analyzer = QwenCaptureQualityAnalyzer("test-model", client=client)

    result = analyzer.analyze(Path("photo.jpg"), "left_front", "rfy6613")

    assert result["passed"] is True
    assert result["plate_text"] == "RFY-6613"
    assert result["plate_match"] is True
    assert any(check["code"] == "vlm_plate_match" for check in result["checks"])


def test_exterior_plate_text_mismatch_requires_retake():
    client = FakeClient([
        inference(prediction("right_rear")),
        inference(prediction("other", plate_text="ABC-1234")),
    ])
    analyzer = QwenCaptureQualityAnalyzer("test-model", client=client)

    result = analyzer.analyze(Path("photo.jpg"), "right_rear", "RFY-6613")

    assert result["passed"] is False
    assert result["plate_match"] is False
    assert "plate_mismatch" in result["issues"]
    check = next(check for check in result["checks"] if check["code"] == "vlm_plate_match")
    assert check["status"] == "fail"


def test_merge_replaces_old_semantic_checks_but_keeps_technical_checks():
    cv_result = {
        "passed": False,
        "quality_score": 80,
        "blur_score": 100,
        "brightness_score": 90,
        "vehicle_coverage": 0.5,
        "plate_detected": False,
        "warnings": ["舊規則"],
        "retake_instruction": "舊規則",
        "model_confidence": 0.5,
        "detected_view": "exterior",
        "metrics": {
            "checks": [
                {"code": "resolution", "status": "pass", "instruction": None},
                {"code": "exact_view", "status": "fail", "instruction": "舊角度規則"},
                {"code": "plate", "status": "fail", "instruction": "舊車牌規則"},
            ]
        },
    }
    qwen_result = {
        "passed": True,
        "detected_view": "right_front",
        "plate_detected": True,
        "retake_instruction": None,
        "warnings": [],
        "model_confidence": 0.95,
        "checks": [{"code": "vlm_content", "status": "pass"}],
        "issues": [],
        "model_explanation": "符合標準",
        "inference_seconds": 1.5,
        "gpu_peak_memory_mb": 1234,
        "plate_method": "target_vehicle_second_pass",
    }

    merged = merge_capture_quality(cv_result, qwen_result)

    assert merged["passed"] is True
    assert merged["metrics"]["capture_decision"] == "accept"
    assert [check["code"] for check in merged["metrics"]["checks"]] == [
        "resolution",
        "vlm_content",
    ]
