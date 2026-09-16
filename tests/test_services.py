from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.app.services.damage import DemoDamageDetector
from backend.app.services.quality import DemoQualityAnalyzer
from backend.app.services.risk import assess_risk
from backend.app.services.vehicle import YoloVehicleDetector


def textured(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(45, 215, (540, 960, 3), dtype=np.uint8)
    cv2.rectangle(image, (180, 170), (780, 430), (210, 35, 44), -1)
    cv2.putText(image, f"CAR {seed}", (310, 330), cv2.FONT_HERSHEY_SIMPLEX, 2, (240, 240, 240), 6)
    return image


def test_quality_reports_specific_blur_instruction(tmp_path: Path):
    path = tmp_path / "blur.jpg"
    cv2.imwrite(str(path), np.full((600, 900, 3), 120, np.uint8))
    result = DemoQualityAnalyzer().analyze(path, "left_front", [])
    assert result["passed"] is False
    assert "影像模糊" in result["retake_instruction"]
    assert result["metrics"]["inference_mode"] == "demo_rules"


def test_damage_pipeline_creates_visualization(tmp_path: Path):
    baseline = textured(8)
    current = baseline.copy()
    cv2.line(current, (500, 260), (720, 360), (250, 250, 250), 14)
    base_path, current_path, output = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "heat.jpg"
    cv2.imwrite(str(base_path), baseline)
    cv2.imwrite(str(current_path), current)
    result = DemoDamageDetector().compare(base_path, current_path, output)
    assert output.exists()
    assert 0 <= result["difference_score"] <= 1
    assert "ssim_score" in result
    assert result["inference_mode"] == "demo_cv_pipeline"


def test_risk_thresholds_and_reasons():
    low = assess_risk(0.01, 92, 95, 0.9, 0, False)
    high = assess_risk(0.95, 15, 55, 0.4, 2, True)
    assert low["risk_level"] == "green"
    assert high["risk_level"] == "red"
    assert len(high["reasons"]) >= 3
class StubVehicleDetector:
    def __init__(self, result: dict):
        self.result = result

    def detect(self, _image: np.ndarray, retry_rotations: bool = True) -> dict:
        return self.result


def checks_by_code(result: dict) -> dict:
    return {item["code"]: item for item in result["metrics"]["checks"]}


def test_capture_review_rejects_exterior_without_vehicle(tmp_path: Path):
    path = tmp_path / "no-car.jpg"
    cv2.imwrite(str(path), textured(11))
    detector = StubVehicleDetector({"available": True, "detected": False, "confidence": 0.0, "bbox": None})
    result = DemoQualityAnalyzer(detector).analyze(path, "left_front", [])
    assert result["passed"] is False
    assert result["metrics"]["capture_decision"] == "retake"
    assert checks_by_code(result)["vehicle"]["status"] == "fail"


def test_capture_review_rejects_vehicle_that_is_too_far(tmp_path: Path):
    path = tmp_path / "far.jpg"
    cv2.imwrite(str(path), textured(12))
    detector = StubVehicleDetector({
        "available": True,
        "detected": True,
        "confidence": 0.8,
        "bbox": {"x": 400, "y": 220, "width": 120, "height": 100},
        "coverage": 0.025,
        "touching_edges": [],
    })
    result = DemoQualityAnalyzer(detector).analyze(path, "right_rear", [])
    assert result["passed"] is False
    assert checks_by_code(result)["distance"]["status"] == "fail"
    assert "太遠" in result["retake_instruction"]


def test_capture_review_rejects_exterior_photo_in_interior_slot(tmp_path: Path):
    path = tmp_path / "wrong-scene.jpg"
    cv2.imwrite(str(path), textured(13))
    detector = StubVehicleDetector({
        "available": True,
        "detected": True,
        "confidence": 0.87,
        "bbox": {"x": 160, "y": 100, "width": 650, "height": 340},
        "coverage": 0.43,
        "touching_edges": [],
    })
    result = DemoQualityAnalyzer(detector).analyze(path, "interior_front", [])
    assert result["passed"] is False
    assert checks_by_code(result)["scene"]["status"] == "fail"
    assert "車內照片" in result["retake_instruction"]


def test_capture_review_returns_transparent_pending_capabilities(tmp_path: Path):
    path = tmp_path / "structured.jpg"
    cv2.imwrite(str(path), textured(14))
    detector = StubVehicleDetector({
        "available": True,
        "detected": True,
        "confidence": 0.9,
        "bbox": {"x": 180, "y": 120, "width": 600, "height": 300},
        "coverage": 0.35,
        "touching_edges": [],
    })
    result = DemoQualityAnalyzer(detector).analyze(path, "left_rear", [])
    checks = checks_by_code(result)
    assert checks["vehicle"]["mode"] == "yolov4_tiny"
    assert checks["exact_view"]["status"] == "review"
    assert checks["exact_view"]["mode"] == "model_pending"
    assert result["metrics"]["capture_decision"] == "confirm"
    assert "plate_is_shape_candidate_not_ocr" in result["metrics"]["limitations"]
def test_interior_portrait_with_full_coverage_requires_confirmation_not_retake(tmp_path: Path):
    path = tmp_path / "portrait-interior.jpg"
    cv2.imwrite(str(path), np.rot90(textured(21)))
    result = DemoQualityAnalyzer().analyze(path, "interior_front", [])
    orientation = checks_by_code(result)["orientation"]
    assert result["metrics"]["orientation"] == "portrait"
    assert orientation["status"] == "review"
    assert orientation["instruction"] is None
    assert result["passed"] is True


def test_exterior_accepts_portrait_when_vehicle_is_complete(tmp_path: Path):
    path = tmp_path / "portrait-exterior.jpg"
    cv2.imwrite(str(path), np.rot90(textured(22)))
    detector = StubVehicleDetector({
        "available": True,
        "detected": True,
        "confidence": 0.9,
        "bbox": {"x": 80, "y": 180, "width": 380, "height": 540},
        "coverage": 0.4,
        "touching_edges": [],
    })
    result = DemoQualityAnalyzer(detector).analyze(path, "left_front", [])
    assert result["metrics"]["orientation"] == "portrait"
    assert checks_by_code(result)["orientation"]["status"] == "pass"
    assert result["passed"] is True
def test_vehicle_detector_recovers_rotation_when_primary_fails(monkeypatch):
    detector = YoloVehicleDetector.__new__(YoloVehicleDetector)
    detector.net = object()
    attempted: list[str] = []

    def fake_detect_once(_image: np.ndarray, rotation_applied: str = "none") -> dict:
        attempted.append(rotation_applied)
        if rotation_applied == "counterclockwise_90":
            return {
                "available": True,
                "detected": True,
                "confidence": 0.81,
                "coverage": 0.47,
                "bbox": {"x": 10, "y": 20, "width": 300, "height": 180},
                "rotation_applied": rotation_applied,
            }
        return {
            "available": True,
            "detected": False,
            "confidence": 0.0,
            "coverage": 0.0,
            "bbox": None,
            "rotation_applied": rotation_applied,
        }

    monkeypatch.setattr(detector, "_detect_once", fake_detect_once)
    result = detector.detect(np.zeros((1280, 720, 3), dtype=np.uint8))
    assert result["detected"] is True
    assert result["rotation_applied"] == "counterclockwise_90"
    assert result["rotation_retried"] is True
    assert attempted == ["none", "counterclockwise_90", "clockwise_90", "rotate_180"]
def test_rotated_large_vehicle_in_interior_slot_is_low_score_retake(tmp_path: Path):
    path = tmp_path / "rotated-exterior-in-interior.jpg"
    cv2.imwrite(str(path), textured(31))
    detector = StubVehicleDetector({
        "available": True,
        "detected": True,
        "confidence": 0.23,
        "bbox": {"x": 30, "y": 20, "width": 700, "height": 420},
        "coverage": 0.66,
        "touching_edges": ["top"],
        "rotation_applied": "counterclockwise_90",
        "rotation_retried": True,
    })
    result = DemoQualityAnalyzer(detector).analyze(path, "interior_front", [])
    checks = checks_by_code(result)
    assert result["passed"] is False
    assert result["quality_score"] == 20.0
    assert result["detected_view"] == "exterior"
    assert checks["scene"]["status"] == "fail"
    assert checks["interior_coverage"]["mode"] == "not_applicable"
    assert checks["cleanliness_preview"]["mode"] == "not_applicable"
