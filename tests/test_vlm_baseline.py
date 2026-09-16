from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.build_angle_eval_set import sample_manifest_rows, view_for_image_type
from scripts.evaluate_angle_vlm import calculate_metrics, evaluate_rows, load_completed
from src.vlm.json_parser import ModelOutputError, parse_model_output
from src.vlm.qwen_vl_client import InferenceResult


VALID_PREDICTION = {
    "detected_view": "left_front",
    "vehicle_visible": True,
    "plate_visible": True,
    "quality_passed": True,
    "quality_issues": [],
    "confidence": 0.92,
    "explanation": "車輛左前方完整入鏡",
    "retake_instruction": None,
}


def test_image_type_mapping():
    assert view_for_image_type(1) == "left_front"
    assert view_for_image_type("2") == "right_front"
    assert view_for_image_type(3) == "left_rear"
    assert view_for_image_type(4) == "right_rear"
    assert view_for_image_type(10) == "interior_front"
    assert view_for_image_type(11) == "interior_rear"
    assert view_for_image_type(7) is None


def test_json_parsing():
    assert parse_model_output(json.dumps(VALID_PREDICTION))["confidence"] == 0.92


def test_markdown_fence_cleanup():
    raw = "```json\n" + json.dumps(VALID_PREDICTION, ensure_ascii=False) + "\n```"
    assert parse_model_output(raw)["detected_view"] == "left_front"


def test_invalid_json_error():
    try:
        parse_model_output("not json")
    except ModelOutputError as error:
        assert "無法解析" in str(error)
    else:
        raise AssertionError("invalid JSON should raise ModelOutputError")


def test_manifest_sampling_is_stratified_and_unique_by_order(tmp_path: Path):
    rows = []
    for view_index, view in enumerate(("left_front", "right_front"), 1):
        for order_index in range(4):
            path = tmp_path / f"{view}-{order_index}.png"
            Image.new("RGB", (8, 8), (view_index * 20, 0, 0)).save(path)
            rows.append({
                "image_path": str(path), "order_number": f"order-{order_index}",
                "car_no": "CAR", "image_type": str(view_index),
                "ground_truth_view": view, "original_filename": path.name,
            })
    sampled, warnings = sample_manifest_rows(rows, per_class=3, seed=42)
    assert len(sampled) == 6
    for view in ("left_front", "right_front"):
        selected = [row for row in sampled if row["ground_truth_view"] == view]
        assert len(selected) == 3
        assert len({row["order_number"] for row in selected}) == 3
    assert any("interior_front" in warning for warning in warnings)


class FakeClient:
    def __init__(self):
        self.calls = 0

    def infer(self, image_path: Path) -> InferenceResult:
        self.calls += 1
        return InferenceResult(dict(VALID_PREDICTION), json.dumps(VALID_PREDICTION), True, 1.0, 100.0)


def test_resume_skips_completed_rows(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    rows = [
        {"image_path": str(tmp_path / f"{index}.png"), "order_number": str(index),
         "car_no": "CAR", "image_type": "1", "ground_truth_view": "left_front",
         "original_filename": f"{index}.png"}
        for index in range(2)
    ]
    first_client = FakeClient()
    evaluate_rows(rows, output, first_client, resume=False, limit=1)
    assert first_client.calls == 1
    second_client = FakeClient()
    results = evaluate_rows(rows, output, second_client, resume=True, limit=None)
    assert second_client.calls == 1
    assert len(results) == 2
    _, completed = load_completed(output / "predictions.jsonl")
    assert completed == {row["image_path"] for row in rows}


def test_metrics_calculation():
    rows = [
        {"ground_truth_view": "left_front", "detected_view": "left_front", "json_valid": True,
         "inference_success": True, "inference_seconds": 1.0, "gpu_peak_memory_mb": 100.0},
        {"ground_truth_view": "left_front", "detected_view": "right_front", "json_valid": True,
         "inference_success": True, "inference_seconds": 3.0, "gpu_peak_memory_mb": 200.0},
        {"ground_truth_view": "right_front", "detected_view": "right_front", "json_valid": True,
         "inference_success": True, "inference_seconds": 2.0, "gpu_peak_memory_mb": 150.0},
        {"ground_truth_view": "right_front", "json_valid": False, "inference_success": False,
         "inference_seconds": 4.0, "gpu_peak_memory_mb": 250.0},
    ]
    metrics, matrix = calculate_metrics(rows)
    assert metrics["accuracy"] == 2 / 3
    assert metrics["valid_json_rate"] == 0.75
    assert metrics["inference_success_rate"] == 0.75
    assert metrics["average_inference_seconds"] == 2.5
    assert metrics["peak_gpu_memory_mb"] == 250.0
    assert matrix[0][0] == 1 and matrix[0][1] == 1 and matrix[1][1] == 1
