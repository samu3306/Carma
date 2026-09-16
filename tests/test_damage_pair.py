from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.services.qwen_damage import QwenDamageDetector
from src.vlm.damage_pair import validate_damage_prediction
from src.vlm.json_parser import ModelOutputError
from src.vlm.qwen_vl_client import InferenceResult


def valid_prediction(**overrides):
    value = {
        "decision": "new_damage",
        "damage_type": "scratch",
        "severity": "medium",
        "vehicle_area": "右前保險桿",
        "confidence": 0.85,
        "changed_regions": ["新增線狀刮痕"],
        "explanation": "借車照沒有，還車照出現新刮痕",
    }
    value.update(overrides)
    return value


def test_damage_schema_accepts_valid_new_damage():
    result = validate_damage_prediction(valid_prediction())
    assert result["decision"] == "new_damage"
    assert result["damage_type"] == "scratch"


def test_damage_schema_normalizes_non_damage_fields():
    result = validate_damage_prediction(valid_prediction(
        decision="uncertain",
        damage_type="scratch",
        severity="medium",
    ))
    assert result["damage_type"] == "none"
    assert result["severity"] == "none"
    assert result["changed_regions"] == []


def test_damage_schema_rejects_new_damage_without_type():
    with pytest.raises(ModelOutputError):
        validate_damage_prediction(valid_prediction(damage_type="none"))


class FakeAnalyzer:
    def infer_pair(self, before_path: Path, after_path: Path):
        return InferenceResult(
            valid_prediction(), "{}", True, 1.2, 1000.0
        )


def test_qwen_damage_adapter_returns_backend_contract(tmp_path: Path):
    before = tmp_path / "before.jpg"
    after = tmp_path / "after.jpg"
    output = tmp_path / "comparison.jpg"
    cv2.imwrite(str(before), np.full((120, 200, 3), 80, np.uint8))
    cv2.imwrite(str(after), np.full((120, 200, 3), 100, np.uint8))

    result = QwenDamageDetector(FakeAnalyzer()).compare(before, after, output)

    assert output.is_file()
    assert result["suspected_new_damage"] is True
    assert result["requires_manual_review"] is True
    assert result["severity"] == "medium"
    assert result["inference_mode"] == "qwen3_vl_pair_plus_orb_ssim"
    assert "ssim_score" in result
    assert "alignment_confidence" in result
    assert "疑似新增刮痕" in result["explanation"]
