from backend.app.services.angle_classifier import apply_angle_to_quality, apply_trained_angle


def qwen_result(*, issues=None, passed=True):
    return {
        "passed": passed,
        "review_required": False,
        "detected_view": "other",
        "model_confidence": 0.9,
        "issues": issues or [],
        "checks": [{"code": "vlm_exact_view"}, {"code": "vlm_content"}],
        "retake_instruction": None,
        "warnings": [],
    }


def angle(view, confidence):
    return {
        "detected_view": view,
        "confidence": confidence,
        "method": "human_trained_mobilenetv2_svm",
    }


def test_trained_angle_replaces_qwen_angle():
    result = apply_trained_angle(qwen_result(), angle("left_front", 0.88), "left_front", 0.72)
    assert result["passed"] is True
    assert result["detected_view"] == "left_front"
    assert result["checks"][0]["code"] == "trained_exact_view"
    assert result["checks"][0]["status"] == "pass"


def test_mirror_mismatch_requires_retake():
    result = apply_trained_angle(qwen_result(), angle("right_rear", 0.91), "left_rear", 0.72)
    assert result["passed"] is False
    assert result["review_required"] is False
    assert result["checks"][0]["status"] == "fail"
    assert result["issues"] == ["wrong_view"]


def test_high_confidence_non_mirror_mismatch_requires_retake():
    result = apply_trained_angle(qwen_result(), angle("right_rear", 0.91), "left_front", 0.72)
    assert result["passed"] is False
    assert result["issues"] == ["wrong_view"]
    assert result["retake_instruction"]


def test_low_confidence_cross_category_mismatch_requires_retake():
    result = apply_trained_angle(qwen_result(), angle("interior_rear", 0.5), "left_front", 0.72)
    assert result["passed"] is False
    assert result["review_required"] is False
    assert result["issues"] == ["wrong_view"]
    assert "車外拍攝欄位" in result["retake_instruction"]


def test_low_confidence_same_category_mismatch_requires_retake():
    result = apply_trained_angle(qwen_result(), angle("interior_rear", 0.5), "interior_front", 0.72)
    assert result["passed"] is False
    assert result["review_required"] is False
    assert result["issues"] == ["wrong_view"]


def test_very_low_confidence_mirror_mismatch_still_requires_retake():
    result = apply_trained_angle(qwen_result(), angle("right_front", 0.01), "left_front", 0.72)
    assert result["passed"] is False
    assert result["checks"][0]["status"] == "fail"


def test_qwen_wrong_view_cannot_be_softened_by_low_confidence_classifier():
    result = apply_trained_angle(
        qwen_result(issues=["wrong_view"], passed=False),
        angle("right_rear", 0.2),
        "interior_front",
        0.72,
    )
    assert result["passed"] is False
    assert result["review_required"] is False
    assert result["issues"] == ["wrong_view"]


def test_non_angle_qwen_failure_is_preserved():
    result = apply_trained_angle(
        qwen_result(issues=["blur"], passed=False),
        angle("left_front", 0.9),
        "left_front",
        0.72,
    )
    assert result["passed"] is False
    assert result["issues"] == ["blur"]


def cv_result(*, passed=True, review=True):
    return {
        "passed": passed,
        "detected_view": "exterior",
        "model_confidence": 0.5,
        "warnings": [],
        "retake_instruction": None,
        "metrics": {
            "checks": [
                {"code": "resolution", "status": "pass"},
                {"code": "exact_view", "status": "review"},
                *([{"code": "plate", "status": "review"}] if review else []),
            ],
            "review_required": True,
            "capture_decision": "confirm",
            "limitations": ["exact_six_view_unverified"],
        },
    }


def test_onnx_angle_accepts_matching_view_without_qwen():
    result = apply_angle_to_quality(
        cv_result(review=False),
        angle("left_front", 0.83),
        "left_front",
    )
    assert result["passed"] is True
    assert result["detected_view"] == "left_front"
    assert result["metrics"]["capture_decision"] == "accept"
    assert result["metrics"]["checks"][-1]["status"] == "pass"
    assert "exact_six_view_unverified" not in result["metrics"]["limitations"]


def test_onnx_angle_rejects_wrong_view_without_qwen():
    result = apply_angle_to_quality(
        cv_result(),
        angle("right_front", 0.91),
        "left_front",
    )
    assert result["passed"] is False
    assert result["metrics"]["capture_decision"] == "retake"
    assert result["retake_instruction"]
