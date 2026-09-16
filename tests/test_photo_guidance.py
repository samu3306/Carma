from src.vlm.photo_guidance import build_photo_guidance


def prediction(**overrides):
    value = {
        "detected_view": "left_front",
        "vehicle_visible": True,
        "plate_visible": True,
        "quality_passed": True,
        "quality_issues": [],
        "confidence": 0.9,
        "explanation": "測試",
    }
    value.update(overrides)
    return value


def test_passed_photo_has_no_retake_instruction():
    result = build_photo_guidance(prediction(), "left_front")
    assert result["passed"] is True
    assert result["issues"] == []
    assert result["retake_instruction"] is None


def test_fixed_guidance_combines_model_and_derived_issues():
    result = build_photo_guidance(
        prediction(
            detected_view="right_front",
            plate_visible=False,
            quality_passed=False,
            quality_issues=["blur"],
        ),
        "left_front",
    )
    assert result["passed"] is False
    assert result["issues"] == ["blur", "wrong_view", "plate_not_visible"]
    assert "穩定手機" in result["retake_instruction"]
    assert "指定位置" in result["retake_instruction"]
    assert "車牌" in result["retake_instruction"]
    assert "偵測角度為「右前」" in result["model_explanation"]
    assert "預期的「左前」" in result["model_explanation"]


def test_interior_does_not_require_plate():
    result = build_photo_guidance(
        prediction(detected_view="interior_front", plate_visible=False),
        "interior_front",
    )
    assert result["passed"] is True
    assert "plate_not_visible" not in result["issues"]


def test_interior_ignores_plate_issue_and_plate_only_failure():
    result = build_photo_guidance(
        prediction(
            detected_view="interior_rear",
            plate_visible=False,
            quality_passed=False,
            quality_issues=["plate_not_visible"],
            explanation="車牌不可見。",
        ),
        "interior_rear",
    )
    assert result["passed"] is True
    assert result["issues"] == []
    assert result["retake_instruction"] is None
    assert "不檢查車牌" in result["model_explanation"]


def test_interior_keeps_non_plate_issues():
    result = build_photo_guidance(
        prediction(
            detected_view="interior_front",
            quality_passed=False,
            quality_issues=["plate_not_visible", "too_dark"],
        ),
        "interior_front",
    )
    assert result["passed"] is False
    assert result["issues"] == ["too_dark"]
    assert "光線充足" in result["retake_instruction"]


def test_interior_front_cropped_guidance_names_required_seats():
    result = build_photo_guidance(
        prediction(
            detected_view="interior_front",
            quality_passed=False,
            quality_issues=["vehicle_cropped"],
        ),
        "interior_front",
    )
    assert result["passed"] is False
    assert "至少一張前座" in result["retake_instruction"]


def test_exterior_missing_plate_is_included_with_other_model_issue():
    result = build_photo_guidance(
        prediction(
            detected_view="right_rear",
            plate_visible=False,
            quality_passed=False,
            quality_issues=["obstruction"],
        ),
        "right_rear",
    )
    assert result["issues"] == ["obstruction", "plate_not_visible"]
    assert "車牌清楚可見" in result["retake_instruction"]


def test_unvalidated_exterior_angle_requires_review_not_retake():
    result = build_photo_guidance(
        prediction(detected_view="left_rear"),
        "right_rear",
        angle_review_required=True,
    )
    assert result["passed"] is False
    assert result["decision"] == "review"
    assert result["review_required"] is True
    assert result["issues"] == []
    assert "人工確認" in result["retake_instruction"]


def test_model_wrong_view_issue_is_reviewed_not_used_for_exterior_retake():
    result = build_photo_guidance(
        prediction(
            detected_view="left_rear",
            quality_passed=False,
            quality_issues=["wrong_view"],
        ),
        "right_rear",
        angle_review_required=True,
    )
    assert result["decision"] == "review"
    assert result["issues"] == []
    assert result["review_required"] is True
