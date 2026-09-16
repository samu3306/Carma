"""Deterministic retake guidance derived from validated VLM issue codes."""

from __future__ import annotations

from typing import Any

from .schemas import DETECTED_VIEWS, QUALITY_ISSUES


EXTERIOR_VIEWS = {"left_front", "right_front", "left_rear", "right_rear"}
INTERIOR_VIEWS = {"interior_front", "interior_rear"}
VIEW_LABELS = {
    "left_front": "左前",
    "right_front": "右前",
    "left_rear": "左後",
    "right_rear": "右後",
    "interior_front": "車前座",
    "interior_rear": "車後座",
    "other": "其他",
}
GUIDANCE_TEMPLATES = {
    "blur": "請穩定手機並重新拍攝，確認車身與車牌清晰。",
    "too_dark": "請移至光線充足的位置，避免在昏暗環境拍攝。",
    "overexposed": "請避開強光直射，點選車身重新對焦後拍攝。",
    "too_far": "請靠近車輛，讓車身占畫面主要區域。",
    "too_close": "請往後退，確保指定角度的車身完整入鏡。",
    "vehicle_cropped": "請往後退並調整構圖，確保車輛主要區域完整入鏡。",
    "wrong_view": "目前角度不符合指定位置，請依畫面示範重新拍攝。",
    "plate_not_visible": "請調整距離與角度，確認車牌清楚可見。",
    "plate_mismatch": "照片中的車牌與本次訂單車牌不符，請確認車輛後重新拍攝。",
    "obstruction": "請移除遮擋物或更換拍攝位置後重新拍攝。",
    "other": "照片不符合取還車紀錄需求，請依標準示範重新拍攝。",
}


def build_photo_guidance(
    prediction: dict[str, Any],
    expected_view: str,
    angle_review_required: bool = False,
) -> dict[str, Any]:
    if expected_view not in DETECTED_VIEWS:
        raise ValueError(f"不支援的預期角度：{expected_view}")
    issues = [issue for issue in prediction.get("quality_issues", []) if issue in QUALITY_ISSUES]
    ignored_plate_issue = expected_view in INTERIOR_VIEWS and "plate_not_visible" in issues
    ignored_angle_issue = angle_review_required and "wrong_view" in issues
    if angle_review_required:
        issues = [issue for issue in issues if issue != "wrong_view"]
    if expected_view in INTERIOR_VIEWS:
        # A licence plate is not part of either cabin-photo requirement. The VLM
        # can still emit this exterior-only issue, so enforce the business rule
        # deterministically after inference.
        issues = [issue for issue in issues if issue != "plate_not_visible"]
    if (
        not angle_review_required
        and prediction.get("detected_view") != expected_view
        and "wrong_view" not in issues
    ):
        issues.append("wrong_view")
    if not prediction.get("vehicle_visible", False) and "other" not in issues:
        issues.append("other")
    if (
        expected_view in EXTERIOR_VIEWS
        and not prediction.get("plate_visible", False)
        and "plate_not_visible" not in issues
    ):
        issues.append("plate_not_visible")
    if (
        not prediction.get("quality_passed", False)
        and not issues
        and not ignored_plate_issue
        and not ignored_angle_issue
    ):
        issues.append("other")
    issues = list(dict.fromkeys(issues))
    quality_passed = (
        bool(prediction.get("quality_passed")) or ignored_plate_issue or ignored_angle_issue
    )
    review_required = bool(angle_review_required and not issues and quality_passed)
    passed = not issues and quality_passed and not review_required
    decision = "retake" if issues or not quality_passed else ("review" if review_required else "pass")
    instructions = [GUIDANCE_TEMPLATES[issue] for issue in issues]
    if "vehicle_cropped" in issues and expected_view == "interior_front":
        instructions[issues.index("vehicle_cropped")] = (
            "請拉遠並重新構圖，清楚拍到方向盤與儀表區、中控區，以及至少一張前座的主要椅面與椅背。"
        )
    elif "vehicle_cropped" in issues and expected_view == "interior_rear":
        instructions[issues.index("vehicle_cropped")] = (
            "請拉遠並重新構圖，讓完整後排座椅的椅面與椅背入鏡。"
        )
    explanation = prediction.get("explanation")
    if ignored_plate_issue:
        explanation = (
            "車前座與車後座照片不檢查車牌；"
            + ("未發現其他需重拍問題。" if passed else "已忽略不適用的車牌問題，並依其他問題判定。")
        )
    if "wrong_view" in issues:
        detected_label = VIEW_LABELS.get(prediction.get("detected_view"), "無法辨識")
        expected_label = VIEW_LABELS.get(expected_view, expected_view)
        other_issues = [issue for issue in issues if issue != "wrong_view"]
        explanation = (
            f"照片的清晰度或車牌可能符合要求，但偵測角度為「{detected_label}」，"
            f"與預期的「{expected_label}」不符，因此整體判定需要重拍。"
        )
        if other_issues:
            explanation += f" 另外偵測到：{'、'.join(other_issues)}。"

    return {
        "expected_view": expected_view,
        "detected_view": prediction.get("detected_view"),
        "passed": passed,
        "decision": decision,
        "review_required": review_required,
        "issues": issues,
        "retake_instruction": (
            None
            if passed
            else (
                "左右角度目前無法可靠自動判定，請由人工確認是否符合指定側別。"
                if review_required
                else " ".join(instructions)
            )
        ),
        "model_confidence": prediction.get("confidence"),
        "model_explanation": explanation,
    }
