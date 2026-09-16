from __future__ import annotations

from ..config import get_risk_config


def assess_risk(
    damage_score: float,
    cleanliness_score: float,
    average_quality: float,
    average_confidence: float,
    missing_count: int,
    requires_manual_review: bool,
) -> dict:
    config = get_risk_config()
    weights = config["weights"]
    damage_risk = max(0.0, min(100.0, damage_score * 100))
    dirty_risk = max(0.0, min(100.0, 100 - cleanliness_score))
    quality_risk = max(0.0, min(100.0, 100 - average_quality))
    confidence_risk = max(0.0, min(100.0, 100 - average_confidence * 100))
    missing_risk = min(100.0, missing_count / 6 * 100)
    manual_risk = 100.0 if requires_manual_review else 0.0
    components = {
        "damage": round(damage_risk, 1),
        "cleanliness": round(dirty_risk, 1),
        "quality": round(quality_risk, 1),
        "confidence": round(confidence_risk, 1),
        "missing_views": round(missing_risk, 1),
        "manual_review": round(manual_risk, 1),
    }
    score = round(sum(components[key] * weights[key] for key in weights))
    reasons: list[str] = []
    if damage_risk >= 12:
        reasons.append(f"影像差異風險 {damage_risk:.0f}/100")
    if dirty_risk >= 45:
        reasons.append(f"車內髒污風險 {dirty_risk:.0f}/100")
    if average_quality < 70:
        reasons.append(f"平均照片品質僅 {average_quality:.0f}/100")
    if missing_count:
        reasons.append(f"缺少 {missing_count} 個必要拍攝位置")
    if requires_manual_review:
        reasons.append("影像角度或模型信心需要人工複核")
    if not reasons:
        reasons.append("照片完整且未發現超過門檻的異常")
    if score <= config["thresholds"]["green_max"]:
        level, action = "green", "正常完成，可自動結案"
    elif score <= config["thresholds"]["yellow_max"]:
        level, action = "yellow", "送客服人工複核後再結案"
    else:
        level, action = "red", "立即預警、暫停再次派車並建立營運任務"
    return {
        "risk_score": score,
        "risk_level": level,
        "reasons": reasons,
        "recommended_action": action,
        "components": components,
    }
