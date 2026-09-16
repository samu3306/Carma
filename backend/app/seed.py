from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

from .config import get_settings
from .db import SessionLocal, init_db
from .models import (
    Alert,
    CleanlinessResult,
    DamageComparison,
    InspectionImage,
    InspectionSession,
    OperationTask,
    QualityCheckResult,
    RentalOrder,
    RiskAssessment,
    Vehicle,
    utcnow,
)

VIEWS = ["left_front", "right_front", "left_rear", "right_rear", "interior_front", "interior_rear"]
settings = get_settings()


def fixture(path: Path, label: str, dirty: bool = False, scratch: bool = False, blurry: bool = False) -> None:
    canvas = np.full((720, 1280, 3), (235, 237, 240), np.uint8)
    cv2.rectangle(canvas, (110, 80), (1170, 640), (207, 211, 216), -1)
    if "interior" in label:
        cv2.rectangle(canvas, (210, 170), (1070, 580), (58, 61, 67), -1)
        cv2.rectangle(canvas, (270, 300), (580, 555), (95, 99, 105), -1)
        cv2.rectangle(canvas, (700, 300), (1010, 555), (95, 99, 105), -1)
        if dirty:
            rng = np.random.default_rng(42)
            for _ in range(90):
                x, y = int(rng.integers(230, 1040)), int(rng.integers(200, 570))
                color = tuple(int(v) for v in rng.integers(20, 230, 3))
                cv2.circle(canvas, (x, y), int(rng.integers(3, 18)), color, -1)
    else:
        points = np.array([[230, 450], [320, 300], [860, 280], [1060, 420], [1090, 520], [180, 520]])
        cv2.fillPoly(canvas, [points], (225, 44, 52))
        cv2.rectangle(canvas, (390, 320), (820, 410), (70, 80, 90), -1)
        cv2.circle(canvas, (350, 520), 82, (38, 40, 44), -1)
        cv2.circle(canvas, (920, 520), 82, (38, 40, 44), -1)
        cv2.rectangle(canvas, (520, 455), (730, 505), (248, 248, 245), -1)
        cv2.putText(canvas, "DEMO-168", (545, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2)
        if scratch:
            cv2.line(canvas, (720, 390), (900, 440), (242, 242, 240), 10)
            cv2.line(canvas, (740, 405), (910, 450), (45, 45, 48), 3)
    cv2.putText(canvas, label, (130, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (34, 34, 38), 3)
    if blurry:
        canvas = cv2.GaussianBlur(canvas, (51, 51), 16)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def add_quality(db, image: InspectionImage, passed: bool = True) -> None:
    db.add(QualityCheckResult(
        image_id=image.id,
        passed=passed,
        quality_score=91 if passed else 28,
        blur_score=180 if passed else 8,
        brightness_score=133,
        vehicle_coverage=0.62,
        plate_detected=image.expected_view.startswith(("left", "right")),
        warnings=[] if passed else ["影像模糊，請穩定手機後重新拍攝"],
        retake_instruction=None if passed else "影像模糊，請穩定手機後重新拍攝",
        model_confidence=0.62,
        metrics={"inference_mode": "demo_seed"},
    ))


def seed() -> None:
    init_db()
    scenarios = [
        ("DEMO-BLUR", "RBL-1680", "模糊重拍", 46, "yellow"),
        ("DEMO-DAMAGE", "RDM-1681", "疑似新刮痕", 78, "red"),
        ("DEMO-DIRTY", "RDT-1682", "車內嚴重髒污", 73, "red"),
        ("DEMO-CLEAN", "RCL-1683", "合格自動完成", 12, "green"),
    ]
    with SessionLocal() as db:
        for number, plate, scenario, score, level in scenarios:
            if db.scalar(select(RentalOrder).where(RentalOrder.order_number == number)):
                continue
            vehicle = Vehicle(plate_number=plate, make_model="Toyota Yaris Demo")
            db.add(vehicle)
            db.flush()
            now = utcnow()
            order = RentalOrder(
                order_number=number,
                vehicle_id=vehicle.id,
                pickup_time=now - timedelta(hours=4),
                return_time=now,
                pickup_location="台北車站 Demo 站",
                return_location="台北車站 Demo 站",
                status="completed" if level == "green" else "attention_required",
            )
            db.add(order)
            db.flush()
            sessions = {}
            images = {}
            for inspection_type in ("pickup", "return"):
                session = InspectionSession(
                    order_id=order.id,
                    inspection_type=inspection_type,
                    location="台北車站 Demo 站",
                    status="completed",
                    started_at=now - timedelta(minutes=12 if inspection_type == "pickup" else 4),
                    completed_at=now - timedelta(minutes=9 if inspection_type == "pickup" else 1),
                )
                db.add(session)
                db.flush()
                sessions[inspection_type] = session
                for view in VIEWS:
                    is_dirty = number == "DEMO-DIRTY" and inspection_type == "return" and view.startswith("interior")
                    is_scratch = number == "DEMO-DAMAGE" and inspection_type == "return" and view == "left_front"
                    is_blur = number == "DEMO-BLUR" and inspection_type == "return" and view == "left_front"
                    path = settings.upload_dir / "demo" / number / f"{inspection_type}-{view}.jpg"
                    fixture(path, f"{scenario} {inspection_type} {view}", is_dirty, is_scratch, is_blur)
                    image = InspectionImage(
                        inspection_id=session.id,
                        order_id=order.id,
                        vehicle_id=vehicle.id,
                        inspection_type=inspection_type,
                        expected_view=view,
                        detected_view=view,
                        image_path=str(path),
                        quality_status="failed" if is_blur else "passed",
                        model_version="demo-seed-v1",
                    )
                    db.add(image)
                    db.flush()
                    add_quality(db, image, not is_blur)
                    images[(inspection_type, view)] = image

            damage_score = 0.44 if number == "DEMO-DAMAGE" else 0.04
            heatmap = settings.analysis_dir / f"{number}-left_front.jpg"
            fixture(heatmap, f"{scenario} 差異熱區", scratch=number == "DEMO-DAMAGE")
            damage = DamageComparison(
                order_id=order.id,
                baseline_image_id=images[("pickup", "left_front")].id,
                current_image_id=images[("return", "left_front")].id,
                view="left_front",
                difference_score=damage_score,
                ssim_score=0.56 if number == "DEMO-DAMAGE" else 0.96,
                suspected_new_damage=number == "DEMO-DAMAGE",
                severity="high" if number == "DEMO-DAMAGE" else "none",
                confidence=0.82,
                requires_manual_review=number == "DEMO-BLUR",
                issue_regions=[{"x": 720, "y": 390, "width": 190, "height": 70, "score": 0.88}] if number == "DEMO-DAMAGE" else [],
                heatmap_path=str(heatmap),
                explanation="Demo 影像比對發現新增刮痕熱區" if number == "DEMO-DAMAGE" else "Demo 影像比對未發現明顯差異",
            )
            db.add(damage)
            for view in ("interior_front", "interior_rear"):
                dirty = number == "DEMO-DIRTY"
                db.add(CleanlinessResult(
                    order_id=order.id,
                    image_id=images[("return", view)].id,
                    cleanliness_level="dirty" if dirty else "clean",
                    cleanliness_score=22 if dirty else 88,
                    detected_issues=["座椅與腳踏區疑似有大量雜物"] if dirty else [],
                    issue_regions=[],
                    immediate_cleaning_required=dirty,
                    confidence=0.68,
                    explanation="Demo 規則：高視覺雜訊，需立即清潔" if dirty else "Demo 規則：未見明顯髒污",
                ))
            reasons = {
                "DEMO-BLUR": ["左前照片模糊，需要補拍與人工複核"],
                "DEMO-DAMAGE": ["左前影像新增局部結構差異，疑似刮痕"],
                "DEMO-DIRTY": ["前後車內髒污分數超過立即清潔門檻"],
                "DEMO-CLEAN": ["六個角度完整且未發現超過門檻的異常"],
            }[number]
            assessment = RiskAssessment(
                order_id=order.id,
                inspection_id=sessions["return"].id,
                risk_score=score,
                risk_level=level,
                reasons=reasons,
                recommended_action="正常完成，可自動結案" if level == "green" else "立即派工處理" if level == "red" else "客服人工複核",
                components={"damage": damage_score * 100, "cleanliness": 78 if number == "DEMO-DIRTY" else 12, "quality": 72 if number == "DEMO-BLUR" else 9, "confidence": 32, "missing_views": 0, "manual_review": 100 if number == "DEMO-BLUR" else 0},
            )
            db.add(assessment)
            db.flush()
            if level != "green":
                db.add(Alert(order_id=order.id, risk_assessment_id=assessment.id, alert_type="demo_scenario", severity=level, message=reasons[0]))
            if number == "DEMO-DAMAGE":
                db.add(OperationTask(order_id=order.id, task_type="repair", status="pending", priority="urgent", notes="檢查左前刮痕並評估索賠"))
            if number == "DEMO-DIRTY":
                db.add(OperationTask(order_id=order.id, task_type="cleaning", status="assigned", priority="urgent", assignee="Demo 清潔班", notes="立即清潔車內"))
        db.commit()
    print("Demo seed ready: DEMO-BLUR, DEMO-DAMAGE, DEMO-DIRTY, DEMO-CLEAN")


if __name__ == "__main__":
    seed()
