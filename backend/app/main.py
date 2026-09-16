from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import get_risk_config, get_settings
from .db import get_db, init_db
from .models import (
    Alert,
    CleanlinessResult,
    DamageEvidence,
    DamageComparison,
    DatasetAnnotation,
    HumanReview,
    KnownDamage,
    KnownDamageAcknowledgement,
    InspectionImage,
    InspectionSession,
    OperationTask,
    QualityCheckResult,
    RentalOrder,
    RiskAssessment,
    Vehicle,
    utcnow,
)
from .schemas import DatasetAnnotationUpdate, EvidencePurpose, InspectionCreate, OrderCreate, ReviewCreate, TaskCreate, TaskUpdate, ViewName
from .services.cleanliness import DemoCleanlinessClassifier
from .services.angle_classifier import (
    DinoV2AngleClassifier,
    DinoV2OnnxAngleClassifier,
    TrainedAngleClassifier,
    apply_angle_to_quality,
    apply_trained_angle,
)
from .services.damage import DemoDamageDetector
from .services.labeling import annotation_dict, export_annotations_csv, labeling_stats, sync_manifest
from .services.quality import DemoQualityAnalyzer
from .services.qwen_quality import QwenCaptureQualityAnalyzer, merge_capture_quality
from .services.qwen_damage import QwenDamageDetector
from src.vlm.damage_pair import QwenDamagePairAnalyzer, load_damage_references
from src.vlm.qwen_vl_client import QwenVLClient
from .services.risk import assess_risk
from .services.vehicle import YoloVehicleDetector

settings = get_settings()
required_views = get_risk_config()["required_views"]


def required_views_for(inspection_type: str) -> list[str]:
    """Pickup records the four exterior baselines; return also checks both cabin views."""
    return required_views[:4] if inspection_type == "pickup" else required_views

vehicle_detector = YoloVehicleDetector(settings.yolo_config_path, settings.yolo_weights_path) if settings.vehicle_detector_enabled else None
quality_analyzer = DemoQualityAnalyzer(vehicle_detector)
qwen_client = (
    QwenVLClient(settings.qwen_quality_model)
    if settings.qwen_quality_enabled or settings.qwen_damage_enabled
    else None
)
qwen_quality_analyzer = (
    QwenCaptureQualityAnalyzer(settings.qwen_quality_model, client=qwen_client)
    if settings.qwen_quality_enabled
    else None
)
def build_angle_classifier():
    if not settings.angle_classifier_enabled:
        return None
    if settings.angle_classifier_backend == "dinov2_onnx":
        return DinoV2OnnxAngleClassifier(
            settings.dinov2_onnx_path,
            settings.dinov2_angle_classifier_dir,
        )
    if settings.angle_classifier_backend == "dinov2":
        return DinoV2AngleClassifier(
            settings.dinov2_model_name,
            settings.dinov2_cache_dir,
            settings.dinov2_angle_classifier_dir,
            settings.dinov2_device,
        )
    return TrainedAngleClassifier(
        settings.angle_feature_extractor_path, settings.angle_classifier_dir
    )


angle_classifier = build_angle_classifier()
qwen_damage_analyzer = (
    QwenDamagePairAnalyzer(
        qwen_client, load_damage_references(settings.qwen_damage_few_shot_manifest)
    )
    if settings.qwen_damage_enabled
    else None
)
damage_detector = (
    QwenDamageDetector(qwen_damage_analyzer)
    if qwen_damage_analyzer is not None
    else DemoDamageDetector()
)
cleanliness_classifier = DemoCleanlinessClassifier()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    if angle_classifier is not None:
        warmup = getattr(angle_classifier, "warmup", None)
        if warmup is not None:
            warmup()
    if qwen_client is not None:
        qwen_client.load()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="黑客松 MVP。所有影像模型輸出皆為明確標示的 Demo 規則或 CV pipeline。",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")



def error_payload(code: str, message: str, details=None):
    return {"error": {"code": code, "message": message, "details": details}}


@app.exception_handler(RequestValidationError)
async def validation_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=error_payload("validation_error", "輸入資料驗證失敗", exc.errors()))


@app.exception_handler(HTTPException)
async def http_handler(_request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        content = exc.detail
    else:
        content = error_payload("request_error", str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def unexpected_handler(_request: Request, exc: Exception):
    return JSONResponse(status_code=500, content=error_payload("internal_error", "系統處理失敗", str(exc)))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "demo_mode": settings.demo_mode,
        "model_version": settings.model_version,
        "qwen_quality_enabled": settings.qwen_quality_enabled,
        "qwen_quality_model": settings.qwen_quality_model if settings.qwen_quality_enabled else None,
        "qwen_damage_enabled": settings.qwen_damage_enabled,
        "angle_classifier_enabled": angle_classifier is not None,
        "angle_classifier": angle_classifier.method if angle_classifier else None,
        "cleanliness_analysis_enabled": settings.cleanliness_analysis_enabled,
        "database_backend": settings.database_backend,
        "storage_backend": settings.storage_backend,
        "persistent_storage": (
            settings.database_backend == "postgresql"
            and settings.storage_backend == "cloud_storage_mount"
        ),
        "qwen_damage_few_shot_pairs": (
            len(qwen_damage_analyzer.references) if qwen_damage_analyzer else 0
        ),
    }
def require_local_labeling(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(403, "資料標註功能僅允許透過本機或 SSH 連接埠轉發使用")


@app.post("/api/labeling/sync")
def sync_labeling_dataset(request: Request, db: Session = Depends(get_db)):
    require_local_labeling(request)
    created = sync_manifest(db, settings.dataset_manifest_path)
    return {"created": created, "stats": labeling_stats(db)}


@app.get("/api/labeling/stats")
def get_labeling_stats(request: Request, db: Session = Depends(get_db)):
    require_local_labeling(request)
    sync_manifest(db, settings.dataset_manifest_path)
    return labeling_stats(db)


@app.get("/api/labeling/items")
def list_labeling_items(
    request: Request,
    review_status: str = "unreviewed",
    category: str = "",
    search: str = "",
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    require_local_labeling(request)
    sync_manifest(db, settings.dataset_manifest_path)
    if review_status not in {"", "unreviewed", "approved", "relabelled", "invalid", "excluded"}:
        raise HTTPException(422, "不支援的標註狀態")
    if category not in {"", "left_front", "right_front", "left_rear", "right_rear", "interior_front", "interior_rear", "other"}:
        raise HTTPException(422, "不支援的影像分類")
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)
    query = select(DatasetAnnotation)
    count_query = select(func.count()).select_from(DatasetAnnotation)
    if review_status:
        query = query.where(DatasetAnnotation.review_status == review_status)
        count_query = count_query.where(DatasetAnnotation.review_status == review_status)
    if category:
        effective_label = func.coalesce(DatasetAnnotation.actual_label, DatasetAnnotation.original_label)
        query = query.where(effective_label == category)
        count_query = count_query.where(effective_label == category)
    if search:
        query = query.where(DatasetAnnotation.source_path.ilike(f"%{search}%"))
        count_query = count_query.where(DatasetAnnotation.source_path.ilike(f"%{search}%"))
    total = db.scalar(count_query) or 0
    items = db.scalars(query.order_by(DatasetAnnotation.source_path).offset(offset).limit(limit)).all()
    return {"items": [annotation_dict(item) for item in items], "total": total, "offset": offset, "limit": limit}


@app.get("/api/labeling/items/{annotation_id}/image")
def get_labeling_image(annotation_id: str, request: Request, db: Session = Depends(get_db)):
    require_local_labeling(request)
    item = db.get(DatasetAnnotation, annotation_id)
    if not item:
        raise HTTPException(404, "找不到標註照片")
    root = settings.dataset_source_dir.resolve()
    image_path = (root / item.source_path).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(403, "不允許的照片路徑") from exc
    if not image_path.is_file():
        raise HTTPException(404, "原始照片不存在")
    return FileResponse(image_path)


@app.patch("/api/labeling/items/{annotation_id}")
def update_labeling_item(
    annotation_id: str,
    payload: DatasetAnnotationUpdate,
    request: Request,
    db: Session = Depends(get_db),
):
    require_local_labeling(request)
    item = db.get(DatasetAnnotation, annotation_id)
    if not item:
        raise HTTPException(404, "找不到標註資料")
    if payload.review_status in {"approved", "relabelled"} and (payload.actual_label is None or payload.acceptable is not True):
        raise HTTPException(422, "通過或改標的照片必須選擇實際分類並標記為可接受")
    if payload.review_status == "invalid" and (payload.acceptable is not False or payload.retake_reason is None):
        raise HTTPException(422, "無效照片必須標記不可接受並選擇退件原因")
    if payload.review_status == "excluded" and payload.acceptable is not False:
        raise HTTPException(422, "排除照片必須標記為不可接受")

    item.actual_label = payload.actual_label
    item.acceptable = payload.acceptable
    item.retake_reason = payload.retake_reason
    item.review_status = payload.review_status
    item.notes = payload.notes
    item.reviewer = payload.reviewer
    item.updated_at = utcnow()
    db.commit()
    db.refresh(item)
    return annotation_dict(item)


@app.get("/api/labeling/export.csv")
def export_labeling_annotations(request: Request, db: Session = Depends(get_db)):
    require_local_labeling(request)
    sync_manifest(db, settings.dataset_manifest_path)
    return Response(
        export_annotations_csv(db),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="carma-annotations.csv"'},
    )


def image_url(path: str | None) -> str | None:
    if not path:
        return None
    try:
        relative = Path(path).resolve().relative_to(settings.upload_dir.resolve())
        return f"/uploads/{relative.as_posix()}"
    except ValueError:
        return None


def quality_dict(result: QualityCheckResult, image: InspectionImage | None = None) -> dict:
    payload = {
        "passed": result.passed,
        "quality_score": result.quality_score,
        "blur_score": result.blur_score,
        "brightness_score": result.brightness_score,
        "vehicle_coverage": result.vehicle_coverage,
        "plate_detected": result.plate_detected,
        "warnings": result.warnings,
        "retake_instruction": result.retake_instruction,
        "model_confidence": result.model_confidence,
        "metrics": result.metrics,
    }
    if image:
        payload.update({"detected_view": image.detected_view, "expected_view": image.expected_view})
    return payload


def image_dict(image: InspectionImage) -> dict:
    return {
        "id": image.id,
        "inspection_id": image.inspection_id,
        "inspection_type": image.inspection_type,
        "expected_view": image.expected_view,
        "detected_view": image.detected_view,
        "image_url": image_url(image.image_path),
        "captured_at": image.captured_at,
        "quality_status": image.quality_status,
        "model_version": image.model_version,
        "quality": quality_dict(image.quality_result, image) if image.quality_result else None,
    }


def order_dict(order: RentalOrder, include_inspections: bool = False) -> dict:
    payload = {
        "id": order.id,
        "order_number": order.order_number,
        "plate_number": order.vehicle.plate_number,
        "vehicle_id": order.vehicle_id,
        "make_model": order.vehicle.make_model,
        "pickup_time": order.pickup_time,
        "return_time": order.return_time,
        "pickup_location": order.pickup_location,
        "return_location": order.return_location,
        "status": order.status,
        "created_at": order.created_at,
    }
    if include_inspections:
        payload["inspections"] = [
            {
                "id": session.id,
                "inspection_type": session.inspection_type,
                "location": session.location,
                "status": session.status,
                "started_at": session.started_at,
                "completed_at": session.completed_at,
                "images": [image_dict(image) for image in session.images],
            }
            for session in sorted(order.inspections, key=lambda value: value.started_at)
        ]
    return payload


@app.post("/api/orders", status_code=201)
def create_order(payload: OrderCreate, db: Session = Depends(get_db)):
    vehicle = db.scalar(select(Vehicle).where(Vehicle.plate_number == payload.plate_number))
    if not vehicle:
        vehicle = Vehicle(plate_number=payload.plate_number, make_model=payload.make_model)
        db.add(vehicle)
        db.flush()
    order = RentalOrder(
        order_number=payload.order_number,
        vehicle_id=vehicle.id,
        pickup_time=payload.pickup_time,
        return_time=payload.return_time,
        pickup_location=payload.pickup_location,
        return_location=payload.return_location,
    )
    db.add(order)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "訂單編號已存在")
    db.refresh(order)
    return order_dict(order)


@app.get("/api/orders")
def list_orders(
    plate: str | None = None,
    order_number: str | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
):
    query = select(RentalOrder).join(Vehicle).order_by(RentalOrder.created_at.desc())
    if plate:
        query = query.where(Vehicle.plate_number.ilike(f"%{plate}%"))
    if order_number:
        query = query.where(RentalOrder.order_number.ilike(f"%{order_number}%"))
    if status:
        query = query.where(RentalOrder.status == status)
    return [order_dict(item) for item in db.scalars(query).all()]


@app.get("/api/orders/{order_id}")
def get_order(order_id: str, db: Session = Depends(get_db)):
    order = db.get(RentalOrder, order_id)
    if not order:
        raise HTTPException(404, "找不到訂單")
    return order_dict(order, include_inspections=True)


def completed_pickup_baseline(
    db: Session, order_id: str
) -> tuple[InspectionSession | None, dict[str, InspectionImage], list[str]]:
    """Return the newest completed pickup session with all exterior baseline views."""
    pickups = db.scalars(
        select(InspectionSession)
        .where(
            InspectionSession.order_id == order_id,
            InspectionSession.inspection_type == "pickup",
            InspectionSession.status == "completed",
        )
        .order_by(InspectionSession.started_at.desc())
    ).all()
    for pickup in pickups:
        baseline = latest_passed_images(pickup)
        missing = [view for view in required_views[:4] if view not in baseline]
        if not missing:
            return pickup, baseline, []
    return None, {}, list(required_views[:4])


@app.post("/api/inspections", status_code=201)
def create_inspection(payload: InspectionCreate, db: Session = Depends(get_db)):
    order = db.get(RentalOrder, payload.order_id)
    if not order:
        raise HTTPException(404, "找不到訂單")
    if payload.inspection_type == "return":
        _, _, missing = completed_pickup_baseline(db, order.id)
        if missing:
            raise HTTPException(
                409,
                detail=error_payload(
                    "pickup_baseline_required",
                    "此訂單尚未完成取車基準照，無法開始還車拍攝",
                    {"missing_views": missing},
                ),
            )
    if payload.inspection_type == "pickup":
        active_damage_ids = set(db.scalars(
            select(KnownDamage.id).where(
                KnownDamage.vehicle_id == order.vehicle_id,
                KnownDamage.status == "active",
            )
        ).all())
        acknowledged_ids = set(db.scalars(
            select(KnownDamageAcknowledgement.known_damage_id).where(
                KnownDamageAcknowledgement.order_id == order.id
            )
        ).all())
        missing_acknowledgements = active_damage_ids - acknowledged_ids
        if missing_acknowledgements:
            raise HTTPException(
                409,
                detail=error_payload(
                    "known_damage_acknowledgement_required",
                    "請先確認此車既有車損，再開始取車拍攝",
                    {"count": len(missing_acknowledgements)},
                ),
            )

    session = InspectionSession(order_id=order.id, inspection_type=payload.inspection_type, location=payload.location)
    db.add(session)
    order.status = f"{payload.inspection_type}_capturing"
    db.commit()
    db.refresh(session)
    return {
        "id": session.id,
        "order_id": session.order_id,
        "inspection_type": session.inspection_type,
        "status": session.status,
        "required_views": required_views_for(session.inspection_type),
    }


def run_quality_check(image: InspectionImage, db: Session) -> QualityCheckResult:
    sibling_hashes = db.scalars(
        select(InspectionImage.perceptual_hash).where(
            InspectionImage.inspection_id == image.inspection_id,
            InspectionImage.id != image.id,
            InspectionImage.expected_view != image.expected_view,
            InspectionImage.quality_status == "passed",
            InspectionImage.perceptual_hash.is_not(None),
        )
    ).all()
    image_path = Path(image.image_path)
    result = analyze_capture_quality(
        image_path,
        image.expected_view,
        list(sibling_hashes),
        expected_plate=image.inspection.order.vehicle.plate_number,
    )
    image.model_version = settings.model_version
    image.detected_view = result["detected_view"]
    image.perceptual_hash = result["perceptual_hash"]
    image.quality_status = "passed" if result["passed"] else "failed"
    if image.quality_result:
        db.delete(image.quality_result)
        db.flush()
    record = QualityCheckResult(
        image_id=image.id,
        passed=result["passed"],
        quality_score=result["quality_score"],
        blur_score=result["blur_score"],
        brightness_score=result["brightness_score"],
        vehicle_coverage=result["vehicle_coverage"],
        plate_detected=result["plate_detected"],
        warnings=result["warnings"],
        retake_instruction=result["retake_instruction"],
        model_confidence=result["model_confidence"],
        metrics=result["metrics"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def analyze_capture_quality(
    image_path: Path,
    expected_view: str,
    sibling_hashes: list[str] | None = None,
    expected_plate: str | None = None,
) -> dict:
    """Run the same capture stack with or without an order-backed image record."""

    result = quality_analyzer.analyze(image_path, expected_view, sibling_hashes or [])
    if qwen_quality_analyzer is not None:
        qwen_result = qwen_quality_analyzer.analyze(
            image_path, expected_view, expected_plate=expected_plate
        )
        if angle_classifier is not None:
            angle_result = angle_classifier.predict(image_path)
            qwen_result = apply_trained_angle(
                qwen_result, angle_result, expected_view,
                settings.angle_classifier_retake_confidence,
            )
        result = merge_capture_quality(result, qwen_result)
    elif angle_classifier is not None:
        angle_result = angle_classifier.predict(image_path)
        result = apply_angle_to_quality(result, angle_result, expected_view)
    detected_view = result.get("detected_view")
    if detected_view in required_views and detected_view != expected_view:
        instruction = result.get("retake_instruction") or (
            "拍攝位置與官方要求不符，請依站位示意重新拍攝；位置必須完全相同。"
        )
        result["passed"] = False
        result["review_required"] = False
        result["retake_instruction"] = instruction
        result["warnings"] = [instruction]
        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            metrics["capture_decision"] = "retake"
            metrics["review_required"] = False
    return result


@app.post("/api/inspections/{inspection_id}/images", status_code=201)
async def upload_image(
    inspection_id: str,
    expected_view: ViewName = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    session = db.get(InspectionSession, inspection_id)
    if not session:
        raise HTTPException(404, "找不到檢查工作階段")
    content_type = file.content_type or ""
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "僅支援 JPG、PNG 或 WebP 圖片")
    suffix = { "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp" }[content_type]
    target_dir = settings.upload_dir / session.order_id / session.inspection_type
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{expected_view}-{uuid.uuid4().hex}{suffix}"
    total = 0
    with target.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > settings.max_upload_mb * 1024 * 1024:
                output.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, f"圖片不可超過 {settings.max_upload_mb} MB")
            output.write(chunk)
    image = InspectionImage(
        inspection_id=session.id,
        order_id=session.order_id,
        vehicle_id=session.order.vehicle_id,
        inspection_type=session.inspection_type,
        expected_view=expected_view,
        image_path=str(target),
        original_filename=file.filename,
        model_version=settings.model_version,
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    try:
        quality = run_quality_check(image, db)
    except (ValueError, RuntimeError) as exc:
        db.delete(image)
        db.commit()
        target.unlink(missing_ok=True)
        status_code = 422 if isinstance(exc, ValueError) else 503
        raise HTTPException(status_code, str(exc)) from exc
    return {
        "image": image_dict(image),
        "quality": quality_dict(quality, image),
        "demo_mode": not settings.qwen_quality_enabled,
        "quality_engine": "qwen3_vl_plus_cv" if settings.qwen_quality_enabled else "demo_rules",
    }


@app.post("/api/images/{image_id}/quality-check")
def quality_check(image_id: str, db: Session = Depends(get_db)):
    image = db.get(InspectionImage, image_id)
    if not image:
        raise HTTPException(404, "找不到圖片")
    result = run_quality_check(image, db)
    return quality_dict(result, image) | {"image_id": image.id, "demo_mode": True}


@app.post("/api/model-test/quality")
async def model_test_quality(
    request: Request,
    expected_view: ViewName = Form(...),
    expected_plate: str | None = Form(None),
    file: UploadFile = File(...),
):
    """Local-only, order-free quality test. The uploaded image is always temporary."""

    require_local_labeling(request)
    content_type = file.content_type or ""
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "僅支援 JPG、PNG 或 WebP 圖片")
    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[content_type]
    target_dir = settings.analysis_dir / "model-tests"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4().hex}{suffix}"
    total = 0
    try:
        with target.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"圖片不可超過 {settings.max_upload_mb} MB")
                output.write(chunk)
        result = analyze_capture_quality(
            target, expected_view, expected_plate=expected_plate
        )
        return {
            "expected_view": expected_view,
            "detected_view": result["detected_view"],
            "passed": result["passed"],
            "quality_score": result["quality_score"],
            "model_confidence": result["model_confidence"],
            "retake_instruction": result["retake_instruction"],
            "warnings": result["warnings"],
            "metrics": result["metrics"],
            "model_version": settings.model_version,
            "temporary_file_deleted": True,
        }
    finally:
        target.unlink(missing_ok=True)


def latest_passed_images(session: InspectionSession) -> dict[str, InspectionImage]:
    selected: dict[str, InspectionImage] = {}
    for image in sorted(session.images, key=lambda value: value.captured_at):
        if image.quality_status == "passed":
            selected[image.expected_view] = image
    return selected


@app.post("/api/inspections/{inspection_id}/analyze")
def analyze_inspection(inspection_id: str, db: Session = Depends(get_db)):
    started = time.perf_counter()
    session = db.get(InspectionSession, inspection_id)
    if not session:
        raise HTTPException(404, "找不到檢查工作階段")
    current = latest_passed_images(session)
    missing = [view for view in required_views_for(session.inspection_type) if view not in current]
    if missing:
        raise HTTPException(
            409,
            detail=error_payload("incomplete_inspection", "缺少合格的必要拍攝位置，無法完成分析", {"missing_views": missing}),
        )

    old_damage = db.scalars(select(DamageComparison).where(DamageComparison.order_id == session.order_id)).all()
    old_clean = db.scalars(select(CleanlinessResult).where(CleanlinessResult.order_id == session.order_id)).all()
    for record in [*old_damage, *old_clean]:
        db.delete(record)
    db.flush()

    damage_results = []
    requires_review = False
    max_damage = 0.0
    if session.inspection_type == "return":
        _, baseline, missing_baseline = completed_pickup_baseline(db, session.order_id)
        if missing_baseline:
            raise HTTPException(
                409,
                detail=error_payload(
                    "pickup_baseline_required",
                    "此訂單缺少完整取車基準照，無法進行還車車損比對",
                    {"missing_views": missing_baseline},
                ),
            )
        for view in required_views[:4]:
            output = settings.analysis_dir / f"{session.order_id}-{view}-{uuid.uuid4().hex[:8]}.jpg"
            data = damage_detector.compare(Path(baseline[view].image_path), Path(current[view].image_path), output)
            record = DamageComparison(
                order_id=session.order_id,
                baseline_image_id=baseline[view].id,
                current_image_id=current[view].id,
                view=view,
                difference_score=data["difference_score"],
                ssim_score=data["ssim_score"],
                suspected_new_damage=data["suspected_new_damage"],
                severity=data["severity"],
                confidence=data["confidence"],
                requires_manual_review=data["requires_manual_review"],
                issue_regions=data["issue_regions"],
                heatmap_path=data["heatmap_path"],
                explanation=data["explanation"],
                model_version=data.get("inference_mode", "demo-change-v1"),
            )
            db.add(record)
            damage_results.append(record)
            max_damage = max(max_damage, data["difference_score"])
            requires_review = requires_review or data["requires_manual_review"]
    else:
        requires_review = True

    cleanliness_results = []
    for view in required_views[4:] if settings.cleanliness_analysis_enabled and session.inspection_type == "return" else []:
        data = cleanliness_classifier.classify(Path(current[view].image_path))
        record = CleanlinessResult(
            order_id=session.order_id,
            image_id=current[view].id,
            cleanliness_level=data["cleanliness_level"],
            cleanliness_score=data["cleanliness_score"],
            detected_issues=data["detected_issues"],
            issue_regions=data["issue_regions"],
            immediate_cleaning_required=data["immediate_cleaning_required"],
            confidence=data["confidence"],
            explanation=data["explanation"],
        )
        db.add(record)
        cleanliness_results.append(record)
    db.flush()

    qualities = [image.quality_result for image in current.values()]
    average_quality = sum(value.quality_score for value in qualities) / len(qualities)
    average_confidence = sum(value.model_confidence for value in qualities) / len(qualities)
    cleanliness_score = (
        min(value.cleanliness_score for value in cleanliness_results)
        if cleanliness_results else 100.0
    )
    risk_data = assess_risk(max_damage, cleanliness_score, average_quality, average_confidence, 0, requires_review)
    suspected_damage = [item for item in damage_results if item.suspected_new_damage]
    if suspected_damage:
        if risk_data["risk_level"] == "green":
            risk_data["risk_level"] = "yellow"
            risk_data["risk_score"] = max(
                risk_data["risk_score"], get_risk_config()["thresholds"]["green_max"] + 1
            )
        risk_data["recommended_action"] = "疑似新車損，送營運人員比對取還車照片"
    if suspected_damage and not any("疑似新車損" in reason for reason in risk_data["reasons"]):
        view_labels = {
            "left_front": "左前", "right_front": "右前",
            "left_rear": "左後", "right_rear": "右後",
        }
        views = "、".join(view_labels.get(item.view, item.view) for item in suspected_damage)
        risk_data["reasons"].insert(0, f"疑似新車損：{views}")
    previous = db.scalar(select(RiskAssessment).where(RiskAssessment.inspection_id == session.id))
    if previous:
        previous_alerts = db.scalars(
            select(Alert).where(Alert.risk_assessment_id == previous.id)
        ).all()
        for alert in previous_alerts:
            db.delete(alert)
        db.delete(previous)
        db.flush()
    assessment = RiskAssessment(order_id=session.order_id, inspection_id=session.id, **risk_data)
    db.add(assessment)
    db.flush()
    if risk_data["risk_level"] != "green":
        db.add(Alert(
            order_id=session.order_id,
            risk_assessment_id=assessment.id,
            alert_type="new_damage" if suspected_damage else "inspection_risk",
            severity=risk_data["risk_level"],
            message="；".join(risk_data["reasons"]),
        ))
    if suspected_damage:
        existing_task = db.scalar(
            select(OperationTask).where(
                OperationTask.order_id == session.order_id,
                OperationTask.task_type == "claim_review",
                OperationTask.status.in_(["pending", "assigned", "reviewing"]),
            )
        )
        if existing_task is None:
            db.add(OperationTask(
                order_id=session.order_id,
                task_type="claim_review",
                status="pending",
                priority="urgent",
                notes="Qwen 配對比對發現疑似新車損，請人工核對取車與還車照片。",
            ))
    session.status = "completed"
    session.completed_at = utcnow()
    session.order.status = "completed" if risk_data["risk_level"] == "green" else "attention_required"
    db.commit()
    return {
        "inspection_id": session.id,
        "risk": risk_data,
        "damage_comparisons": [
            {
                "view": item.view,
                "difference_score": item.difference_score,
                "suspected_new_damage": item.suspected_new_damage,
                "requires_manual_review": item.requires_manual_review,
                "severity": item.severity,
                "confidence": item.confidence,
                "model_version": item.model_version,
                "heatmap_url": image_url(item.heatmap_path),
                "explanation": item.explanation,
            }
            for item in damage_results
        ],
        "cleanliness": [
            {
                "view": current_view,
                "cleanliness_level": item.cleanliness_level,
                "cleanliness_score": item.cleanliness_score,
                "detected_issues": item.detected_issues,
                "immediate_cleaning_required": item.immediate_cleaning_required,
                "confidence": item.confidence,
                "explanation": item.explanation,
            }
            for current_view, item in zip(required_views[4:], cleanliness_results)
        ],
        "processing_ms": round((time.perf_counter() - started) * 1000),
        "demo_mode": True,
    }


def case_dict(assessment: RiskAssessment, db: Session, details: bool = False) -> dict:
    order = db.get(RentalOrder, assessment.order_id)
    tasks = db.scalars(select(OperationTask).where(OperationTask.order_id == order.id)).all()
    damage = db.scalars(select(DamageComparison).where(DamageComparison.order_id == order.id)).all()
    clean = db.scalars(select(CleanlinessResult).where(CleanlinessResult.order_id == order.id)).all()
    alerts = db.scalars(
        select(Alert).where(Alert.order_id == order.id).order_by(Alert.created_at.desc())
    ).all()
    payload = {
        "id": assessment.id,
        "order_id": order.id,
        "order_number": order.order_number,
        "plate_number": order.vehicle.plate_number,
        "vehicle_id": order.vehicle_id,
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "reasons": assessment.reasons,
        "recommended_action": assessment.recommended_action,
        "status": order.status,
        "has_damage": any(item.suspected_new_damage for item in damage),
        "has_cleanliness_issue": any(item.cleanliness_level == "dirty" for item in clean),
        "created_at": assessment.created_at,
        "tasks": [{"id": task.id, "task_type": task.task_type, "status": task.status, "priority": task.priority, "assignee": task.assignee, "notes": task.notes} for task in tasks],
        "alerts": [{
            "id": alert.id, "alert_type": alert.alert_type,
            "severity": alert.severity, "message": alert.message,
            "acknowledged": alert.acknowledged, "created_at": alert.created_at,
        } for alert in alerts],
    }
    if details:
        payload["order"] = order_dict(order, include_inspections=True)
        payload["components"] = assessment.components
        payload["damage_comparisons"] = [
            {
                "id": item.id,
                "view": item.view,
                "difference_score": item.difference_score,
                "ssim_score": item.ssim_score,
                "suspected_new_damage": item.suspected_new_damage,
                "severity": item.severity,
                "confidence": item.confidence,
                "requires_manual_review": item.requires_manual_review,
                "issue_regions": item.issue_regions,
                "model_version": item.model_version,
                "heatmap_url": image_url(item.heatmap_path),
                "explanation": item.explanation,
            }
            for item in damage
        ]
        payload["cleanliness"] = [
            {
                "cleanliness_level": item.cleanliness_level,
                "cleanliness_score": item.cleanliness_score,
                "detected_issues": item.detected_issues,
                "immediate_cleaning_required": item.immediate_cleaning_required,
                "confidence": item.confidence,
                "explanation": item.explanation,
            }
            for item in clean
        ]
        reviews = db.scalars(select(HumanReview).where(HumanReview.order_id == order.id)).all()
        payload["reviews"] = [{"id": item.id, "reviewer": item.reviewer, "decision": item.decision, "notes": item.notes, "labels": item.labels, "created_at": item.created_at} for item in reviews]
        evidence = db.scalars(select(DamageEvidence).where(DamageEvidence.order_id == order.id).order_by(DamageEvidence.captured_at)).all()
        payload["damage_evidence"] = [{"id": item.id, "purpose": item.purpose, "related_view": item.related_view, "notes": item.notes, "image_url": image_url(item.image_path), "captured_at": item.captured_at} for item in evidence]
        known = db.scalars(select(KnownDamage).where(KnownDamage.vehicle_id == order.vehicle_id, KnownDamage.status == "active").order_by(KnownDamage.confirmed_at.desc())).all()
        payload["known_damages"] = [known_damage_dict(item) for item in known]
    return payload


@app.get("/api/cases")
def list_cases(
    risk: str | None = None,
    plate: str | None = None,
    order_number: str | None = None,
    anomaly_type: str | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
):
    query = select(RiskAssessment).join(RentalOrder).join(Vehicle).order_by(RiskAssessment.created_at.desc())
    if risk:
        query = query.where(RiskAssessment.risk_level == risk)
    if plate:
        query = query.where(Vehicle.plate_number.ilike(f"%{plate}%"))
    if order_number:
        query = query.where(RentalOrder.order_number.ilike(f"%{order_number}%"))
    if status:
        query = query.where(RentalOrder.status == status)
    cases = [case_dict(item, db) for item in db.scalars(query).all()]
    if anomaly_type == "damage":
        cases = [item for item in cases if item["has_damage"]]
    elif anomaly_type == "cleanliness":
        cases = [item for item in cases if item["has_cleanliness_issue"]]
    return cases


@app.get("/api/cases/{case_id}")
def get_case(case_id: str, db: Session = Depends(get_db)):
    assessment = db.get(RiskAssessment, case_id)
    if not assessment:
        raise HTTPException(404, "找不到案件")
    return case_dict(assessment, db, details=True)


@app.post("/api/cases/{case_id}/review", status_code=201)
def create_review(case_id: str, payload: ReviewCreate, db: Session = Depends(get_db)):
    assessment = db.get(RiskAssessment, case_id)
    if not assessment:
        raise HTTPException(404, "找不到案件")
    review = HumanReview(order_id=assessment.order_id, **payload.model_dump())
    db.add(review)
    promoted = 0
    if payload.decision == "confirmed_damage":
        assessment_order = db.get(RentalOrder, assessment.order_id)
        suspected = db.scalars(
            select(DamageComparison).where(
                DamageComparison.order_id == assessment.order_id,
                DamageComparison.suspected_new_damage.is_(True),
            )
        ).all()
        for comparison in suspected:
            existing = db.scalar(
                select(KnownDamage).where(KnownDamage.source_comparison_id == comparison.id)
            )
            if existing is None:
                db.add(KnownDamage(
                    vehicle_id=assessment_order.vehicle_id,
                    source_order_id=assessment.order_id,
                    source_comparison_id=comparison.id,
                    view=comparison.view,
                    description=comparison.explanation,
                    severity=comparison.severity,
                    issue_regions=comparison.issue_regions,
                    status="active",
                ))
                promoted += 1
    if payload.decision in {"approved", "no_damage", "dismissed"}:
        assessment_order = db.get(RentalOrder, assessment.order_id)
        assessment_order.status = "completed"
    db.commit()
    db.refresh(review)
    return {"id": review.id, "order_id": review.order_id, **payload.model_dump(), "created_at": review.created_at, "known_damage_records_created": promoted}


@app.post("/api/cases/{case_id}/tasks", status_code=201)
def create_task(case_id: str, payload: TaskCreate, db: Session = Depends(get_db)):
    assessment = db.get(RiskAssessment, case_id)
    if not assessment:
        raise HTTPException(404, "找不到案件")
    task = OperationTask(order_id=assessment.order_id, **payload.model_dump())
    db.add(task)
    db.commit()
    db.refresh(task)
    return {"id": task.id, "order_id": task.order_id, "status": task.status, **payload.model_dump()}


@app.patch("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(404, "找不到預警")
    alert.acknowledged = True
    db.commit()
    return {"id": alert.id, "acknowledged": alert.acknowledged}


@app.patch("/api/tasks/{task_id}")
def update_task(task_id: str, payload: TaskUpdate, db: Session = Depends(get_db)):
    task = db.get(OperationTask, task_id)
    if not task:
        raise HTTPException(404, "找不到任務")
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(422, "至少提供一個要更新的欄位")
    for key, value in updates.items():
        setattr(task, key, value)
    task.updated_at = utcnow()
    db.commit()
    return {"id": task.id, "order_id": task.order_id, "task_type": task.task_type, "status": task.status, "priority": task.priority, "assignee": task.assignee, "notes": task.notes, "updated_at": task.updated_at}


@app.get("/api/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    today = datetime.now(timezone.utc).date().isoformat()
    assessments = db.scalars(select(RiskAssessment).where(func.date(RiskAssessment.created_at) == today)).all()
    damage_count = db.scalar(select(func.count(func.distinct(DamageComparison.order_id))).where(DamageComparison.suspected_new_damage.is_(True), func.date(DamageComparison.created_at) == today)) or 0
    dirty_count = db.scalar(select(func.count(func.distinct(CleanlinessResult.order_id))).where(CleanlinessResult.cleanliness_level == "dirty", func.date(CleanlinessResult.created_at) == today)) or 0
    completed = db.scalars(select(InspectionSession).where(func.date(InspectionSession.completed_at) == today)).all()
    durations = [(item.completed_at - item.started_at).total_seconds() for item in completed]
    return {
        "today_inspections": len(assessments),
        "normal_count": sum(item.risk_level == "green" for item in assessments),
        "review_count": sum(item.risk_level == "yellow" for item in assessments),
        "high_risk_count": sum(item.risk_level == "red" for item in assessments),
        "new_damage_count": damage_count,
        "dirty_count": dirty_count,
        "average_processing_seconds": round(sum(durations) / len(durations), 1) if durations else 0,
        "demo_mode": True,
    }


def known_damage_dict(item: KnownDamage, acknowledged: bool = False) -> dict:
    return {
        "id": item.id,
        "vehicle_id": item.vehicle_id,
        "source_order_id": item.source_order_id,
        "view": item.view,
        "description": item.description,
        "severity": item.severity,
        "issue_regions": item.issue_regions,
        "status": item.status,
        "first_seen_at": item.first_seen_at,
        "confirmed_at": item.confirmed_at,
        "resolved_at": item.resolved_at,
        "acknowledged": acknowledged,
    }


@app.get("/api/orders/{order_id}/known-damages")
def get_order_known_damages(order_id: str, db: Session = Depends(get_db)):
    order = db.get(RentalOrder, order_id)
    if not order:
        raise HTTPException(404, "找不到訂單")
    items = db.scalars(
        select(KnownDamage)
        .where(KnownDamage.vehicle_id == order.vehicle_id, KnownDamage.status == "active")
        .order_by(KnownDamage.confirmed_at.desc())
    ).all()
    acknowledged_ids = set(db.scalars(
        select(KnownDamageAcknowledgement.known_damage_id)
        .where(KnownDamageAcknowledgement.order_id == order.id)
    ).all())
    return {
        "order_id": order.id,
        "plate_number": order.vehicle.plate_number,
        "items": [known_damage_dict(item, item.id in acknowledged_ids) for item in items],
        "all_acknowledged": all(item.id in acknowledged_ids for item in items),
    }


@app.post("/api/orders/{order_id}/known-damages/acknowledge")
def acknowledge_known_damages(order_id: str, db: Session = Depends(get_db)):
    order = db.get(RentalOrder, order_id)
    if not order:
        raise HTTPException(404, "找不到訂單")
    items = db.scalars(
        select(KnownDamage).where(
            KnownDamage.vehicle_id == order.vehicle_id,
            KnownDamage.status == "active",
        )
    ).all()
    acknowledged_ids = set(db.scalars(
        select(KnownDamageAcknowledgement.known_damage_id)
        .where(KnownDamageAcknowledgement.order_id == order.id)
    ).all())
    for item in items:
        if item.id not in acknowledged_ids:
            db.add(KnownDamageAcknowledgement(
                order_id=order.id,
                known_damage_id=item.id,
                acknowledged_by="customer",
            ))
    db.commit()
    return {"order_id": order.id, "acknowledged_count": len(items), "all_acknowledged": True}


@app.post("/api/inspections/{inspection_id}/evidence", status_code=201)
async def upload_damage_evidence(
    inspection_id: str,
    purpose: EvidencePurpose = Form(...),
    related_view: ViewName | None = Form(None),
    notes: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    session = db.get(InspectionSession, inspection_id)
    if not session:
        raise HTTPException(404, "找不到檢查工作階段")
    content_type = file.content_type or ""
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "僅支援 JPG、PNG 或 WebP 圖片")
    if notes and len(notes) > 500:
        raise HTTPException(422, "補充說明不可超過 500 字")
    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[content_type]
    target_dir = settings.upload_dir / session.order_id / session.inspection_type / "evidence"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{purpose}-{uuid.uuid4().hex}{suffix}"
    total = 0
    with target.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > settings.max_upload_mb * 1024 * 1024:
                output.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, f"圖片不可超過 {settings.max_upload_mb} MB")
            output.write(chunk)
    evidence = DamageEvidence(
        inspection_id=session.id,
        order_id=session.order_id,
        vehicle_id=session.order.vehicle_id,
        purpose=purpose,
        related_view=related_view,
        notes=notes,
        image_path=str(target),
        original_filename=file.filename,
    )
    db.add(evidence)
    db.commit()
    db.refresh(evidence)
    return {
        "id": evidence.id,
        "purpose": evidence.purpose,
        "related_view": evidence.related_view,
        "notes": evidence.notes,
        "image_url": image_url(evidence.image_path),
        "captured_at": evidence.captured_at,
        "required_view": False,
    }


class SPAStaticFiles(StaticFiles):
    """Serve the compiled React app and fall back to index.html for client routes."""

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as error:
            if error.status_code != 404:
                raise
            return await super().get_response("index.html", scope)
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        return response


if settings.frontend_dir and settings.frontend_dir.is_dir():
    app.mount(
        "/",
        SPAStaticFiles(directory=settings.frontend_dir, html=True),
        name="frontend",
    )
