from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Vehicle(Base):
    __tablename__ = "vehicles"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    plate_number: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    make_model: Mapped[str | None] = mapped_column(String(80))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RentalOrder(Base):
    __tablename__ = "rental_orders"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_number: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    vehicle_id: Mapped[str] = mapped_column(ForeignKey("vehicles.id"), index=True)
    pickup_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    return_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pickup_location: Mapped[str | None] = mapped_column(String(120))
    return_location: Mapped[str | None] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(30), default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    vehicle: Mapped[Vehicle] = relationship()
    inspections: Mapped[list["InspectionSession"]] = relationship(back_populates="order", cascade="all, delete-orphan")


class InspectionSession(Base):
    __tablename__ = "inspection_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    inspection_type: Mapped[str] = mapped_column(String(10), index=True)
    location: Mapped[str | None] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(30), default="capturing")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    order: Mapped[RentalOrder] = relationship(back_populates="inspections")
    images: Mapped[list["InspectionImage"]] = relationship(back_populates="inspection", cascade="all, delete-orphan")


class InspectionImage(Base):
    __tablename__ = "inspection_images"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspection_sessions.id"), index=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    vehicle_id: Mapped[str] = mapped_column(ForeignKey("vehicles.id"), index=True)
    inspection_type: Mapped[str] = mapped_column(String(10))
    expected_view: Mapped[str] = mapped_column(String(30), index=True)
    detected_view: Mapped[str | None] = mapped_column(String(30))
    image_path: Mapped[str] = mapped_column(Text)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    quality_status: Mapped[str] = mapped_column(String(20), default="pending")
    model_version: Mapped[str] = mapped_column(String(60), default="demo-rules-v1")
    perceptual_hash: Mapped[str | None] = mapped_column(String(64))
    inspection: Mapped[InspectionSession] = relationship(back_populates="images")
    quality_result: Mapped["QualityCheckResult | None"] = relationship(back_populates="image", uselist=False, cascade="all, delete-orphan")


class QualityCheckResult(Base):
    __tablename__ = "quality_check_results"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    image_id: Mapped[str] = mapped_column(ForeignKey("inspection_images.id"), unique=True)
    passed: Mapped[bool] = mapped_column(Boolean)
    quality_score: Mapped[float] = mapped_column(Float)
    blur_score: Mapped[float] = mapped_column(Float)
    brightness_score: Mapped[float] = mapped_column(Float)
    vehicle_coverage: Mapped[float] = mapped_column(Float)
    plate_detected: Mapped[bool] = mapped_column(Boolean)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    retake_instruction: Mapped[str | None] = mapped_column(Text)
    model_confidence: Mapped[float] = mapped_column(Float)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    image: Mapped[InspectionImage] = relationship(back_populates="quality_result")


class DamageComparison(Base):
    __tablename__ = "damage_comparisons"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    baseline_image_id: Mapped[str] = mapped_column(ForeignKey("inspection_images.id"))
    current_image_id: Mapped[str] = mapped_column(ForeignKey("inspection_images.id"))
    view: Mapped[str] = mapped_column(String(30))
    comparison_basis: Mapped[str] = mapped_column(String(40), default="pickup_vs_return")
    difference_score: Mapped[float] = mapped_column(Float)
    ssim_score: Mapped[float] = mapped_column(Float)
    suspected_new_damage: Mapped[bool] = mapped_column(Boolean)
    severity: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    requires_manual_review: Mapped[bool] = mapped_column(Boolean, default=False)
    issue_regions: Mapped[list] = mapped_column(JSON, default=list)
    heatmap_path: Mapped[str | None] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(60), default="demo-change-v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CleanlinessResult(Base):
    __tablename__ = "cleanliness_results"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("inspection_images.id"))
    cleanliness_level: Mapped[str] = mapped_column(String(20))
    cleanliness_score: Mapped[float] = mapped_column(Float)
    detected_issues: Mapped[list] = mapped_column(JSON, default=list)
    issue_regions: Mapped[list] = mapped_column(JSON, default=list)
    immediate_cleaning_required: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(60), default="demo-cleanliness-v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspection_sessions.id"), unique=True)
    risk_score: Mapped[int] = mapped_column(Integer)
    risk_level: Mapped[str] = mapped_column(String(20), index=True)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    recommended_action: Mapped[str] = mapped_column(Text)
    components: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    risk_assessment_id: Mapped[str] = mapped_column(ForeignKey("risk_assessments.id"))
    alert_type: Mapped[str] = mapped_column(String(40))
    severity: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OperationTask(Base):
    __tablename__ = "operation_tasks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    task_type: Mapped[str] = mapped_column(String(30), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    priority: Mapped[str] = mapped_column(String(20), default="normal")
    assignee: Mapped[str | None] = mapped_column(String(80))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class HumanReview(Base):
    __tablename__ = "human_reviews"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(80))
    decision: Mapped[str] = mapped_column(String(30))
    notes: Mapped[str | None] = mapped_column(Text)
    labels: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
class DatasetAnnotation(Base):
    __tablename__ = "dataset_annotations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_path: Mapped[str] = mapped_column(Text, unique=True, index=True)
    image_type: Mapped[int | None] = mapped_column(Integer)
    original_label: Mapped[str] = mapped_column(String(30), index=True)
    actual_label: Mapped[str | None] = mapped_column(String(30), index=True)
    acceptable: Mapped[bool | None] = mapped_column(Boolean)
    retake_reason: Mapped[str | None] = mapped_column(String(40), index=True)
    review_status: Mapped[str] = mapped_column(String(20), default="unreviewed", index=True)
    notes: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class DamageEvidence(Base):
    """Optional user/AI-requested evidence outside the required capture views."""
    __tablename__ = "damage_evidence"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspection_sessions.id"), index=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    vehicle_id: Mapped[str] = mapped_column(ForeignKey("vehicles.id"), index=True)
    purpose: Mapped[str] = mapped_column(String(30), index=True)
    related_view: Mapped[str | None] = mapped_column(String(30), index=True)
    notes: Mapped[str | None] = mapped_column(Text)
    image_path: Mapped[str] = mapped_column(Text)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    reported_by: Mapped[str] = mapped_column(String(20), default="customer")
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KnownDamage(Base):
    """Confirmed vehicle damage ledger. CSV files are exports, never the source of truth."""
    __tablename__ = "known_damages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    vehicle_id: Mapped[str] = mapped_column(ForeignKey("vehicles.id"), index=True)
    source_order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    source_comparison_id: Mapped[str | None] = mapped_column(ForeignKey("damage_comparisons.id"), unique=True)
    view: Mapped[str] = mapped_column(String(30), index=True)
    description: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(20))
    issue_regions: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnownDamageAcknowledgement(Base):
    """Proof that the next renter was shown pre-existing damage before pickup."""
    __tablename__ = "known_damage_acknowledgements"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    order_id: Mapped[str] = mapped_column(ForeignKey("rental_orders.id"), index=True)
    known_damage_id: Mapped[str] = mapped_column(ForeignKey("known_damages.id"), index=True)
    acknowledged_by: Mapped[str] = mapped_column(String(40), default="customer")
    acknowledged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
