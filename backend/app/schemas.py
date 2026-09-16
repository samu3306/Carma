from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ViewName = Literal["left_front", "right_front", "left_rear", "right_rear", "interior_front", "interior_rear"]
InspectionType = Literal["pickup", "return"]
EvidencePurpose = Literal["damage_context", "damage_closeup", "supplementary"]
TaskStatus = Literal["pending", "reviewing", "assigned", "processing", "completed", "dismissed"]


class OrderCreate(BaseModel):
    order_number: str = Field(min_length=3, max_length=40)
    plate_number: str = Field(min_length=4, max_length=20)
    make_model: str | None = Field(default=None, max_length=80)
    pickup_location: str | None = Field(default=None, max_length=120)
    return_location: str | None = Field(default=None, max_length=120)
    pickup_time: datetime | None = None
    return_time: datetime | None = None

    @field_validator("plate_number")
    @classmethod
    def normalize_plate(cls, value: str) -> str:
        return value.strip().upper()


class InspectionCreate(BaseModel):
    order_id: str
    inspection_type: InspectionType
    location: str | None = Field(default=None, max_length=120)


class ReviewCreate(BaseModel):
    reviewer: str = Field(min_length=2, max_length=80)
    decision: Literal["confirmed_damage", "no_damage", "cleaning_required", "reshoot_required", "approved", "dismissed"]
    notes: str | None = Field(default=None, max_length=2000)
    labels: dict = Field(default_factory=dict)


class TaskCreate(BaseModel):
    task_type: Literal["cleaning", "repair", "claim_review", "manual_review", "reshoot"]
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    assignee: str | None = Field(default=None, max_length=80)
    notes: str | None = Field(default=None, max_length=2000)


class TaskUpdate(BaseModel):
    status: TaskStatus | None = None
    priority: Literal["low", "normal", "high", "urgent"] | None = None
    assignee: str | None = Field(default=None, max_length=80)
    notes: str | None = Field(default=None, max_length=2000)

    model_config = ConfigDict(extra="forbid")
DatasetLabel = Literal["left_front", "right_front", "left_rear", "right_rear", "interior_front", "interior_rear", "other"]
AnnotationStatus = Literal["unreviewed", "approved", "relabelled", "invalid", "excluded"]
RetakeReason = Literal["wrong_scene", "wrong_angle", "partial", "blur", "too_dark", "overexposed", "duplicate", "non_vehicle", "corrupt", "other"]


class DatasetAnnotationUpdate(BaseModel):
    actual_label: DatasetLabel | None = None
    acceptable: bool | None = None
    retake_reason: RetakeReason | None = None
    review_status: AnnotationStatus
    notes: str | None = Field(default=None, max_length=1000)
    reviewer: str = Field(default="本機標註員", min_length=1, max_length=80)

    model_config = ConfigDict(extra="forbid")
