from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "iRent 智能車況管家"
    database_url: str = f"sqlite:///{ROOT_DIR / 'carma.db'}"
    upload_dir: Path = ROOT_DIR / "uploads"
    analysis_dir: Path = ROOT_DIR / "uploads" / "analysis"
    dataset_source_dir: Path = ROOT_DIR / "downloaded_images"
    dataset_manifest_path: Path = ROOT_DIR / "datasets" / "by_image_type_flat" / "manifest.csv"
    demo_mode: bool = True
    model_version: str = "demo-rules-v1"
    qwen_quality_enabled: bool = False
    qwen_quality_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_damage_enabled: bool = False
    cleanliness_analysis_enabled: bool = True
    qwen_damage_few_shot_manifest: Path = ROOT_DIR / "data" / "damage" / "few_shot_pairs.csv"
    angle_classifier_enabled: bool = True
    angle_classifier_backend: Literal["dinov2_onnx", "dinov2", "mobilenetv2"] = "dinov2_onnx"
    angle_classifier_dir: Path = ROOT_DIR / "models" / "angle_classifier"
    angle_feature_extractor_path: Path = ROOT_DIR / "models" / "mobilenetv2-12.onnx"
    dinov2_angle_classifier_dir: Path = ROOT_DIR / "models" / "dinov2_angle_classifier"
    dinov2_onnx_path: Path = ROOT_DIR / "models" / "dinov2_angle_classifier" / "dinov2_vits14_matmul_int8.onnx"
    dinov2_model_name: str = "facebook/dinov2-small"
    dinov2_cache_dir: Path = ROOT_DIR / "models" / "huggingface"
    dinov2_device: Literal["auto", "cuda", "cpu"] = "auto"
    angle_classifier_retake_confidence: float = 0.72
    max_upload_mb: int = 15
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    vehicle_detector_enabled: bool = True
    yolo_config_path: Path = ROOT_DIR / "models" / "yolov4-tiny.cfg"
    yolo_weights_path: Path = ROOT_DIR / "models" / "yolov4-tiny.weights"
    frontend_dir: Path | None = None
    storage_backend: Literal["local", "cloud_storage_mount"] = "local"

    model_config = SettingsConfigDict(env_prefix="CARMA_", env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def database_backend(self) -> str:
        if self.database_url.startswith(("postgresql", "postgres")):
            return "postgresql"
        if self.database_url.startswith("sqlite"):
            return "sqlite"
        return "other"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.analysis_dir.mkdir(parents=True, exist_ok=True)
    settings.dataset_source_dir.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache
def get_risk_config() -> dict:
    with (Path(__file__).with_name("risk_config.json")).open(encoding="utf-8") as handle:
        return json.load(handle)
