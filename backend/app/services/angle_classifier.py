"""Runtime adapter for the human-trained six-view classifier."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from scripts.train_retake_baseline import FEATURE_LAYER, preprocess


EXTERIOR_VIEWS = {"left_front", "right_front", "left_rear", "right_rear"}
INTERIOR_VIEWS = {"interior_front", "interior_rear"}


class _LinearSvmAngleHead:
    """Shared metadata, SVM, and centroid confidence calculation."""

    method = "human_trained_linear_svm"

    def __init__(self, model_dir: Path) -> None:
        metadata = np.load(model_dir / "angle_metadata.npz")
        self.mean = metadata["mean"].astype(np.float32)
        self.std = metadata["std"].astype(np.float32)
        self.centroids = metadata["centroids"].astype(np.float32)
        self.labels = [str(value) for value in metadata["labels"].tolist()]
        self.svm = cv2.ml.SVM_load(str(model_dir / "angle_svm.xml"))

    def _classify(self, feature: np.ndarray) -> dict[str, Any]:
        normalized = ((feature.reshape(1, -1) - self.mean) / self.std).astype(np.float32)
        _, prediction = self.svm.predict(normalized)
        class_id = int(prediction.reshape(-1)[0])

        unit = normalized.reshape(-1)
        unit /= max(float(np.linalg.norm(unit)), 1e-8)
        similarities = self.centroids @ unit
        probabilities = np.exp((similarities - similarities.max()) / 0.08)
        probabilities /= probabilities.sum()
        centroid_id = int(probabilities.argmax())
        confidence = float(probabilities[class_id])
        if centroid_id != class_id:
            confidence = min(confidence, 0.49)
        return {
            "detected_view": self.labels[class_id],
            "confidence": round(confidence, 4),
            "centroid_view": self.labels[centroid_id],
            "method": self.method,
        }


class TrainedAngleClassifier(_LinearSvmAngleHead):
    method = "human_trained_mobilenetv2_svm"

    def __init__(self, onnx_path: Path, model_dir: Path) -> None:
        super().__init__(model_dir)
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        self._lock = RLock()

    def predict(self, image_path: Path) -> dict[str, Any]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"無法讀取圖片：{image_path}")
        with self._lock:
            self.net.setInput(preprocess(image))
            feature = self.net.forward(FEATURE_LAYER).reshape(1, -1)
            return self._classify(feature)


class DinoV2AngleClassifier(_LinearSvmAngleHead):
    """Frozen DINOv2 ViT-S/14 embeddings with the human-trained linear SVM."""

    method = "human_trained_dinov2_vits14_svm"

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        model_dir: Path,
        device: str = "auto",
    ) -> None:
        super().__init__(model_dir)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.requested_device = device
        self.device: str | None = None
        self.processor: Any = None
        self.model: Any = None
        self._lock = RLock()

    def load(self) -> None:
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoImageProcessor, AutoModel

            if self.requested_device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif self.requested_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("DINOv2 指定使用 CUDA，但目前 .venv 的 PyTorch 無法使用 CUDA")
            else:
                device = self.requested_device
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True,
                use_fast=False,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True,
            ).to(device)
            self.model.eval()
            self.device = device

    def warmup(self) -> None:
        self.load()

    def predict(self, image_path: Path) -> dict[str, Any]:
        self.load()
        import torch

        try:
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB").copy()
        except OSError as exc:
            raise ValueError(f"無法讀取圖片：{image_path}") from exc
        try:
            with self._lock:
                inputs = self.processor(images=[image], return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    output = self.model(**inputs)
                pooled = getattr(output, "pooler_output", None)
                if pooled is None:
                    pooled = output.last_hidden_state[:, 0]
                feature = pooled.detach().float().cpu().numpy()
                return self._classify(feature)
        finally:
            image.close()


class DinoV2OnnxAngleClassifier(_LinearSvmAngleHead):
    """CPU-only DINOv2 feature extractor using ONNX Runtime."""

    method = "human_trained_dinov2_vits14_onnx_svm"
    _mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    _std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, onnx_path: Path, model_dir: Path) -> None:
        super().__init__(model_dir)
        self.onnx_path = onnx_path
        self.session: Any = None
        self._lock = RLock()

    def load(self) -> None:
        if self.session is not None:
            return
        with self._lock:
            if self.session is not None:
                return
            if not self.onnx_path.is_file():
                raise RuntimeError(f"DINOv2 ONNX 模型不存在：{self.onnx_path}")
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

    def warmup(self) -> None:
        self.load()
        blank = np.zeros((1, 3, 224, 224), dtype=np.float32)
        with self._lock:
            self.session.run(["features"], {"pixel_values": blank})

    @classmethod
    def preprocess(cls, image: Image.Image) -> np.ndarray:
        width, height = image.size
        if width < height:
            resized_width, resized_height = 256, int(256 * height / width)
        else:
            resized_height, resized_width = 256, int(256 * width / height)
        image = image.resize(
            (resized_width, resized_height), Image.Resampling.BICUBIC
        )
        left = (resized_width - 224) // 2
        top = (resized_height - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        values = np.asarray(image, dtype=np.float32) / 255.0
        values = (values - cls._mean) / cls._std
        return values.transpose(2, 0, 1)[None].astype(np.float32)

    def predict(self, image_path: Path) -> dict[str, Any]:
        self.load()
        try:
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                values = self.preprocess(image)
        except OSError as exc:
            raise ValueError(f"無法讀取圖片：{image_path}") from exc
        with self._lock:
            feature = self.session.run(
                ["features"], {"pixel_values": values}
            )[0]
        return self._classify(feature)


def apply_angle_to_quality(
    quality_result: dict[str, Any],
    angle_result: dict[str, Any],
    expected_view: str,
) -> dict[str, Any]:
    """Apply the trained exact-view result without requiring a VLM."""

    result = dict(quality_result)
    metrics = dict(result.get("metrics") or {})
    checks = [
        dict(check)
        for check in metrics.get("checks", [])
        if check.get("code") not in {"exact_view", "trained_exact_view"}
    ]
    detected = str(angle_result["detected_view"])
    confidence = float(angle_result["confidence"])
    mismatch = detected != expected_view
    instruction = None
    if mismatch:
        if expected_view in INTERIOR_VIEWS and detected in EXTERIOR_VIEWS:
            instruction = "這是車內拍攝欄位，請改拍指定車室；車外照片不能使用。"
        elif expected_view in EXTERIOR_VIEWS and detected in INTERIOR_VIEWS:
            instruction = "這是車外拍攝欄位，請依指定角度拍攝車身與車牌。"
        else:
            instruction = "拍攝位置與官方要求不符，請依站位示意重新拍攝指定角度。"
    checks.append({
        "code": "trained_exact_view",
        "label": "六方向角度",
        "status": "fail" if mismatch else "pass",
        "score": confidence,
        "message": (
            f"模型判定為 {detected}，官方要求為 {expected_view}"
            if mismatch else f"模型判定角度為 {detected}"
        ),
        "instruction": instruction,
        "mode": angle_result["method"],
    })
    result["detected_view"] = detected
    result["model_confidence"] = confidence
    if mismatch:
        result["passed"] = False
        result["retake_instruction"] = instruction
        result["warnings"] = [instruction]
    review_required = any(check.get("status") == "review" for check in checks)
    metrics["checks"] = checks
    metrics["review_required"] = review_required
    metrics["capture_decision"] = (
        "retake" if not result.get("passed") else "confirm" if review_required else "accept"
    )
    metrics["angle_method"] = angle_result["method"]
    metrics["angle_confidence"] = confidence
    metrics["limitations"] = [
        item for item in metrics.get("limitations", [])
        if item != "exact_six_view_unverified"
    ]
    result["metrics"] = metrics
    return result


def apply_trained_angle(
    qwen_result: dict[str, Any],
    angle_result: dict[str, Any],
    expected_view: str,
    retake_confidence: float,
) -> dict[str, Any]:
    """Replace Qwen's angle opinion while retaining its other quality findings."""

    result = dict(qwen_result)
    detected = angle_result["detected_view"]
    confidence = float(angle_result["confidence"])
    mismatch = detected != expected_view
    original_issues = list(result.get("issues", []))
    cross_category_mismatch = mismatch and (
        (detected in EXTERIOR_VIEWS and expected_view in INTERIOR_VIEWS)
        or (detected in INTERIOR_VIEWS and expected_view in EXTERIOR_VIEWS)
    )
    # Official capture slots are strict: every predicted view must exactly match.
    # Confidence is still reported for observability, but never permits a mismatch.
    hard_mismatch = mismatch
    review = False

    issues = [issue for issue in original_issues if issue != "wrong_view"]
    non_angle_failed = bool(issues)
    result.update({
        "passed": not non_angle_failed and not hard_mismatch,
        "review_required": review and not non_angle_failed,
        "detected_view": detected,
        "model_confidence": confidence,
        "issues": [*issues, *( ["wrong_view"] if hard_mismatch else [])],
    })
    checks = [
        check for check in result.get("checks", []) if check.get("code") != "vlm_exact_view"
    ]
    checks.insert(0, {
        "code": "trained_exact_view",
        "label": "六方向角度",
        "status": "fail" if hard_mismatch else "review" if review else "pass",
        "score": confidence,
        "message": (
            (
                f"照片類型錯誤：模型判定為 {detected}，但此欄位要求 {expected_view}"
                if cross_category_mismatch
                else f"模型判定為 {detected}，官方要求為 {expected_view}，請重新拍攝"
            )
            if mismatch else f"人工標訓練模型判定角度為 {detected}"
        ),
        "instruction": (
            "這是車內拍攝欄位，請改拍指定車室；車外照片不能使用。"
            if hard_mismatch and expected_view in INTERIOR_VIEWS
            else "這是車外拍攝欄位，請依指定角度拍攝車身與車牌。"
            if hard_mismatch and expected_view in EXTERIOR_VIEWS and detected in INTERIOR_VIEWS
            else "請依官方站位示意重新拍攝指定角度；判定位置必須與要求完全相同。"
            if hard_mismatch
            else None
        ),
        "mode": angle_result["method"],
    })
    result["checks"] = checks
    if hard_mismatch:
        result["retake_instruction"] = checks[0]["instruction"]
        result["warnings"] = [result["retake_instruction"]]
    elif non_angle_failed:
        result["retake_instruction"] = qwen_result.get("retake_instruction")
    else:
        result["retake_instruction"] = None
        result["warnings"] = []
    result["angle_method"] = angle_result["method"]
    return result
