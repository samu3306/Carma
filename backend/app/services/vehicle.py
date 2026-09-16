from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import numpy as np


VEHICLE_CLASS_IDS = {2, 3, 5, 7}
ROTATION_CANDIDATES = (
    ("counterclockwise_90", cv2.ROTATE_90_COUNTERCLOCKWISE),
    ("clockwise_90", cv2.ROTATE_90_CLOCKWISE),
    ("rotate_180", cv2.ROTATE_180),
)


class YoloVehicleDetector:
    """COCO YOLOv4-tiny adapter for vehicle presence, framing, and rotation recovery."""

    def __init__(self, config_path: Path, weights_path: Path, confidence_threshold: float = 0.18):
        self.available = config_path.is_file() and weights_path.is_file()
        self.confidence_threshold = confidence_threshold
        self.net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path)) if self.available else None
        self.output_layers = self.net.getUnconnectedOutLayersNames() if self.net is not None else []
        self._inference_lock = Lock()

    def _detect_once(self, image: np.ndarray, rotation_applied: str = "none") -> dict:
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        with self._inference_lock:
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
        candidates: list[tuple[float, list[int]]] = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(detection[4] * scores[class_id])
                if class_id not in VEHICLE_CLASS_IDS or confidence < self.confidence_threshold:
                    continue
                center_x, center_y, box_width, box_height = detection[:4]
                x = max(0, int((center_x - box_width / 2) * width))
                y = max(0, int((center_y - box_height / 2) * height))
                w = min(width - x, int(box_width * width))
                h = min(height - y, int(box_height * height))
                if w > 0 and h > 0:
                    candidates.append((confidence, [x, y, w, h]))
        base = {
            "available": True,
            "analysis_width": width,
            "analysis_height": height,
            "rotation_applied": rotation_applied,
        }
        if not candidates:
            return base | {"detected": False, "confidence": 0.0, "bbox": None}
        confidence, bbox = max(candidates, key=lambda item: item[0] * item[1][2] * item[1][3])
        x, y, w, h = bbox
        coverage = (w * h) / float(width * height)
        margins = {
            "left": x / width,
            "top": y / height,
            "right": (width - x - w) / width,
            "bottom": (height - y - h) / height,
        }
        return base | {
            "detected": True,
            "confidence": round(confidence, 3),
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "coverage": round(coverage, 3),
            "margins": {key: round(value, 3) for key, value in margins.items()},
            "touching_edges": [key for key, value in margins.items() if value < 0.018],
        }

    @staticmethod
    def _ranking(result: dict) -> float:
        if not result.get("detected"):
            return 0.0
        return float(result["confidence"]) * (0.6 + min(float(result.get("coverage", 0)), 0.7))

    def detect(self, image: np.ndarray, retry_rotations: bool = True) -> dict:
        if self.net is None:
            return {
                "available": False,
                "detected": False,
                "confidence": 0.0,
                "bbox": None,
                "rotation_applied": "none",
                "rotation_retried": False,
            }

        primary = self._detect_once(image)
        reliable_primary = (
            primary.get("detected")
            and float(primary.get("confidence", 0)) >= 0.35
            and float(primary.get("coverage", 0)) >= 0.06
        )
        if reliable_primary or not retry_rotations:
            return primary | {"rotation_retried": False}

        results = [primary]
        for name, code in ROTATION_CANDIDATES:
            results.append(self._detect_once(cv2.rotate(image, code), name))
        best = max(results, key=self._ranking)
        return best | {"rotation_retried": True}
