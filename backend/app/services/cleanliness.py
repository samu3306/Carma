from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .interfaces import CleanlinessClassifier


class DemoCleanlinessClassifier(CleanlinessClassifier):
    """Configurable visual-clutter heuristic, not a trained cleanliness model."""

    def classify(self, image_path: Path) -> dict:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("無法讀取車內圖片")
        image = cv2.resize(image, (640, 360))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 170)
        edge_density = float((edges > 0).mean())
        local_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        high_saturation = float((hsv[:, :, 1] > 145).mean())
        clutter = min(1.0, edge_density * 4.2 + min(local_variance / 5000, 0.35) + high_saturation * 0.7)
        cleanliness_score = round(max(0.0, 100 * (1 - clutter)), 1)
        if cleanliness_score < 40:
            level = "dirty"
            issues = ["視覺雜物或污漬紋理偏多", "建議人工確認座椅與腳踏區"]
            immediate = True
            explanation = "Demo 規則偵測到高邊緣密度與色彩異常，判為髒污"
        elif cleanliness_score < 72:
            level = "normal"
            issues = ["局部區域可能需要簡易整理"] if cleanliness_score < 58 else []
            immediate = False
            explanation = "Demo 規則判為一般可接受整潔度"
        else:
            level = "clean"
            issues = []
            immediate = False
            explanation = "Demo 規則未發現明顯雜物或高密度污漬紋理"
        return {
            "cleanliness_level": level,
            "cleanliness_score": cleanliness_score,
            "detected_issues": issues,
            "issue_regions": [],
            "immediate_cleaning_required": immediate,
            "confidence": 0.62,
            "explanation": explanation,
            "metrics": {"edge_density": round(edge_density, 4), "high_saturation_ratio": round(high_saturation, 4)},
            "inference_mode": "demo_rules",
        }
