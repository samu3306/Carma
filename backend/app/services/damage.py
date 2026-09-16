from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from ..config import get_risk_config
from .interfaces import DamageDetector


class DemoDamageDetector(DamageDetector):
    """ORB alignment + illumination-normalized SSIM difference pipeline."""

    size = (800, 450)

    def _align(self, baseline: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float, bool]:
        base_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(1800)
        key_a, desc_a = orb.detectAndCompute(base_gray, None)
        key_b, desc_b = orb.detectAndCompute(current_gray, None)
        if desc_a is None or desc_b is None or len(key_a) < 12 or len(key_b) < 12:
            return current, 0.15, True
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_b, desc_a, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
        if len(good) < 10:
            return current, min(0.35, len(good) / 25), True
        src = np.float32([key_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([key_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if matrix is None or inliers is None:
            return current, 0.2, True
        confidence = float(inliers.mean())
        aligned = cv2.warpPerspective(current, matrix, self.size)
        mismatch = confidence < get_risk_config()["thresholds"]["angle_mismatch"]
        return aligned, confidence, mismatch

    def compare(self, baseline_path: Path, current_path: Path, output_path: Path) -> dict:
        baseline = cv2.imread(str(baseline_path))
        current = cv2.imread(str(current_path))
        if baseline is None or current is None:
            raise ValueError("無法讀取車損比對圖片")
        baseline = cv2.resize(baseline, self.size)
        current = cv2.resize(current, self.size)
        aligned, alignment_confidence, angle_mismatch = self._align(baseline, current)

        base_gray = cv2.equalizeHist(cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY))
        current_gray = cv2.equalizeHist(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
        ssim_score, diff = structural_similarity(base_gray, current_gray, full=True)
        difference = ((1.0 - diff) * 255).astype("uint8")
        difference = cv2.GaussianBlur(difference, (5, 5), 0)
        _, mask = cv2.threshold(difference, 48, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        significant_area = 0
        visual = aligned.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 260:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            significant_area += int(area)
            regions.append({"x": x, "y": y, "width": w, "height": h, "score": round(min(1.0, area / 12000), 3)})
            cv2.rectangle(visual, (x, y), (x + w, y + h), (0, 36, 255), 3)
        heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(visual, 0.72, heat, 0.28, 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)

        area_ratio = significant_area / float(self.size[0] * self.size[1])
        structural_delta = max(0.0, 1.0 - float(ssim_score))
        difference_score = round(min(1.0, area_ratio * 4.0 + structural_delta * 0.45), 4)
        threshold = get_risk_config()["thresholds"]["damage_suspected"]
        suspected = difference_score >= threshold and not angle_mismatch
        if difference_score >= get_risk_config()["thresholds"]["damage_high"]:
            severity = "high"
        elif suspected:
            severity = "medium"
        elif difference_score >= threshold * 0.55:
            severity = "low"
        else:
            severity = "none"
        if angle_mismatch:
            explanation = "影像特徵對齊信心不足，角度差異可能過大；不直接判定車損，請人工複核或補拍"
        elif suspected:
            explanation = f"對齊後發現 {len(regions)} 個顯著結構差異熱區，疑似新增車損（Demo 規則）"
        else:
            explanation = "對齊後未發現超過門檻的局部結構差異（Demo 規則）"
        return {
            "difference_score": difference_score,
            "ssim_score": round(float(ssim_score), 4),
            "suspected_new_damage": suspected,
            "severity": severity,
            "confidence": round(max(0.25, min(0.88, alignment_confidence * 0.9)), 3),
            "requires_manual_review": angle_mismatch,
            "issue_regions": regions,
            "heatmap_path": str(output_path),
            "explanation": explanation,
            "alignment_confidence": round(alignment_confidence, 3),
            "inference_mode": "demo_cv_pipeline",
        }
