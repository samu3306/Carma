from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .interfaces import QualityAnalyzer
from .vehicle import YoloVehicleDetector


EXTERIOR_VIEWS = {"left_front", "right_front", "left_rear", "right_rear"}
INTERIOR_VIEWS = {"interior_front", "interior_rear"}
def apply_detected_rotation(image: np.ndarray, rotation: str) -> np.ndarray:
    rotation_codes = {
        "counterclockwise_90": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "clockwise_90": cv2.ROTATE_90_CLOCKWISE,
        "rotate_180": cv2.ROTATE_180,
    }
    code = rotation_codes.get(rotation)
    return cv2.rotate(image, code) if code is not None else image



def dhash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8))
    bits = resized[:, 1:] > resized[:, :-1]
    return f"{sum(int(value) << index for index, value in enumerate(bits.flatten())):016x}"


def hash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def interior_zone_coverage(gray: np.ndarray, edges: np.ndarray) -> tuple[int, list[dict]]:
    height, width = gray.shape
    zones: list[dict] = []
    active = 0
    for row in range(2):
        for column in range(3):
            y1, y2 = row * height // 2, (row + 1) * height // 2
            x1, x2 = column * width // 3, (column + 1) * width // 3
            tile = gray[y1:y2, x1:x2]
            tile_edges = edges[y1:y2, x1:x2]
            contrast = float(tile.std())
            edge_density = float((tile_edges > 0).mean())
            is_active = contrast >= 22 or edge_density >= 0.035
            active += int(is_active)
            zones.append({
                "row": row,
                "column": column,
                "active": is_active,
                "contrast": round(contrast, 1),
                "edge_density": round(edge_density, 3),
            })
    return active, zones


def plate_candidate(image: np.ndarray, bbox: dict | None) -> bool:
    if bbox:
        x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        roi = image[y:y + height, x:x + width]
    else:
        roi = image
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 180)
    area_total = gray.shape[0] * gray.shape[1]
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        _, _, width, height = cv2.boundingRect(contour)
        ratio = width / max(height, 1)
        relative = (width * height) / max(area_total, 1)
        if 2.3 < ratio < 6.8 and 0.0015 < relative < 0.085:
            return True
    return False


class DemoQualityAnalyzer(QualityAnalyzer):
    """Structured capture review with optional YOLO vehicle detection.

    Exact six-view classification and plate OCR remain unverified Demo capabilities.
    """

    def __init__(self, vehicle_detector: YoloVehicleDetector | None = None):
        self.vehicle_detector = vehicle_detector

    def analyze(self, image_path: Path, expected_view: str, existing_hashes: list[str]) -> dict:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("無法解碼圖片，請改用 JPG、PNG 或 WebP")
        height, width = image.shape[:2]
        if width >= height * 1.1:
            orientation = "landscape"
        elif height >= width * 1.1:
            orientation = "portrait"
        else:
            orientation = "square"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        dark_ratio = float((gray < 25).mean())
        over_ratio = float((gray > 245).mean())
        edges = cv2.Canny(gray, 60, 160)
        fallback_foreground = cv2.dilate(edges, np.ones((9, 9), np.uint8), iterations=2)
        fallback_coverage = float(min(0.95, max(0.05, fallback_foreground.mean() / 255.0 * 1.7)))

        vehicle = (
            self.vehicle_detector.detect(image, retry_rotations=True)
            if self.vehicle_detector is not None
            else {"available": False, "detected": False, "confidence": 0.0, "bbox": None}
        )
        detected_rotation = str(vehicle.get("rotation_applied", "none"))
        analysis_image = apply_detected_rotation(image, detected_rotation)
        analysis_height, analysis_width = analysis_image.shape[:2]
        if analysis_width >= analysis_height * 1.1:
            orientation = "landscape"
        elif analysis_height >= analysis_width * 1.1:
            orientation = "portrait"
        else:
            orientation = "square"
        coverage = float(vehicle.get("coverage", fallback_coverage) if vehicle.get("detected") else fallback_coverage)
        plate = plate_candidate(analysis_image, vehicle.get("bbox")) if expected_view in EXTERIOR_VIEWS else False
        active_zones, zones = interior_zone_coverage(gray, edges)
        image_hash = dhash(image)
        duplicate = any(hash_distance(image_hash, old) <= 4 for old in existing_hashes if old)

        warnings: list[str] = []
        critical: list[str] = []
        checks: list[dict] = []

        def add_check(code: str, label: str, status: str, score: float | None, message: str, instruction: str | None, mode: str) -> None:
            checks.append({
                "code": code,
                "label": label,
                "status": status,
                "score": score,
                "message": message,
                "instruction": instruction,
                "mode": mode,
            })
            if status == "fail" and instruction:
                critical.append(instruction)
            elif status == "review":
                warnings.append(message)

        resolution_ok = min(width, height) >= 480
        add_check("resolution", "解析度", "pass" if resolution_ok else "fail", min(width, height), f"{width} × {height}", None if resolution_ok else "影像尺寸過低，請使用原始相機畫質重新拍攝", "cv_rule")
        if expected_view in INTERIOR_VIEWS:
            orientation_ok = orientation == "landscape"
            portrait_coverage_ok = active_zones >= 5
            orientation_status = "pass" if orientation_ok else "review" if portrait_coverage_ok else "fail"
            orientation_message = "橫向拍攝，可涵蓋完整車室" if orientation_ok else f"直向拍攝，內容涵蓋 {active_zones}/6 區"
            orientation_instruction = None if orientation_ok or portrait_coverage_ok else "畫面只涵蓋局部車室，建議將手機橫放後重新拍攝"
            add_check(
                "orientation", "拍攝方向", orientation_status, analysis_width / max(analysis_height, 1),
                orientation_message, orientation_instruction,
                "image_dimensions",
            )
        else:
            direction_label = {"landscape": "橫向", "portrait": "直向", "square": "接近正方形"}[orientation]
            add_check("orientation", "拍攝方向", "pass", analysis_width / max(analysis_height, 1), f"{direction_label}照片；車外照以車身完整為準", None, "image_dimensions")

        if detected_rotation != "none":
            add_check("rotation_recovery", "方向校正", "pass", 1, f"偵測到方向資訊不一致，已自動校正：{detected_rotation}", None, "rotation_retry")

        severe_blur_threshold = 25 if expected_view in INTERIOR_VIEWS else 45
        if blur < severe_blur_threshold:
            sharp_status = "fail"
            sharp_message = "偵測到明顯手震或失焦"
            sharp_instruction = "影像模糊，請穩定手機並點擊拍攝區域對焦後重新拍攝"
        elif expected_view in INTERIOR_VIEWS and blur < 45:
            sharp_status, sharp_message, sharp_instruction = "review", "略有模糊，但仍可辨識主要車室", None
        else:
            sharp_status, sharp_message, sharp_instruction = "pass", "畫面清晰", None
        add_check("sharpness", "清晰度", sharp_status, round(blur, 1), sharp_message, sharp_instruction, "cv_rule")

        exposure_ok = not (brightness < 38 or dark_ratio > 0.55 or brightness > 220 or over_ratio > 0.35)
        if brightness < 38 or dark_ratio > 0.55:
            exposure_instruction = "影像過暗，請移至光線充足處或開啟照明後重新拍攝"
        elif brightness > 220 or over_ratio > 0.35:
            exposure_instruction = "影像過曝，請避開強光或降低曝光後重新拍攝"
        else:
            exposure_instruction = None
        add_check("exposure", "曝光", "pass" if exposure_ok else "fail", round(brightness, 1), "曝光正常" if exposure_ok else exposure_instruction or "曝光異常", exposure_instruction, "cv_rule")

        lens_ok = contrast >= 25 or blur >= 110
        add_check("lens", "鏡頭潔淨", "pass" if lens_ok else "fail", round(contrast, 1), "對比正常" if lens_ok else "畫面低對比，鏡頭可能有油污或霧氣", None if lens_ok else "畫面霧化，請擦拭鏡頭並重新拍攝", "cv_rule")

        add_check("duplicate", "照片重複", "fail" if duplicate else "pass", 0 if duplicate else 1, "與其他位置高度重複" if duplicate else "未發現重複", "此照片與同訂單其他照片高度重複，請移動到指定位置重新拍攝" if duplicate else None, "perceptual_hash")

        if expected_view in EXTERIOR_VIEWS:
            if vehicle.get("available"):
                if not vehicle.get("detected"):
                    add_check("vehicle", "車輛入鏡", "fail", 0, "AI 未偵測到完整車輛", "未偵測到車輛，請對準車身並依畫面站位重新拍攝", "yolov4_tiny")
                else:
                    add_check("vehicle", "車輛入鏡", "pass", vehicle["confidence"], "已偵測到車輛", None, "yolov4_tiny")
                    vehicle_coverage = float(vehicle["coverage"])
                    if vehicle_coverage < 0.08:
                        add_check("distance", "拍攝距離", "fail", vehicle_coverage, "車輛在畫面中占比過小", "車輛距離太遠，請往前靠近至車身約占畫面一半", "yolov4_tiny")
                    elif vehicle_coverage > 0.82 or len(vehicle.get("touching_edges", [])) >= 2:
                        add_check("distance", "拍攝距離", "fail", vehicle_coverage, "車身過近或多處超出畫面", "距離太近或車身被裁切，請往後退一步並完整納入車頭與車側", "yolov4_tiny")
                    else:
                        add_check("distance", "拍攝距離", "pass", vehicle_coverage, "車身占比適中", None, "yolov4_tiny")
                    touching = vehicle.get("touching_edges", [])
                    add_check("framing", "車身完整", "review" if touching else "pass", 1 - len(touching) / 4, f"車身接近畫面邊緣：{', '.join(touching)}" if touching else "車身未明顯裁切", None, "yolov4_tiny")
            else:
                fallback_ok = fallback_coverage >= 0.16
                add_check("vehicle", "車輛入鏡", "pass" if fallback_ok else "fail", round(fallback_coverage, 3), "使用輪廓規則估計" if fallback_ok else "主要拍攝區域占比過小", None if fallback_ok else "車輛距離太遠，請往前靠近", "fallback_cv_rule")

            add_check("plate", "車牌可見", "pass" if plate else "review", 1 if plate else 0, "找到車牌形狀候選" if plate else "未找到清楚車牌候選，請確認車牌完整可辨識", None, "contour_candidate")
            add_check("exact_view", "左右與前後角度", "review", None, "專用六角度模型尚未達到可靠門檻，請依俯視站位圖確認", None, "model_pending")
            detected_view = "exterior"
        else:
            vehicle_confidence = float(vehicle.get("confidence", 0))
            vehicle_coverage = float(vehicle.get("coverage", 0))
            high_confidence_vehicle = vehicle_confidence >= 0.45 and vehicle_coverage >= 0.10
            recovered_large_vehicle = detected_rotation != "none" and vehicle_confidence >= 0.20 and vehicle_coverage >= 0.35
            dominant_vehicle = vehicle_confidence >= 0.20 and vehicle_coverage >= 0.62
            wrong_scene = bool(vehicle.get("available") and vehicle.get("detected") and (high_confidence_vehicle or recovered_large_vehicle or dominant_vehicle))
            add_check("scene", "車內／車外", "fail" if wrong_scene else "pass", vehicle.get("confidence", 0), "疑似上傳車外照片" if wrong_scene else "未發現完整車外車輛", "這一格需要車內照片，請拍攝座椅、腳踏區與車室全景" if wrong_scene else None, "yolov4_tiny" if vehicle.get("available") else "cv_rule")
            edge_density = float((edges > 0).mean())
            if wrong_scene:
                add_check("interior_coverage", "車內涵蓋", "review", None, "已判定為車外場景，跳過車內涵蓋評分", None, "not_applicable")
                add_check("cleanliness_preview", "髒污預覽", "review", None, "已判定為車外場景，跳過車內整潔度評分", None, "not_applicable")
            else:
                coverage_ok = active_zones >= 4
                add_check("interior_coverage", "車內涵蓋", "pass" if coverage_ok else "fail", active_zones / 6, f"車內有效區域 {active_zones}/6", None if coverage_ok else "只拍到局部座位，請橫向拍攝並包含座椅、腳踏區及另一側車室", "zone_coverage")
                dirty_hint = edge_density > 0.19
                add_check("cleanliness_preview", "髒污預覽", "review" if dirty_hint else "pass", round(edge_density, 3), "紋理較複雜，可能有雜物或座椅花紋，完成後由整潔度模組複核" if dirty_hint else "未見明顯高密度雜物紋理", None, "demo_rule")
            add_check("exact_view", "前／後車內", "review", None, "前後車內專用模型尚未達到可靠門檻，請依拍攝指引確認範圍", None, "model_pending")
            detected_view = "exterior" if wrong_scene else "interior"

        deductions = (
            min(35, max(0, 45 - blur) * 0.6)
            + min(25, abs(brightness - 128) * 0.18)
            + (30 if not resolution_ok else 0)
            + (35 if duplicate else 0)
            + min(35, len(critical) * 12)
        )
        score = round(max(0.0, min(100.0, 100 - deductions)), 1)
        failed_codes = {item["code"] for item in checks if item["status"] == "fail"}
        if "scene" in failed_codes:
            score = min(score, 20.0)
        elif "interior_coverage" in failed_codes:
            score = min(score, 45.0)
        passed = not critical
        review_required = any(item["status"] == "review" for item in checks)
        return {
            "passed": passed,
            "quality_score": score,
            "detected_view": detected_view,
            "expected_view": expected_view,
            "blur_score": round(blur, 2),
            "brightness_score": round(brightness, 2),
            "vehicle_coverage": round(coverage, 3),
            "plate_detected": plate,
            "warnings": critical + warnings,
            "retake_instruction": critical[0] if critical else None,
            "model_confidence": float(vehicle.get("confidence", 0.55)) if expected_view in EXTERIOR_VIEWS else 0.62,
            "perceptual_hash": image_hash,
            "metrics": {
                "width": width,
                "height": height,
                "orientation": orientation,
                "contrast": round(contrast, 2),
                "dark_ratio": round(dark_ratio, 3),
                "analysis_width": analysis_width,
                "analysis_height": analysis_height,
                "rotation_applied": detected_rotation,
                "overexposed_ratio": round(over_ratio, 3),
                "duplicate": duplicate,
                "vehicle_detection": vehicle,
                "interior_active_zones": active_zones,
                "interior_zones": zones,
                "checks": checks,
                "review_required": review_required,
                "capture_decision": "retake" if not passed else "confirm" if review_required else "accept",
                "inference_mode": "hybrid_yolo_cv_demo" if vehicle.get("available") else "demo_rules",
                "limitations": ["exact_six_view_unverified", "plate_is_shape_candidate_not_ocr"],
            },
        }
