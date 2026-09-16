"""Backend adapter for Qwen pickup/return damage-pair comparison."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.vlm.damage_pair import QwenDamagePairAnalyzer
from src.vlm.qwen_vl_client import ImageLoadError, InferenceError
from .damage import DemoDamageDetector


class QwenDamageDetector:
    """Return the existing DamageDetector contract from Qwen pair predictions."""

    size = (800, 450)

    def __init__(
        self,
        analyzer: QwenDamagePairAnalyzer,
        cv_detector: DemoDamageDetector | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.cv_detector = cv_detector or DemoDamageDetector()

    def compare(
        self, baseline_path: Path, current_path: Path, output_path: Path
    ) -> dict:
        cv_result = self.cv_detector.compare(
            baseline_path, current_path, output_path
        )
        error: str | None = None
        try:
            inference = self.analyzer.infer_pair(baseline_path, current_path)
            prediction = inference.prediction if inference.json_valid else None
            error = inference.error
        except (ImageLoadError, InferenceError, OSError) as exc:
            inference = None
            prediction = None
            error = f"{type(exc).__name__}: {exc}"

        if prediction is None:
            prediction = {
                "decision": "uncertain",
                "damage_type": "none",
                "severity": "none",
                "vehicle_area": "",
                "confidence": 0.0,
                "changed_regions": [],
                "explanation": f"模型無法完成可靠比較，需人工複核。{error or ''}".strip(),
            }

        decision = prediction["decision"]
        confidence = float(prediction["confidence"])
        suspected = decision == "new_damage"
        uncertain = decision == "uncertain"
        cv_suspected = bool(cv_result["suspected_new_damage"])
        model_disagreement = not uncertain and cv_suspected != suspected
        severity = prediction["severity"] if suspected else "none"
        difference_score = (
            confidence
            if suspected
            else round((1.0 - confidence) * 0.10, 4)
            if decision == "no_new_damage"
            else 0.20
        )
        explanation = prediction["explanation"]
        if suspected:
            explanation = (
                f"疑似新增{self._type_label(prediction['damage_type'])}"
                f"（{prediction['vehicle_area'] or '位置待確認'}）：{explanation}"
            )
        elif uncertain:
            explanation = f"照片對無法可靠判定新車損，需人工複核：{explanation}"
        else:
            explanation = f"未發現可確認的新車損：{explanation}"
        if cv_result["requires_manual_review"]:
            explanation += "；CV 特徵對齊信心不足，需人工確認兩張照片是否可可靠對位"
        elif model_disagreement:
            explanation += "；VLM 與 CV 差異熱區判斷不一致，已送人工複核"
        return {
            "difference_score": round(float(difference_score), 4),
            "ssim_score": cv_result["ssim_score"],
            "suspected_new_damage": suspected,
            "severity": severity,
            "confidence": confidence,
            "requires_manual_review": (
                suspected
                or uncertain
                or bool(cv_result["requires_manual_review"])
                or model_disagreement
            ),
            "issue_regions": cv_result["issue_regions"],
            "heatmap_path": str(output_path),
            "explanation": explanation,
            "alignment_confidence": cv_result["alignment_confidence"],
            "inference_mode": "qwen3_vl_pair_plus_orb_ssim",
            "decision": decision,
            "damage_type": prediction["damage_type"],
            "vehicle_area": prediction["vehicle_area"],
            "changed_regions": prediction["changed_regions"],
            "cv_difference_score": cv_result["difference_score"],
            "cv_suspected_new_damage": cv_suspected,
            "model_disagreement": model_disagreement,
            "inference_seconds": (
                inference.inference_seconds if inference is not None else None
            ),
            "gpu_peak_memory_mb": (
                inference.gpu_peak_memory_mb if inference is not None else None
            ),
        }

    @classmethod
    def _write_comparison(
        cls,
        baseline_path: Path,
        current_path: Path,
        output_path: Path,
        decision: str,
    ) -> None:
        baseline = cv2.imread(str(baseline_path))
        current = cv2.imread(str(current_path))
        if baseline is None or current is None:
            raise ValueError("無法讀取車損比對圖片")
        width, height = cls.size
        before = cv2.resize(baseline, (width // 2, height))
        after = cv2.resize(current, (width // 2, height))
        canvas = np.hstack([before, after])
        cv2.rectangle(canvas, (0, 0), (width // 2, 38), (20, 20, 20), -1)
        cv2.rectangle(canvas, (width // 2, 0), (width, 38), (20, 20, 20), -1)
        cv2.putText(
            canvas, "PICKUP", (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )
        color = (0, 0, 255) if decision == "new_damage" else (0, 180, 255) if decision == "uncertain" else (30, 180, 80)
        cv2.putText(
            canvas, f"RETURN - {decision.upper()}", (width // 2 + 14, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    @staticmethod
    def _type_label(value: str) -> str:
        return {
            "scratch": "刮痕",
            "dent": "凹陷",
            "crack": "裂痕",
            "paint_loss": "掉漆",
            "broken_part": "零件破損",
            "other": "車損",
        }.get(value, "車損")
