from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class QualityAnalyzer(ABC):
    @abstractmethod
    def analyze(self, image_path: Path, expected_view: str, existing_hashes: list[str]) -> dict[str, Any]:
        raise NotImplementedError


class DamageDetector(ABC):
    @abstractmethod
    def compare(self, baseline_path: Path, current_path: Path, output_path: Path) -> dict[str, Any]:
        raise NotImplementedError


class CleanlinessClassifier(ABC):
    @abstractmethod
    def classify(self, image_path: Path) -> dict[str, Any]:
        raise NotImplementedError
