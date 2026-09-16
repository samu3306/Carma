"""Cleanup, extraction, and validation of model-produced JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from .schemas import PredictionValidationError, validate_prediction


FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


class ModelOutputError(ValueError):
    """Raised when raw model text cannot be parsed and validated."""


def clean_json_text(raw_text: str) -> str:
    text = raw_text.strip().lstrip("\ufeff")
    match = FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def extract_json_object(text: str) -> Any:
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as direct_error:
        decoder = json.JSONDecoder()
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                pass
        raise ModelOutputError(f"無法解析模型 JSON：{direct_error}") from direct_error


def parse_model_output(raw_text: str) -> dict[str, Any]:
    try:
        return validate_prediction(extract_json_object(raw_text))
    except PredictionValidationError as error:
        raise ModelOutputError(f"模型 JSON schema 驗證失敗：{error}") from error
