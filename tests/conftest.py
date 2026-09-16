from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_ROOT = Path("/tmp/carma-pytest")
TEST_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = TEST_ROOT / "test.db"
DB_PATH.unlink(missing_ok=True)
os.environ["CARMA_DATABASE_URL"] = f"sqlite:///{DB_PATH}"
os.environ["CARMA_UPLOAD_DIR"] = str(TEST_ROOT / "uploads")
os.environ["CARMA_ANALYSIS_DIR"] = str(TEST_ROOT / "uploads" / "analysis")
os.environ["CARMA_VEHICLE_DETECTOR_ENABLED"] = "false"
os.environ["CARMA_QWEN_QUALITY_ENABLED"] = "false"
os.environ["CARMA_QWEN_DAMAGE_ENABLED"] = "false"
os.environ["CARMA_ANGLE_CLASSIFIER_ENABLED"] = "false"
os.environ["CARMA_CLEANLINESS_ANALYSIS_ENABLED"] = "true"
DATASET_ROOT = TEST_ROOT / "dataset"
DATASET_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["CARMA_DATASET_SOURCE_DIR"] = str(DATASET_ROOT)
os.environ["CARMA_DATASET_MANIFEST_PATH"] = str(TEST_ROOT / "dataset_manifest.csv")

from backend.app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client
