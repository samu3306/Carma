from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

VIEWS = ["left_front", "right_front", "left_rear", "right_rear", "interior_front", "interior_rear"]


def image_bytes(seed: int, blurred: bool = False) -> bytes:
    rng = np.random.default_rng(seed)
    image = rng.integers(55, 205, (600, 900, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 160), (780, 470), (215, 37 + seed % 20, 43), -1)
    cv2.putText(image, f"VIEW-{seed}", (240, 340), cv2.FONT_HERSHEY_SIMPLEX, 2, (245, 245, 245), 6)
    if blurred:
        image = cv2.GaussianBlur(image, (61, 61), 20)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def create_order(client, number: str = "TEST-1001", plate: str | None = None):
    response = client.post("/api/orders", json={
        "order_number": number,
        "plate_number": plate or ("T-" + number.replace("TEST-", "")[-12:]),
        "pickup_location": "pytest 站",
        "return_location": "pytest 站"
    })
    assert response.status_code == 201, response.text
    return response.json()


def create_session(client, order_id: str, kind: str):
    response = client.post("/api/inspections", json={"order_id": order_id, "inspection_type": kind, "location": "pytest 站"})
    assert response.status_code == 201, response.text
    return response.json()


def upload_six(client, inspection_id: str, seed_offset: int = 0):
    results = {}
    for index, view in enumerate(VIEWS):
        response = client.post(
            f"/api/inspections/{inspection_id}/images",
            data={"expected_view": view},
            files={"file": (f"{view}.jpg", image_bytes(seed_offset + index + 10), "image/jpeg")},
        )
        assert response.status_code == 201, response.text
        results[view] = response.json()
        assert results[view]["quality"]["passed"] is True, results[view]["quality"]
    return results


class StubDamageDetector:
    def compare(self, baseline_path: Path, current_path: Path, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), np.full((120, 240, 3), 90, np.uint8))
        damaged = "left_front" in current_path.name
        return {
            "difference_score": 0.9 if damaged else 0.01,
            "ssim_score": 0.0,
            "suspected_new_damage": damaged,
            "severity": "medium" if damaged else "none",
            "confidence": 0.9,
            "requires_manual_review": damaged,
            "issue_regions": [],
            "heatmap_path": str(output_path),
            "explanation": "疑似新增刮痕" if damaged else "未發現新車損",
            "inference_mode": "qwen3_vl_pair_few_shot",
        }


def test_health_and_uniform_error_format(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["demo_mode"] is True
    invalid = client.post("/api/orders", json={"order_number": "x", "plate_number": "z"})
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"


def test_model_quality_can_be_tested_without_order(client):
    response = client.post(
        "/api/model-test/quality",
        data={"expected_view": "left_front"},
        files={"file": ("test.jpg", image_bytes(77), "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["expected_view"] == "left_front"
    assert payload["temporary_file_deleted"] is True
    assert payload["metrics"]["checks"]
    test_dir = Path("/tmp/carma-pytest/uploads/analysis/model-tests")
    assert not list(test_dir.glob("*"))


def test_incomplete_inspection_is_blocked(client):
    order = create_order(client, "TEST-INCOMPLETE")
    session = create_session(client, order["id"], "pickup")
    response = client.post(f"/api/inspections/{session['id']}/analyze")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "incomplete_inspection"
    assert len(response.json()["error"]["details"]["missing_views"]) == 4


def test_return_requires_completed_pickup_baseline(client):
    order = create_order(client, "TEST-RETURN-BASELINE")
    response = client.post("/api/inspections", json={
        "order_id": order["id"],
        "inspection_type": "return",
        "location": "pytest 站",
    })
    assert response.status_code == 409
    payload = response.json()["error"]
    assert payload["code"] == "pickup_baseline_required"
    assert payload["details"]["missing_views"] == VIEWS[:4]


def test_full_pickup_return_case_review_and_task_flow(client, monkeypatch):
    order = create_order(client, "TEST-FULL", "LEDGER-001")
    pickup = create_session(client, order["id"], "pickup")
    upload_six(client, pickup["id"], 100)
    pickup_analysis = client.post(f"/api/inspections/{pickup['id']}/analyze")
    assert pickup_analysis.status_code == 200, pickup_analysis.text

    monkeypatch.setattr("backend.app.main.damage_detector", StubDamageDetector())
    returned = create_session(client, order["id"], "return")
    evidence = client.post(
        f"/api/inspections/{returned['id']}/evidence",
        data={"purpose": "damage_closeup", "related_view": "left_front", "notes": "使用者主動補充"},
        files={"file": ("damage.jpg", image_bytes(3333), "image/jpeg")},
    )
    assert evidence.status_code == 201, evidence.text
    assert evidence.json()["required_view"] is False
    upload_six(client, returned["id"], 100)
    result = client.post(f"/api/inspections/{returned['id']}/analyze")
    assert result.status_code == 200, result.text
    payload = result.json()
    assert len(payload["damage_comparisons"]) == 4
    assert len(payload["cleanliness"]) == 2
    assert payload["risk"]["risk_level"] in {"green", "yellow", "red"}

    cases = client.get("/api/cases")
    assert cases.status_code == 200
    case = next(item for item in cases.json() if item["order_number"] == "TEST-FULL")
    detail = client.get(f"/api/cases/{case['id']}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert len(detail_payload["order"]["inspections"]) == 2
    assert detail_payload["has_damage"] is True
    assert any(item["alert_type"] == "new_damage" for item in detail_payload["alerts"])
    assert any(item["task_type"] == "claim_review" for item in detail_payload["tasks"])
    alert = next(item for item in detail_payload["alerts"] if item["alert_type"] == "new_damage")
    acknowledged = client.patch(f"/api/alerts/{alert['id']}/acknowledge")
    assert acknowledged.status_code == 200
    assert acknowledged.json()["acknowledged"] is True

    task = client.post(f"/api/cases/{case['id']}/tasks", json={"task_type": "manual_review", "priority": "high", "notes": "pytest"})
    assert task.status_code == 201
    updated = client.patch(f"/api/tasks/{task.json()['id']}", json={"status": "reviewing", "assignee": "測試專員"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "reviewing"

    confirmed = client.post(f"/api/cases/{case['id']}/review", json={"reviewer": "測試專員", "decision": "confirmed_damage", "notes": "確認為新增刮痕"})
    assert confirmed.status_code == 201, confirmed.text
    assert confirmed.json()["known_damage_records_created"] == 1
    history = client.get(f"/api/orders/{order['id']}/known-damages")
    assert history.status_code == 200
    assert len(history.json()["items"]) == 1

    next_order = create_order(client, "TEST-NEXT-RENTER", "LEDGER-001")
    blocked = client.post("/api/inspections", json={"order_id": next_order["id"], "inspection_type": "pickup", "location": "pytest 站"})
    assert blocked.status_code == 409
    assert blocked.json()["error"]["code"] == "known_damage_acknowledgement_required"
    acknowledged = client.post(f"/api/orders/{next_order['id']}/known-damages/acknowledge")
    assert acknowledged.status_code == 200
    allowed = client.post("/api/inspections", json={"order_id": next_order["id"], "inspection_type": "pickup", "location": "pytest 站"})
    assert allowed.status_code == 201

    detail_after = client.get(f"/api/cases/{case['id']}").json()
    assert len(detail_after["damage_evidence"]) == 1
    assert len(detail_after["known_damages"]) == 1

    review = client.post(f"/api/cases/{case['id']}/review", json={"reviewer": "測試專員", "decision": "approved", "notes": "測試確認"})
    assert review.status_code == 201


def test_blurred_upload_returns_retake_reason(client):
    order = create_order(client, "TEST-BLUR")
    session = create_session(client, order["id"], "pickup")
    response = client.post(
        f"/api/inspections/{session['id']}/images",
        data={"expected_view": "left_front"},
        files={"file": ("blur.jpg", image_bytes(900, blurred=True), "image/jpeg")},
    )
    assert response.status_code == 201
    quality = response.json()["quality"]
    assert quality["passed"] is False
    assert quality["retake_instruction"]


def test_failed_photo_does_not_poison_duplicate_check_for_another_view(client):
    order = create_order(client, "TEST-FAILED-DUPLICATE")
    session = create_session(client, order["id"], "pickup")
    content = image_bytes(901, blurred=True)
    first = client.post(
        f"/api/inspections/{session['id']}/images",
        data={"expected_view": "right_front"},
        files={"file": ("wrong-slot.jpg", content, "image/jpeg")},
    )
    assert first.status_code == 201
    assert first.json()["quality"]["passed"] is False

    second = client.post(
        f"/api/inspections/{session['id']}/images",
        data={"expected_view": "left_front"},
        files={"file": ("correct-slot.jpg", content, "image/jpeg")},
    )
    assert second.status_code == 201
    assert second.json()["quality"]["metrics"]["duplicate"] is False

def test_fake_image_is_rejected_and_not_kept(client):
    order = create_order(client, "TEST-FAKE-IMAGE")
    session = create_session(client, order["id"], "pickup")
    response = client.post(
        f"/api/inspections/{session['id']}/images",
        data={"expected_view": "left_front"},
        files={"file": ("fake.jpg", b"not actually an image", "image/jpeg")},
    )
    assert response.status_code == 422
    detail = client.get(f"/api/orders/{order['id']}").json()
    assert detail["inspections"][0]["images"] == []


def test_local_labeling_sync_update_image_and_export(client):
    dataset_root = Path("/tmp/carma-pytest/dataset")
    relative_path = Path("TST-1001/ORDER-1/01_left_front__sample.jpg")
    image_path = dataset_root / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes(1234))

    manifest_path = Path("/tmp/carma-pytest/dataset_manifest.csv")
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "image_type", "category"])
        writer.writeheader()
        writer.writerow({
            "source_path": relative_path.as_posix(),
            "image_type": "1",
            "category": "left_front",
        })

    synced = client.post("/api/labeling/sync")
    assert synced.status_code == 200, synced.text
    assert synced.json()["stats"]["total"] == 1

    listed = client.get("/api/labeling/items")
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 1
    item = listed.json()["items"][0]
    assert item["original_label"] == "left_front"

    image = client.get(item["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/")

    missing_reason = client.patch(f"/api/labeling/items/{item['id']}", json={
        "actual_label": None,
        "acceptable": False,
        "retake_reason": None,
        "review_status": "invalid",
    })
    assert missing_reason.status_code == 422

    updated = client.patch(f"/api/labeling/items/{item['id']}", json={
        "actual_label": "right_front",
        "acceptable": True,
        "retake_reason": None,
        "review_status": "relabelled",
        "notes": "左右預分類修正",
    })
    assert updated.status_code == 200, updated.text
    assert updated.json()["actual_label"] == "right_front"

    stats = client.get("/api/labeling/stats")
    corrected = client.get("/api/labeling/items?review_status=&category=right_front")
    assert corrected.status_code == 200
    assert corrected.json()["total"] == 1
    old_category = client.get("/api/labeling/items?review_status=&category=left_front")
    assert old_category.json()["total"] == 0

    assert stats.json()["reviewed"] == 1


def test_pickup_flow_requires_only_four_exterior_views(client):
    order = create_order(client, "TEST-PICKUP-FOUR")
    session = create_session(client, order["id"], "pickup")
    assert session["required_views"] == VIEWS[:4]

    for index, view in enumerate(VIEWS[:4]):
        response = client.post(
            f"/api/inspections/{session['id']}/images",
            data={"expected_view": view},
            files={"file": (f"{view}.jpg", image_bytes(1500 + index), "image/jpeg")},
        )
        assert response.status_code == 201, response.text
        assert response.json()["quality"]["passed"] is True

    analyzed = client.post(f"/api/inspections/{session['id']}/analyze")
    assert analyzed.status_code == 200, analyzed.text
