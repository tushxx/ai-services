import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_prediction_endpoint():
    payload = {
        "category": "Network",
        "service": "Corporate VPN",
        "impact": "High",
        "urgency": "High",
        "affected_users": 180,
        "service_criticality": "Business_Critical",
        "reported_source": "Service Portal",
    }
    response = client.post("/api/v1/predictions/priority", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_priority"] in {"P1", "P2", "P3", "P4"}
    assert 0 <= body["confidence"] <= 1
    assert set(body["probabilities"]).issubset({"P1", "P2", "P3", "P4"})


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
