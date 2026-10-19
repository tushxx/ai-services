from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_create_incident_and_get():
    payload = {
        "title": "Test VPN incident",
        "description": "Test users cannot connect to corporate VPN.",
        "category": "Network",
        "service": "Corporate VPN",
        "impact": "High",
        "urgency": "High",
        "affected_users": 25,
        "service_criticality": "Business_Critical",
        "reported_source": "Service Portal",
    }

    response = client.post("/api/v1/incidents", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["incident_id"].startswith("INC-")
    assert body["assignment_group"] == "Network Operations"
    assert body["predicted_priority"] in {"P1", "P2", "P3", "P4"}

    get_response = client.get(f"/api/v1/incidents/{body['incident_id']}")
    assert get_response.status_code == 200
    assert get_response.json()["title"] == payload["title"]


def test_update_and_close_incident():
    payload = {
        "title": "Test close incident",
        "description": "Testing lifecycle.",
        "category": "Application",
        "service": "Email Service",
        "impact": "Medium",
        "urgency": "Medium",
        "affected_users": 5,
        "service_criticality": "Business_Critical",
        "reported_source": "Email",
    }

    created = client.post("/api/v1/incidents", json=payload).json()
    incident_id = created["incident_id"]

    update = client.put(
        f"/api/v1/incidents/{incident_id}",
        json={"final_priority": "P2", "status": "IN_PROGRESS"},
    )
    assert update.status_code == 200
    assert update.json()["final_priority"] == "P2"

    close = client.patch(f"/api/v1/incidents/{incident_id}/close")
    assert close.status_code == 200
    assert close.json()["status"] == "CLOSED"


def test_analytics():
    response = client.get("/api/v1/analytics/summary")
    assert response.status_code == 200
    body = response.json()
    assert "total_incidents" in body
    assert "by_priority" in body
    assert "by_category" in body
