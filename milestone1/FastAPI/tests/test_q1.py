import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

Q1_DIR = Path(__file__).resolve().parents[1] / "q1"
for mod in ("database", "models", "schemas", "main"):
    sys.modules.pop(mod, None)
sys.path.insert(0, str(Q1_DIR))

from database import Base, get_db  # noqa: E402
from main import app  # noqa: E402

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=True, bind=engine)


DEPOT = {"name": "Central Depot", "region": "Karnataka"}
VEHICLE = {
    "plate_number": "KA-05-AB-1234",
    "vehicle_type": "truck",
    "mileage": 12000,
}


@pytest.fixture
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def seeded(client):
    depot = client.post("/depots", json=DEPOT).json()
    vehicle = client.post(f"/depots/{depot['id']}/vehicles", json=VEHICLE).json()
    return depot, vehicle


def test_create_depot(client):
    response = client.post("/depots", json=DEPOT)
    assert response.status_code == 201
    assert response.json() == {
        "id": 1,
        "name": "Central Depot",
        "region": "Karnataka",
    }


def test_add_vehicle(client):
    depot_id = client.post("/depots", json=DEPOT).json()["id"]
    response = client.post(f"/depots/{depot_id}/vehicles", json=VEHICLE)
    assert response.status_code == 201
    assert response.json() == {
        "id": 1,
        "plate_number": "KA-05-AB-1234",
        "vehicle_type": "truck",
        "mileage": 12000,
    }


def test_get_depot_nests_vehicles(seeded, client):
    depot, _ = seeded
    response = client.get(f"/depots/{depot['id']}")
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "name": "Central Depot",
        "region": "Karnataka",
        "vehicles": [
            {
                "id": 1,
                "plate_number": "KA-05-AB-1234",
                "vehicle_type": "truck",
                "mileage": 12000,
            }
        ],
    }


def test_get_depot_filter_and_sort(seeded, client):
    depot, _ = seeded
    client.post(
        f"/depots/{depot['id']}/vehicles",
        json={
            "plate_number": "KA-05-CD-9999",
            "vehicle_type": "van",
            "mileage": 3000,
        },
    )
    client.post(
        f"/depots/{depot['id']}/vehicles",
        json={
            "plate_number": "KA-05-EF-0001",
            "vehicle_type": "truck",
            "mileage": 500,
        },
    )

    response = client.get(
        f"/depots/{depot['id']}",
        params={"vehicle_type": "truck", "sort": "mileage"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["name"] == "Central Depot"
    assert body["region"] == "Karnataka"
    assert [v["vehicle_type"] for v in body["vehicles"]] == ["truck", "truck"]
    assert [v["mileage"] for v in body["vehicles"]] == [500, 12000]


def test_update_vehicle(seeded, client):
    depot, vehicle = seeded
    response = client.put(
        f"/depots/{depot['id']}/vehicles/{vehicle['id']}",
        json={
            "plate_number": "KA-05-AB-1234",
            "vehicle_type": "truck",
            "mileage": 18500,
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "plate_number": "KA-05-AB-1234",
        "vehicle_type": "truck",
        "mileage": 18500,
    }


def test_get_missing_depot_returns_404(client):
    response = client.get("/depots/999")
    assert response.status_code == 404


def test_update_vehicle_wrong_depot_returns_404(seeded, client):
    _, vehicle = seeded
    other = client.post(
        "/depots", json={"name": "South Depot", "region": "Karnataka"}
    ).json()
    response = client.put(
        f"/depots/{other['id']}/vehicles/{vehicle['id']}",
        json={
            "plate_number": "KA-05-AB-1234",
            "vehicle_type": "truck",
            "mileage": 18500,
        },
    )
    assert response.status_code == 404


def test_create_depot_empty_name_returns_422(client):
    response = client.post(
        "/depots", json={"name": "", "region": "Karnataka"}
    )
    assert response.status_code == 422


def test_invalid_sort_returns_422(seeded, client):
    depot, _ = seeded
    response = client.get(f"/depots/{depot['id']}", params={"sort": "region"})
    assert response.status_code == 422