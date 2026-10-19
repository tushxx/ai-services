import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

Q2_DIR = Path(__file__).resolve().parents[1] / "q2"
for mod in ("database", "models", "schemas", "main"):
    sys.modules.pop(mod, None)
sys.path.insert(0, str(Q2_DIR))

from database import Base, get_db  # noqa: E402
from main import app  # noqa: E402

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=True, bind=engine)


STORE = {"name": "Fresh Corner", "city": "Pune"}
ITEM = {
    "name": "Organic Tomatoes",
    "category": "produce",
    "stock_qty": 40,
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
    store = client.post("/stores", json=STORE).json()
    item = client.post(f"/stores/{store['id']}/items", json=ITEM).json()
    return store, item


def test_create_store(client):
    response = client.post("/stores", json=STORE)
    assert response.status_code == 201
    assert response.json() == {
        "id": 1,
        "name": "Fresh Corner",
        "city": "Pune",
    }


def test_add_item(client):
    store_id = client.post("/stores", json=STORE).json()["id"]
    response = client.post(f"/stores/{store_id}/items", json=ITEM)
    assert response.status_code == 201
    assert response.json() == {
        "id": 1,
        "name": "Organic Tomatoes",
        "category": "produce",
        "stock_qty": 40,
    }


def test_get_store_nests_items(seeded, client):
    store, _ = seeded
    response = client.get(f"/stores/{store['id']}")
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "name": "Fresh Corner",
        "city": "Pune",
        "items": [
            {
                "id": 1,
                "name": "Organic Tomatoes",
                "category": "produce",
                "stock_qty": 40,
            }
        ],
    }


def test_get_store_filter_and_sort(seeded, client):
    store, _ = seeded
    client.post(
        f"/stores/{store['id']}/items",
        json={"name": "Soap", "category": "household", "stock_qty": 10},
    )
    client.post(
        f"/stores/{store['id']}/items",
        json={"name": "Spinach", "category": "produce", "stock_qty": 5},
    )

    response = client.get(
        f"/stores/{store['id']}",
        params={"category": "produce", "sort": "stock_qty"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["name"] == "Fresh Corner"
    assert body["city"] == "Pune"
    assert [i["category"] for i in body["items"]] == ["produce", "produce"]
    assert [i["stock_qty"] for i in body["items"]] == [5, 40]


def test_delete_item(seeded, client):
    store, item = seeded
    client.post(
        f"/stores/{store['id']}/items",
        json={"name": "Milk", "category": "dairy", "stock_qty": 12},
    )

    response = client.delete(f"/stores/{store['id']}/items/{item['id']}")
    assert response.status_code == 204
    assert response.content == b""

    body = client.get(f"/stores/{store['id']}").json()
    assert [i["id"] for i in body["items"]] == [2]
    assert body["name"] == "Fresh Corner"


def test_get_missing_store_returns_404(client):
    response = client.get("/stores/999")
    assert response.status_code == 404


def test_delete_item_wrong_store_returns_404(seeded, client):
    _, item = seeded
    other = client.post(
        "/stores", json={"name": "City Mart", "city": "Mumbai"}
    ).json()
    response = client.delete(f"/stores/{other['id']}/items/{item['id']}")
    assert response.status_code == 404


def test_create_store_empty_name_returns_422(client):
    response = client.post("/stores", json={"name": "", "city": "Pune"})
    assert response.status_code == 422


def test_invalid_sort_returns_422(seeded, client):
    store, _ = seeded
    response = client.get(f"/stores/{store['id']}", params={"sort": "city"})
    assert response.status_code == 422