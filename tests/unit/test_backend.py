import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_create_and_get_run():
    payload = {
        "mode": "research",
        "seed": 42,
        "experiment_type": "evolve",
        "task_sequence": ["A", "B", "A"]
    }
    res = client.post("/runs", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert data["state"] == "CREATED"

    run_id = data["run_id"]
    res_get = client.get(f"/runs/{run_id}")
    assert res_get.status_code == 200
    assert res_get.json()["run_id"] == run_id


def test_run_lifecycle_start_stop():
    res = client.post("/runs", json={"mode": "research", "seed": 42})
    run_id = res.json()["run_id"]

    start_res = client.post(f"/runs/{run_id}/start")
    assert start_res.status_code == 200
    assert start_res.json()["state"] == "RUNNING"

    pause_res = client.post(f"/runs/{run_id}/stop")
    assert pause_res.status_code == 200
    assert pause_res.json()["state"] in ("STOPPING", "STOPPED")

def test_world_preview_endpoint():
    response = client.get("/world/preview")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    
    assert "agent_pos" in data
    assert "target_pos" in data
    assert "grid_size" in data
