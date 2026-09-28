import pytest
import time
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_stop_e2e_normal_training():
    # Start a run
    res = client.post("/runs", json={
        "type": "continuous",
        "families": ["HierarchicalNavigation"],
        "levels": 1,
        "seed": 42
    })
    assert res.status_code == 200
    run_id = res.json()["run_id"]
    
    # Start it
    res = client.post(f"/runs/{run_id}/start")
    assert res.status_code == 200
    
    # Wait for RUNNING
    time.sleep(1.0)
    res = client.get(f"/runs/{run_id}")
    state = res.json()["state"]
    assert state == "RUNNING"
    
    # Issue stop
    res = client.post(f"/runs/{run_id}/stop")
    assert res.status_code == 200
    
    # Wait for STOPPED to settle
    max_wait = 50
    state = "STOPPING"
    for _ in range(max_wait):
        time.sleep(0.1)
        res = client.get(f"/runs/{run_id}")
        state = res.json()["state"]
        if state == "STOPPED":
            break
            
    assert state == "STOPPED"
    
    # Verify events
    res = client.get(f"/runs/{run_id}/events")
    events = res.json()
    types = [e["event_type"] for e in events]
    assert "RUN_STOPPED" in types
    assert "EXPERIMENT_FAILED" not in types
    assert "EXPERIMENT_COMPLETED" not in types
