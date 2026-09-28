import requests
import websocket
import json
import time

def on_message(ws, message):
    data = json.loads(message)
    print(f"[WS] {data['event_type']} -> phase={data.get('phase')} step={data.get('step')}")
    if data['event_type'] in ['RUN_COMPLETED', 'ERROR']:
        print("Closing WS on end state")
        ws.close()

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### WebSocket Closed ###")

def on_open(ws):
    print("### WebSocket Opened ###")
    # Tell REST to start
    run_id = ws.run_id
    print(f"Starting run {run_id} via REST...")
    res = requests.post(f"http://127.0.0.1:8000/runs/{run_id}/start")
    print(res.json())

import pytest

@pytest.mark.skip(reason="Requires live server")
def test_integration():
    print("Creating run...")
    res = requests.post("http://127.0.0.1:8000/runs", json={
        "mode": "research",
        "seed": 42,
        "experiment_type": "evolve",
        "families": ["HierarchicalNavigation"],
        "levels": 1
    })
    run_id = res.json()["run_id"]
    print(f"Created {run_id}")
    
    ws_url = f"ws://127.0.0.1:8000/runs/{run_id}/stream"
    ws = websocket.WebSocketApp(ws_url,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.run_id = run_id
    ws.run_forever()

if __name__ == "__main__":
    test_integration()
