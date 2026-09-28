import requests
import time


def main():
    base_url = "http://127.0.0.1:8000"
    print("=== EVOLVE Member 4 Backend Mock Run Demo ===")

    # 1. Health check
    res = requests.get(f"{base_url}/health")
    print("Health check response:", res.json())

    # 2. Create run
    req = {
        "mode": "demo",
        "seed": 42,
        "experiment_type": "evolve",
        "task_sequence": ["A", "B", "A"]
    }
    res = requests.post(f"{base_url}/runs", json=req)
    run_data = res.json()
    run_id = run_data["run_id"]
    print(f"Created run: {run_id}, initial state: {run_data['state']}")

    # 3. Start run
    res = requests.post(f"{base_url}/runs/{run_id}/start")
    print("Start run response:", res.json())

    # 4. Get run details
    res = requests.get(f"{base_url}/runs/{run_id}")
    print("Fetched run details:", res.json())

    print("\nMock run demo completed successfully!")


if __name__ == "__main__":
    main()
