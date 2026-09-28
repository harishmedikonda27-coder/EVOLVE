import pytest
import threading
import time
from backend.experiments.run import run_continuous_experiment, run_controls

def test_stop_mechanism():
    # Start the experiment in a separate thread so we can stop it
    results = {}
    
    def run_exp():
        try:
            res = run_continuous_experiment(
                run_id="test_stop_001",
                num_levels=1,
                seed=42,
                max_steps=5000,
                eval_freq=100
            )
            results.update(res)
        except Exception as e:
            results["error"] = str(e)

    from backend.experiments.run import run_controls, RunControl
    run_controls["test_stop_001"] = RunControl()
    
    t = threading.Thread(target=run_exp)
    t.start()
    
    # Wait a bit to ensure it enters the loop
    time.sleep(0.5)
    
    # Trigger stop
    run_controls["test_stop_001"].stop_requested.set()
    
    # Wait for thread to finish
    t.join(timeout=5.0)
    
    assert not t.is_alive(), "Experiment thread did not stop in time"
    assert "error" not in results, f"Experiment failed with error: {results.get('error')}"
    # The results should have task_outcomes if it completed or stopped normally
    assert "task_outcomes" in results
