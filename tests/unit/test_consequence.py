import pytest
from evolution.consequence import LongHorizonTracker
from evolution.memory import InterventionMemory
from evolution.state import AdaptationState

def test_long_horizon_tracker():
    memory = InterventionMemory(capacity=10)
    tracker = LongHorizonTracker(memory=memory, horizon_tasks=1)
    
    state = AdaptationState(
        current_performance=0.5,
        stagnation_slope=0.1,
        novelty=0.2,
        ood_shift=0.0,
        interference=0.1,
        plasticity_auc=0.5,
        utilization_mean=0.9,
        utilization_variance=0.1,
        module_count=2,
        parameter_count=1000,
        budget_remaining=3,
        youngest_module_age=500
    )
    
    # Register intervention at end of task 1
    inv_id = tracker.register_intervention(state, "GROW", 1000, "task_1")
    
    # Next state registered
    tracker.register_next_state(inv_id, state)
    
    # Check pending
    assert len(tracker.pending) == 1
    
    # Task 1 completes - shouldn't resolve yet since horizon_tasks=1 means 1 task *after* current
    resolved = tracker.on_task_complete("task_1", final_perf=0.6, plasticity_auc=0.5, retention=1.0)
    assert len(resolved) == 0
    assert len(tracker.pending) == 1
    assert tracker.pending[0].cumulative_meta_reward > 0
    
    # Task 2 completes - this should resolve it
    resolved2 = tracker.on_task_complete("task_2", final_perf=0.8, plasticity_auc=0.6, retention=0.9)
    assert len(resolved2) == 1
    assert len(tracker.pending) == 0
    
    # Check memory
    assert len(memory) == 1
