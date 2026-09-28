import pytest
from evolution.consequence import LongHorizonTracker, PendingIntervention
from evolution.memory import InterventionMemory
from evolution.state import AdaptationState

def test_long_horizon_no_temporal_leakage():
    memory = InterventionMemory(capacity=10)
    tracker = LongHorizonTracker(memory, horizon_tasks=1)
    
    # Create an initial state
    state = AdaptationState(
        current_performance=0.5,
        stagnation_slope=0.0,
        novelty=1.0,
        ood_shift=0.0,
        interference=0.0,
        plasticity_auc=0.0,
        utilization_mean=0.5,
        utilization_variance=0.1,
        module_count=2,
        parameter_count=1000,
        budget_remaining=5000,
        youngest_module_age=10
    )
    
    # Register intervention
    inv_id = tracker.register_intervention(state, "GROW", step=100, current_task="task1")
    
    import copy
    next_state = copy.deepcopy(state)
    tracker.register_next_state(inv_id, next_state)
    
    # Verify it is pending
    assert len(tracker.pending) == 1
    pending_inv = tracker.pending[0]
    assert pending_inv.id == inv_id
    
    # Now simulate future information corrupting the original state reference
    # Suppose the agent's performance drops to 0.1 in the future
    state.current_performance = 0.1
    
    # Ensure the tracker's state_before is unaffected (because of deepcopy)
    assert pending_inv.state_at_decision.current_performance == 0.5
    
    # Resolve the intervention by advancing horizon
    resolved = tracker.on_task_complete("task2", final_perf=0.8, plasticity_auc=1.0, retention=1.0)
    
    assert len(resolved) == 1
    assert len(tracker.pending) == 0
    assert len(memory.buffer) == 1
    
    # mem_entry is (state, action, reward, next_state, done)
    mem_entry = memory.buffer[0]
    assert mem_entry[0].current_performance == 0.5
