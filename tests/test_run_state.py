import pytest
from backend.state import RunStateMachine, RunState

def test_valid_transitions():
    sm = RunStateMachine(RunState.CREATED)
    assert sm.current_state == RunState.CREATED
    sm.transition_to(RunState.INITIALIZING)
    assert sm.current_state == RunState.INITIALIZING
    sm.transition_to(RunState.RUNNING)
    assert sm.current_state == RunState.RUNNING
    sm.transition_to(RunState.STOPPING)
    assert sm.current_state == RunState.STOPPING
    sm.transition_to(RunState.STOPPED)
    assert sm.current_state == RunState.STOPPED

def test_invalid_transition_stopping_to_completed():
    sm = RunStateMachine(RunState.STOPPING)
    with pytest.raises(ValueError, match="Invalid state transition from 'RunState.STOPPING' to 'RunState.COMPLETED'"):
        sm.transition_to(RunState.COMPLETED)

def test_terminal_states_are_absorbing():
    sm = RunStateMachine(RunState.COMPLETED)
    with pytest.raises(ValueError):
        sm.transition_to(RunState.RUNNING)

    sm = RunStateMachine(RunState.FAILED)
    with pytest.raises(ValueError):
        sm.transition_to(RunState.STOPPING)

    sm = RunStateMachine(RunState.STOPPED)
    with pytest.raises(ValueError):
        sm.transition_to(RunState.COMPLETED)
