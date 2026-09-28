from enum import Enum
from typing import Set, Dict


class RunState(str, Enum):
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RunStateMachine:
    VALID_TRANSITIONS: Dict[RunState, Set[RunState]] = {
        RunState.CREATED: {RunState.INITIALIZING, RunState.FAILED},
        RunState.INITIALIZING: {RunState.RUNNING, RunState.FAILED},
        RunState.RUNNING: {RunState.PAUSED, RunState.COMPLETED, RunState.FAILED, RunState.STOPPING},
        RunState.PAUSED: {RunState.RUNNING, RunState.COMPLETED, RunState.FAILED, RunState.STOPPING},
        RunState.STOPPING: {RunState.STOPPED, RunState.FAILED},
        RunState.STOPPED: set(),
        RunState.COMPLETED: set(),
        RunState.FAILED: set(),
    }

    def __init__(self, initial_state: RunState = RunState.CREATED):
        self.current_state = initial_state

    def transition_to(self, new_state: RunState) -> RunState:
        if new_state not in self.VALID_TRANSITIONS[self.current_state]:
            raise ValueError(f"Invalid state transition from '{self.current_state}' to '{new_state}'.")
        self.current_state = new_state
        return self.current_state
