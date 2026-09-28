from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from evolution.state import AdaptationState
from evolution.memory import InterventionMemory

@dataclass
class PendingIntervention:
    id: str
    state_at_decision: AdaptationState
    action: str
    decision_step: int
    task_id: str
    next_state: Optional[AdaptationState] = None
    cumulative_meta_reward: float = 0.0

class LongHorizonTracker:
    """
    Tracks delayed outcomes of architectural interventions.
    Ensures future outcomes never leak into decision state.
    Pending interventions remain explicitly pending until the horizon is reached.
    """
    def __init__(self, memory: InterventionMemory, horizon_tasks: int = 1):
        self.memory = memory
        # We evaluate the consequence after `horizon_tasks` have completed.
        self.horizon_tasks = horizon_tasks
        
        self.pending: List[PendingIntervention] = []
        self.completed_tasks_since_decision: Dict[str, int] = {}
        
    def register_intervention(self, state: AdaptationState, action: str, step: int, current_task: str) -> str:
        """Called exactly when a decision is made, before it is executed."""
        import copy
        intervention_id = f"inv_{step}_{action}"
        pending = PendingIntervention(
            id=intervention_id,
            state_at_decision=copy.deepcopy(state),
            action=action,
            decision_step=step,
            task_id=current_task
        )
        self.pending.append(pending)
        self.completed_tasks_since_decision[intervention_id] = 0
        return intervention_id
        
    def register_next_state(self, intervention_id: str, next_state: AdaptationState):
        """Called immediately after the transaction (accept or rollback) finishes."""
        import copy
        for p in self.pending:
            if p.id == intervention_id:
                p.next_state = copy.deepcopy(next_state)
                break
                
    def on_task_complete(self, task_id: str, final_perf: float, plasticity_auc: float, retention: float) -> List[Dict[str, Any]]:
        """
        Called when a task completes. Accrues delayed rewards.
        Resolves interventions that have reached their horizon.
        Returns a list of resolved interventions for event emitting.
        """
        # A simple causal meta-reward emphasizing long-term retention and future adaptability
        task_meta_reward = final_perf + (0.5 * retention) + (0.1 * plasticity_auc)
        
        to_remove = []
        resolved_info = []
        for p in self.pending:
            # Accumulate reward
            p.cumulative_meta_reward += task_meta_reward
            
            # If the completed task is not the one during which the decision was made, advance horizon counter
            if p.task_id != task_id:
                self.completed_tasks_since_decision[p.id] += 1
                
            # If we've reached the horizon, push to memory
            if self.completed_tasks_since_decision[p.id] >= self.horizon_tasks:
                if p.next_state is not None:
                    self.memory.push(
                        state=p.state_at_decision, 
                        action=p.action, 
                        consequence=p.cumulative_meta_reward, 
                        next_state=p.next_state
                    )
                to_remove.append(p)
                resolved_info.append({
                    "intervention_id": p.id,
                    "action": p.action,
                    "decision_step": p.decision_step,
                    "task_id_at_decision": p.task_id,
                    "cumulative_meta_reward": p.cumulative_meta_reward
                })
                
        for p in to_remove:
            self.pending.remove(p)
            if p.id in self.completed_tasks_since_decision:
                del self.completed_tasks_since_decision[p.id]
                
        return resolved_info
