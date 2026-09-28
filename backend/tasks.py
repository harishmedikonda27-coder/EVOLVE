from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json


@dataclass
class TaskProfile:
    task_id: str
    family: str
    config: Dict[str, Any]
    seed: int
    level: int
    best_performance: float = 0.0
    final_performance: float = 0.0
    checkpoint_id: Optional[str] = None
    topology_hash: Optional[str] = None
    eval_history: Dict[int, float] = None  # Mapping from global_step -> performance

    def __post_init__(self):
        if self.eval_history is None:
            self.eval_history = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "config": self.config,
            "seed": self.seed,
            "level": self.level,
            "best_performance": self.best_performance,
            "final_performance": self.final_performance,
            "checkpoint_id": self.checkpoint_id,
            "topology_hash": self.topology_hash,
            "eval_history": self.eval_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskProfile":
        return cls(**data)


class TaskRegistry:
    def __init__(self):
        self.tasks: Dict[str, TaskProfile] = {}

    def register_task(self, task_id: str, family: str, config: Dict[str, Any], seed: int, level: int):
        self.tasks[task_id] = TaskProfile(
            task_id=task_id,
            family=family,
            config=config,
            seed=seed,
            level=level
        )

    def update_performance(self, task_id: str, performance: float, global_step: Optional[int] = None):
        if task_id in self.tasks:
            self.tasks[task_id].final_performance = performance
            if performance > self.tasks[task_id].best_performance:
                self.tasks[task_id].best_performance = performance
            if global_step is not None:
                self.tasks[task_id].eval_history[global_step] = performance

    def save_checkpoint_info(self, task_id: str, checkpoint_id: str, topology_hash: str):
        if task_id in self.tasks:
            self.tasks[task_id].checkpoint_id = checkpoint_id
            self.tasks[task_id].topology_hash = topology_hash

    def get_task(self, task_id: str) -> Optional[TaskProfile]:
        return self.tasks.get(task_id)

    def calculate_interference(self, before_scores: Dict[str, float], after_scores: Dict[str, float]) -> float:
        """
        Calculate actual degradation on previous tasks.
        I_k = max(0, R_before_k - R_after_k)
        Returns average interference over evaluated old tasks.
        """
        interferences = []
        for t_id, before_perf in before_scores.items():
            if t_id in after_scores:
                after_perf = after_scores[t_id]
                forgetting = before_perf - after_perf
                # Clamp to 0. We don't reward positive transfer as negative interference in this metric.
                i_val = max(0.0, forgetting)
                interferences.append(i_val)

        if not interferences:
            return 0.0

        return sum(interferences) / len(interferences)

    def calculate_forgetting(self, task_id: str) -> float:
        """
        Calculate explicit forgetting F_i = max_{t >= i} A[i,t] - A[i,N]
        """
        task = self.tasks.get(task_id)
        if not task or not task.eval_history:
            return 0.0
            
        max_perf = max(task.eval_history.values())
        final_perf = task.final_performance
        return max(0.0, max_perf - final_perf)
        
    def calculate_retention(self, current_step: int) -> float:
        """
        Average retention across all previous tasks.
        Retention = sum(current_perf) / N for past tasks.
        """
        if len(self.tasks) <= 1:
            return 1.0 # No past tasks to retain
            
        retentions = []
        # Exclude the very latest task which is currently training
        task_ids = list(self.tasks.keys())[:-1]
        for t_id in task_ids:
            task = self.tasks[t_id]
            # Use final_performance which is updated constantly, or specific step if available
            retentions.append(task.final_performance)
            
        if not retentions:
            return 1.0
            
        return sum(retentions) / len(retentions)

    def calculate_forward_learning(self, continual_perf: float, fresh_perf: float) -> float:
        """
        FL_i = A_i^{continual}(B) - A_i^{fresh}(B)
        """
        return continual_perf - fresh_perf

    def calculate_novelty(self, current_family: str, current_config: Dict[str, Any]) -> float:
        """
        Calculate task profile distance from previously learned tasks.
        """
        if not self.tasks:
            return 1.0 # First task is entirely novel

        same_family = [t for t in self.tasks.values() if t.family == current_family]
        if not same_family:
            return 1.0
        
        return 0.5
