from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class MetricPoint:
    step: int
    episode: int
    phase: str                           # A, B, A_eval
    reward: float
    success: bool
    performance_gap: float
    stagnation: float
    novelty: float
    interference: float
    utilization: float
    cost: float
    need_score: float
    decision: str
    active_module_count: int
    parameter_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "episode": self.episode,
            "phase": self.phase,
            "reward": self.reward,
            "success": self.success,
            "performance_gap": self.performance_gap,
            "stagnation": self.stagnation,
            "novelty": self.novelty,
            "interference": self.interference,
            "utilization": self.utilization,
            "cost": self.cost,
            "need_score": self.need_score,
            "decision": self.decision,
            "active_module_count": self.active_module_count,
            "parameter_count": self.parameter_count,
        }


class MetricsCollector:
    def __init__(self):
        self.history: List[MetricPoint] = []

    def record(self, point: MetricPoint):
        self.history.append(point)

    def compute_summary(self, a_before: float, a_after: float, b_after: float) -> Dict[str, Any]:
        retention_ratio = a_after / max(a_before, 1e-5)
        forgetting = a_before - a_after

        recent_rewards = [p.reward for p in self.history[-50:]] if self.history else [0.0]
        mean_reward = float(np.mean(recent_rewards))

        return {
            "a_before": a_before,
            "a_after": a_after,
            "b_after": b_after,
            "retention_ratio": float(np.clip(retention_ratio, 0.0, 5.0)),
            "forgetting": float(forgetting),
            "mean_recent_reward": mean_reward,
            "total_logged_steps": len(self.history),
        }
