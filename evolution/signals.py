from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class EvolutionSignals:
    performance_gap: float
    stagnation: float
    novelty: float
    interference: float
    utilization: Optional[float]
    cost: float

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "performance_gap": float(np.clip(self.performance_gap, 0.0, 1.0)),
            "stagnation": float(np.clip(self.stagnation, 0.0, 1.0)),
            "novelty": float(np.clip(self.novelty, 0.0, 1.0)),
            "interference": float(np.clip(self.interference, 0.0, 1.0)),
            "utilization": float(np.clip(self.utilization, 0.0, 1.0)) if self.utilization is not None else None,
            "cost": float(np.clip(self.cost, 0.0, 1.0)),
            "raw_performance_gap": self.performance_gap,
            "raw_stagnation": self.stagnation,
            "raw_novelty": self.novelty,
            "raw_interference": self.interference,
            "raw_utilization": self.utilization,
            "raw_cost": self.cost
        }


class SignalCalculator:
    def compute(
        self,
        current_performance: float,
        target_performance: float = 1.0,
        recent_history: Optional[List[float]] = None,
        task_profile_distance: float = 0.0,
        interference_value: float = 0.0,
        module_utilizations: Optional[List[float]] = None,
        current_param_count: int = 1000,
        max_param_budget: int = 10000,
    ) -> EvolutionSignals:
        # 1. Performance Gap P: scale by target
        p_gap = (target_performance - current_performance) / max(abs(target_performance), 1e-5)

        # 2. Stagnation S
        s_val = 0.0
        if recent_history and len(recent_history) >= 3:
            # simple slope via finite difference
            slope = (recent_history[-1] - recent_history[0]) / float(len(recent_history))
            # if slope is small/negative, stagnation is high
            if slope <= 1e-3:
                s_val = 1.0
            else:
                s_val = max(0.0, 1.0 - (slope * 10))

        # 3. Novelty N
        n_val = task_profile_distance

        # 4. Interference I
        # Use directly provided interference (average degradation)
        # Assuming typical reward scale is around ~1.0. 
        i_val = interference_value

        # 5. Utilization U
        if module_utilizations and len(module_utilizations) > 1:
            u_val = float(np.mean(module_utilizations))
        else:
            u_val = None

        # 6. Cost C
        c_val = current_param_count / float(max(max_param_budget, 1))

        return EvolutionSignals(
            performance_gap=p_gap,
            stagnation=s_val,
            novelty=n_val,
            interference=i_val,
            utilization=u_val,
            cost=c_val,
        )

