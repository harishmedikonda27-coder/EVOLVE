import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AdaptationState:
    """
    Complete state representation for the structural evolution policy.
    Contains strictly causal information available at decision time.
    """
    # PERFORMANCE
    current_performance: float
    stagnation_slope: float
    
    # TASK / ENVIRONMENT
    novelty: float
    ood_shift: float
    
    # CONTINUAL LEARNING
    interference: float
    plasticity_auc: float
    
    # STRUCTURE
    utilization_mean: float
    utilization_variance: float
    module_count: int
    parameter_count: int
    
    # EVOLUTION HISTORY
    budget_remaining: int
    youngest_module_age: int
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.current_performance,
            self.stagnation_slope,
            self.novelty,
            self.ood_shift,
            self.interference,
            self.plasticity_auc,
            self.utilization_mean,
            self.utilization_variance,
            float(self.module_count),
            float(self.parameter_count),
            float(self.budget_remaining),
            float(self.youngest_module_age)
        ], dtype=np.float32)

    @classmethod
    def state_dim(cls) -> int:
        """Returns the flat dimensionality of the state vector."""
        return 12
        
    def to_dict(self) -> Dict[str, float]:
        return {
            "current_performance": self.current_performance,
            "stagnation_slope": self.stagnation_slope,
            "novelty": self.novelty,
            "ood_shift": self.ood_shift,
            "interference": self.interference,
            "plasticity_auc": self.plasticity_auc,
            "utilization_mean": self.utilization_mean,
            "utilization_variance": self.utilization_variance,
            "module_count": float(self.module_count),
            "parameter_count": float(self.parameter_count),
            "budget_remaining": float(self.budget_remaining),
            "youngest_module_age": float(self.youngest_module_age)
        }
