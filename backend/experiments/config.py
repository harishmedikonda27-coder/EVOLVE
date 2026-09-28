import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class BaselineConfig:
    enabled: bool = True
    condition_name: str = "FIXED" # FIXED, RANDOM_VALID_ACTION, DETERMINISTIC_EVOLVE, LEARNED_EVOLVE
    adaptation_mode: str = "none" # none, deterministic, learned
    random_valid_action: bool = False

@dataclass
class ExperimentConfig:
    experiment_id: str
    seeds: List[int]
    task_sequence: List[Dict[str, Any]] # List of LevelConfig dicts
    baselines: List[BaselineConfig]
    
    # Budgets
    max_steps_per_task: int = 20000
    eval_freq: int = 500
    plasticity_probe_budget: int = 500
    plasticity_threshold_active: float = 0.5
    plasticity_threshold_degraded: float = 0.1
    
    # Limits
    max_param_budget: int = 50000
    
    # Consequence tracking
    consequence_horizon: int = 1
    
    # Output
    output_dir: str = "storage/experiments"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "seeds": self.seeds,
            "task_sequence": self.task_sequence,
            "baselines": [{"condition": b.condition_name, "mode": b.adaptation_mode, "random": b.random_valid_action} for b in self.baselines],
            "max_steps_per_task": self.max_steps_per_task,
            "eval_freq": self.eval_freq,
            "plasticity_probe_budget": self.plasticity_probe_budget,
            "plasticity_threshold_active": self.plasticity_threshold_active,
            "plasticity_threshold_degraded": self.plasticity_threshold_degraded,
            "max_param_budget": self.max_param_budget,
            "consequence_horizon": self.consequence_horizon,
            "output_dir": self.output_dir
        }
        
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        baselines = []
        for b in data.get("baselines", []):
            baselines.append(BaselineConfig(
                condition_name=b.get("condition"),
                adaptation_mode=b.get("mode"),
                random_valid_action=b.get("random", False)
            ))
            
        data["baselines"] = baselines
        return cls(**data)
