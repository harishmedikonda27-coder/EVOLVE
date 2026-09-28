import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from environment.base import BaseEnvironment, LevelGenerationError
from environment.families import HierarchicalNavigation, MemoryObservation, AdaptiveDynamics

@dataclass
class LevelConfig:
    level_id: str
    task_variant: str  # "Navigation", "Memory", "Dynamics"
    difficulty_level: int # 1 to 50
    seed: int
    ood_shift: bool = False
    shift_magnitude: float = 0.0

class UnifiedEvolveWorld(BaseEnvironment):
    """
    One unified progressive environment abstraction.
    Internally uses existing mechanics as controlled dimensions of task difficulty.
    """
    def __init__(self, config: LevelConfig, max_steps: int = 300):
        # We instantiate the underlying mechanic based on the task variant
        self.config = config
        self.sub_env = self._instantiate_sub_env(config)
        super().__init__(
            obs_dim=self.sub_env.obs_dim,
            action_dim=self.sub_env.action_dim,
            max_steps=self.sub_env.max_steps
        )
        self.seed_val = config.seed

    def _instantiate_sub_env(self, config: LevelConfig) -> BaseEnvironment:
        if config.task_variant == "Navigation":
            env = HierarchicalNavigation(level=config.difficulty_level)
        elif config.task_variant == "Memory":
            env = MemoryObservation(level=config.difficulty_level)
        elif config.task_variant == "Dynamics":
            env = AdaptiveDynamics(level=config.difficulty_level)
        else:
            raise ValueError(f"Unknown task variant {config.task_variant}")
            
        # Apply OOD Shift if required
        if config.ood_shift:
            # Example explicit shifts
            if config.task_variant == "Navigation":
                # Suddenly increase obstacles ignoring standard scaling
                env.num_obstacles = int(env.num_obstacles * (1.0 + config.shift_magnitude))
            elif config.task_variant == "Dynamics":
                # Suddenly spike slip probability
                env.slip_prob = min(1.0, env.slip_prob + config.shift_magnitude)
            elif config.task_variant == "Memory":
                # Immediately drop cue duration to bare minimum
                env.cue_duration = max(1, int(env.cue_duration - (config.shift_magnitude * 10)))
                
        return env

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed_val = seed
        
        # Sub-environments handle level generation validity internally.
        # HierarchicalNavigation uses 100 retry budget and raises LevelGenerationError.
        try:
            obs, info = self.sub_env.reset(seed=self.seed_val)
        except Exception as e:
            raise LevelGenerationError(f"Failed to generate valid UnifiedEvolveWorld level: {e}")
            
        self.current_step_count = self.sub_env.current_step_count
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, r, term, trunc, info = self.sub_env.step(action)
        self.current_step_count = self.sub_env.current_step_count
        return obs, r, term, trunc, info

    def get_task_profile(self) -> Dict[str, Any]:
        prof = self.sub_env.get_task_profile()
        # Overlay Unified properties
        prof["unified_level_id"] = self.config.level_id
        prof["ood_shift_active"] = self.config.ood_shift
        prof["shift_magnitude"] = self.config.shift_magnitude
        prof["base_family"] = prof["family"]
        prof["family"] = "UnifiedEvolveWorld"
        return prof

    def is_solvable(self) -> bool:
        return self.sub_env.is_solvable()

    def render_state(self) -> Dict[str, Any]:
        return self.sub_env.render_state()
