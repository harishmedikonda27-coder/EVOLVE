from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional


class TaskGenerationFailure(Exception):
    """Raised when the experiment loop's bounded retry limits are completely exhausted."""
    def __init__(self, level: int, requested_seed: int, attempts: int, spec_dict: Dict[str, Any]):
        self.level = level
        self.requested_seed = requested_seed
        self.attempts = attempts
        self.spec_dict = spec_dict
        super().__init__(f"Task generation failed after {attempts} attempts for level {level}, seed {requested_seed}")


class BaseEnvironment(ABC):
    def __init__(self, obs_dim: int = 16, action_dim: int = 4, max_steps: int = 100):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step_count = 0
        self.seed_val: Optional[int] = None

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def get_task_profile(self) -> Dict[str, Any]:
        """Return the environment configuration and task signature."""
        pass

    def is_solvable(self) -> bool:
        """
        Check if the generated environment is solvable.
        Default implementation assumes solvable, but complex generators must override.
        """
        return True

    def render_state(self) -> Dict[str, Any]:
        """Return a structured representation of the environment state."""
        return {}

