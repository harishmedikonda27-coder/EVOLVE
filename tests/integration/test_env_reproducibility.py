import pytest
import numpy as np
import torch
from backend.experiments.run import create_environment

def test_env_reproducibility():
    family_configs = [
        ("HierarchicalNavigation", 1),
        ("MemoryObservation", 1),
        ("AdaptiveDynamics", 1)
    ]
    
    seed = 42
    
    for family, level in family_configs:
        # Create first environment and extract state
        env1 = create_environment(family, level, seed)
        obs1, _ = env1.reset(seed=seed)
        
        state1 = env1.get_task_profile()
        action_sample1 = 1
        next_obs1, r1, t1, tr1, i1 = env1.step(action_sample1)
        
        # Create second environment and extract state
        env2 = create_environment(family, level, seed)
        obs2, _ = env2.reset(seed=seed)
        
        state2 = env2.get_task_profile()
        
        action_sample2 = 1
        next_obs2, r2, t2, tr2, i2 = env2.step(action_sample2)
        
        # Assertions
        assert np.array_equal(obs1, obs2)
        assert np.array_equal(next_obs1, next_obs2)
        assert r1 == r2
        assert t1 == t2
        assert tr1 == tr2
        assert state1 == state2
        
        # Also ensure inner state elements match (e.g., target positions)
        if "goal_pos" in state1:
            assert state1["goal_pos"] == state2["goal_pos"]
