import pytest
from environment.adaptive_frontier.world import AdaptiveFrontierWorld
from environment.adaptive_frontier.difficulty import get_level_spec

def test_adaptive_frontier_reset_and_step():
    spec = get_level_spec(1, seed=42)
    env = AdaptiveFrontierWorld(spec)
    obs, info = env.reset(seed=42)
    assert obs.shape == (16,)
    assert "agent_pos" in env.get_state()

    next_obs, reward, terminated, truncated, step_info = env.step(action=1)
    assert next_obs.shape == (16,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

def test_adaptive_frontier_profile():
    spec = get_level_spec(5, seed=42)
    env = AdaptiveFrontierWorld(spec)
    obs, info = env.reset(seed=42)
    
    profile = env.get_task_profile()
    assert "reward_spec" in profile
    assert "name" in profile
    assert "AdaptiveFrontier" in profile["name"]
