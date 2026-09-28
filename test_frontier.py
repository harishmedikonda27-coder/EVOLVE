import sys
import os
sys.path.append(os.path.dirname(__file__))

from environment.adaptive_frontier import AdaptiveFrontierWorld, get_level_spec

spec = get_level_spec(1, 42)
env = AdaptiveFrontierWorld(spec)

obs, info = env.reset(seed=42)
print("Obs shape:", obs.shape)
print("Info:", info)

obs, reward, is_done, _, info = env.step(1)
print("Step 1 obs shape:", obs.shape)
print("Reward:", reward)
print("Is Done:", is_done)

profile = env.get_task_profile()
print("Task Profile:", profile)
print("ALL TESTS PASSED")
