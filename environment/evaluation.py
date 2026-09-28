import torch
import numpy as np
from typing import Dict, Any, List
from environment.base import BaseEnvironment


class IsolatedEvaluator:
    """Evaluates a brain's performance on an environment using fixed seeds for reproducibility.
    
    This evaluator uses deterministic seeds to ensure fair comparison between
    different model configurations at different points in training.
    """
    def __init__(self, num_episodes: int = 10, seed: int = 999):
        self.num_episodes = num_episodes
        self.seed = seed

    def evaluate(self, brain: torch.nn.Module, env: BaseEnvironment) -> Dict[str, Any]:
        brain.eval()
        rewards: List[float] = []
        successes: List[bool] = []
        episode_lengths: List[int] = []

        with torch.no_grad():
            for ep in range(self.num_episodes):
                ep_seed = self.seed + ep
                obs, _ = env.reset(seed=ep_seed)
                done = False
                total_reward = 0.0
                steps = 0
                episode_success = False

                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action_logits, _ = brain(obs_t)
                    action = int(torch.argmax(action_logits, dim=-1).item())

                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated

                    if terminated and info.get("success"):
                        episode_success = True

                successes.append(episode_success)
                rewards.append(total_reward)
                episode_lengths.append(steps)

        brain.train()
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        success_rate = float(np.mean([1.0 if s else 0.0 for s in successes])) if successes else 0.0
        mean_steps = float(np.mean(episode_lengths)) if episode_lengths else 0.0

        return {
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "mean_steps": mean_steps,
            "episodes_evaluated": self.num_episodes,
        }
