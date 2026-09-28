import torch
import numpy as np
from typing import Dict, Any

from brain.brain import ModularBrain
from environment.scenarios import EnvironmentA, EnvironmentB
from environment.evaluation import IsolatedEvaluator
from training.agent import DQNAgent


def run_always_grow_experiment(steps_a: int = 3000, steps_b: int = 3000, seed: int = 42) -> Dict[str, Any]:
    print(f"[Always-Grow Baseline] Starting A -> B -> A run (seed={seed})...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_a = EnvironmentA()
    env_b = EnvironmentB()
    evaluator = IsolatedEvaluator(num_episodes=10, seed=seed)

    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4, initial_module_id="module_A")
    agent = DQNAgent(brain, learning_rate=0.001)

    # 1. Train on Environment A
    obs, _ = env_a.reset(seed=seed)
    for step in range(steps_a):
        action, _ = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env_a.step(action)
        agent.buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env_a.reset()
        agent.update()

    eval_a_before = evaluator.evaluate(brain, env_a)
    print(f"[Always-Grow Baseline] Pre-B Environment A Score: {eval_a_before['mean_reward']:.3f}")

    # Naive growth trigger at task shift
    print("[Always-Grow Baseline] Naive task shift trigger: adding module_B")
    brain.add_module("module_B", created_step=steps_a, origin_event="task_shift_trigger")
    agent.rebuild_optimizer()

    # 2. Train on Environment B
    obs, _ = env_b.reset(seed=seed + 1)
    for step in range(steps_b):
        action, _ = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env_b.step(action)
        agent.buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env_b.reset()
        agent.update()

    eval_b_after = evaluator.evaluate(brain, env_b)
    eval_a_after = evaluator.evaluate(brain, env_a)
    print(f"[Always-Grow Baseline] Post-B Environment B Score: {eval_b_after['mean_reward']:.3f}")
    print(f"[Always-Grow Baseline] Post-B Environment A Retention Score: {eval_a_after['mean_reward']:.3f}")

    retention_ratio = eval_a_after["mean_reward"] / max(eval_a_before["mean_reward"], 1e-5)
    return {
        "experiment": "always_grow",
        "a_before": eval_a_before["mean_reward"],
        "b_after": eval_b_after["mean_reward"],
        "a_after": eval_a_after["mean_reward"],
        "a_success_before": eval_a_before["success_rate"],
        "b_success_after": eval_b_after["success_rate"],
        "a_success_after": eval_a_after["success_rate"],
        "retention_ratio": retention_ratio,
        "parameter_count": brain.parameter_count(),
    }


if __name__ == "__main__":
    run_always_grow_experiment()
