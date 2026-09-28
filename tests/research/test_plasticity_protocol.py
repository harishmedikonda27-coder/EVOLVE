import pytest
import torch
import numpy as np
import random
import copy
from training.agent import DQNAgent
from brain.brain import ModularBrain
from evolution.plasticity import PlasticityProbeEngine
from evolution.rollback import RollbackManager

def setup_deterministic_agent(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4)
    agent = DQNAgent(brain=brain)
    return agent

def test_plasticity_zero_leakage():
    # Run 1: No probe
    agent1 = setup_deterministic_agent(42)
    
    # Take a few steps to advance state
    obs = np.random.rand(16).astype(np.float32)
    action1, _ = agent1.select_action(obs, evaluate=False)
    
    # Run 2: With probe
    agent2 = setup_deterministic_agent(42)
    action2_pre, _ = agent2.select_action(obs, evaluate=False)
    assert action1 == action2_pre # Pre-condition
    
    # Execute probe
    probe = PlasticityProbeEngine(budget=5, eval_freq=2, probe_level=1)
    rollback = RollbackManager()
    metrics = probe.run_probe(agent2, rollback)
    
    # Now take another step with identical observation
    obs2 = np.ones(16).astype(np.float32)
    action1_post, _ = agent1.select_action(obs2, evaluate=True)
    action2_post, _ = agent2.select_action(obs2, evaluate=True)
    
    assert action1_post == action2_post, "Action diverged after plasticity probe!"
    
    # Verify weights are identical
    for p1, p2 in zip(agent1.brain.parameters(), agent2.brain.parameters()):
        assert torch.equal(p1, p2)
