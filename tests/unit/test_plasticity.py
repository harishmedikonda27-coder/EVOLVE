import torch
import pytest
import numpy as np
from brain.brain import ModularBrain
from training.agent import DQNAgent
from evolution.rollback import RollbackManager
from evolution.plasticity import PlasticityProbeEngine

def test_plasticity_probe_restores_state():
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4)
    agent = DQNAgent(brain)
    rollback = RollbackManager()
    
    # Store initial state
    initial_params = [p.clone() for p in brain.parameters()]
    
    # Populate buffer
    obs = np.zeros(16)
    agent.buffer.push(obs, 0, 1.0, obs, False)
    initial_buffer_size = len(agent.buffer)
    
    # Run probe
    engine = PlasticityProbeEngine(budget=50, eval_freq=10)
    metrics = engine.run_probe(agent, rollback)
    
    # Check parameters were restored
    for p_init, p_curr in zip(initial_params, brain.parameters()):
        assert torch.equal(p_init, p_curr)
        
    # Check buffer was restored
    assert len(agent.buffer) == initial_buffer_size
    
    # Check metrics structure
    assert isinstance(metrics.initial_perf, float)
    assert isinstance(metrics.final_perf, float)
    assert isinstance(metrics.improvement, float)
    assert isinstance(metrics.auc, float)
    assert isinstance(metrics.slope, float)
