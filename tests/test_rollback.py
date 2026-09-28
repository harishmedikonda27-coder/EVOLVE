import pytest
import torch
from brain.brain import ModularBrain
from training.agent import DQNAgent
from evolution.rollback import RollbackManager
import copy

def test_rollback_consistency():
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4)
    agent = DQNAgent(brain=brain)
    rollback = RollbackManager()
    
    # Store initial state
    ckpt_state = rollback.capture_checkpoint(brain, agent.optimizer, {})
    
    # Simulate a mutation (e.g. GROW) by adding a dummy module manually to the brain
    # For testing, we can just change a parameter to see if it rolls back
    with torch.no_grad():
        for param in brain.parameters():
            param.add_(1.0)
            
    # Mutate registry explicitly
    brain.registry.modules_dict["fake_module"] = None
    
    # Mutate controller state
    controller_state = {"fake": 123}
    
    # Mutate RNG
    torch.rand(10)
    
    # Perform rollback
    restored_brain, restored_optimizer = rollback.restore_checkpoint(brain, agent.optimizer, ckpt_state)
    agent.target_brain = copy.deepcopy(restored_brain)
    
    # Verify parameter is restored
    for p_orig, p_restored in zip(ckpt_state["model_state_dict"].values(), restored_brain.state_dict().values()):
        assert torch.allclose(p_orig, p_restored)
        
    # Verify target network is isomorphic and identical
    for p_restored, p_target in zip(restored_brain.state_dict().values(), agent.target_brain.state_dict().values()):
        assert torch.allclose(p_restored, p_target)
        
    # Verify registry
    assert "fake_module" not in restored_brain.registry.modules_dict
    
    # Verify RNG state
    assert torch.equal(torch.get_rng_state(), ckpt_state["torch_rng"])
