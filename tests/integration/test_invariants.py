import pytest
import torch
import numpy as np
from brain.brain import ModularBrain
from evolution.rollback import RollbackManager

def test_checkpoint_invariants():
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4, initial_module_id="start")
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)
    
    # modify state
    loss = brain(torch.randn(2, 16))[0].sum()
    loss.backward()
    optimizer.step()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    rollback = RollbackManager()
    controller_state = {"budget": 10}
    
    expected_rand = torch.rand(1).item()
    
    ckpt = rollback.capture_checkpoint(brain, optimizer, controller_state)
    
    # Mutate
    param_count_before = brain.parameter_count()
    brain.add_module("new_module")
    
    _ = torch.rand(10)
    
    brain, optimizer = rollback.restore_checkpoint(brain, optimizer, ckpt)
    
    # check invariants
    assert "new_module" not in brain.registry.list_active_ids()
    assert brain.parameter_count() == param_count_before
    
    # The RNG should be restored to exactly what it was when capture_checkpoint was called
    # Let's see what the next number would have been
    rand_after_restore = torch.rand(1).item()
    
    torch.manual_seed(42)
    torch.rand(1) # expected_rand
    expected_rand_after_capture = torch.rand(1).item()
    
    assert rand_after_restore == expected_rand_after_capture, "RNG state not perfectly restored"
