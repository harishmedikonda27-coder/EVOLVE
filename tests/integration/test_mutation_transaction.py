import pytest
import torch
import copy
from brain.brain import ModularBrain
from training.agent import DQNAgent
from evolution.rollback import RollbackManager
from evolution.mutation import MutationManager

def test_mutation_transaction_rollback():
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4)
    agent = DQNAgent(brain=brain)
    rollback = RollbackManager()
    mut_manager = MutationManager()
    
    # 1. Capture checkpoint
    ckpt_state = rollback.capture_checkpoint(brain, agent.optimizer, {"test": 1})
    
    # 2. Mutate (GROW)
    mut_manager.execute_grow(brain, step=100)
    agent.rebuild_optimizer()
    
    # Assert mutation happened
    assert len(brain.registry.list_active_ids()) == 2
    assert len(agent.target_brain.registry.list_active_ids()) == 2
    # Actually rebuild_optimizer does:
    # self.target_brain = copy.deepcopy(self.brain)
    assert len(agent.target_brain.registry.list_active_ids()) == 2
    
    # 3. Simulate rejection/exception
    # Perform rollback
    restored_brain, restored_optimizer = rollback.restore_checkpoint(brain, agent.optimizer, ckpt_state)
    agent.target_brain = copy.deepcopy(restored_brain)
    agent.target_brain.eval()
    
    # Assert exact rollback
    assert len(restored_brain.registry.list_active_ids()) == 1
    assert len(agent.target_brain.registry.list_active_ids()) == 1
    
    # Compare target and online strictly
    for p1, p2 in zip(restored_brain.state_dict().values(), agent.target_brain.state_dict().values()):
        assert torch.allclose(p1, p2)
        
    for p1, p2 in zip(ckpt_state["model_state_dict"].values(), restored_brain.state_dict().values()):
        assert torch.allclose(p1, p2)
