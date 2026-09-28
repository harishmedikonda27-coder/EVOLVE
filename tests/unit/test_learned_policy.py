import torch
import pytest
from evolution.state import AdaptationState
from evolution.memory import InterventionMemory
from evolution.learned_policy import LearnedPolicy, FeasibilityMask
from training.meta_agent import MetaTrainer

def test_feasibility_mask():
    masker = FeasibilityMask(max_modules=8, min_modules=1)
    
    # Plenty of budget, normal modules
    mask = masker.get_mask(module_count=4, budget_remaining=100)
    assert mask.sum() == 5 # All feasible
    
    # No budget
    mask = masker.get_mask(module_count=4, budget_remaining=0)
    assert mask[2] == 0 # GROW
    assert mask[3] == 0 # PRUNE
    assert mask[4] == 0 # REWIRE
    
    # Max modules
    mask = masker.get_mask(module_count=8, budget_remaining=100)
    assert mask[2] == 0 # GROW

def test_learned_policy_selects_action():
    policy = LearnedPolicy(state_dim=12, action_dim=5)
    
    # Create dummy state
    state = AdaptationState(
        current_performance=0.5,
        stagnation_slope=0.01,
        novelty=0.1,
        ood_shift=0.0,
        interference=0.0,
        plasticity_auc=0.5,
        utilization_mean=0.8,
        utilization_variance=0.1,
        module_count=2,
        parameter_count=1000,
        budget_remaining=100,
        youngest_module_age=10
    )
    vec = torch.tensor(state.to_vector(), dtype=torch.float32)
    
    action = policy.select_action(vec, module_count=2, budget_remaining=100, epsilon=0.0)
    assert action in ["REUSE", "ADAPT", "GROW", "PRUNE", "REWIRE"]

def test_meta_trainer_updates():
    policy = LearnedPolicy(state_dim=12, action_dim=5)
    trainer = MetaTrainer(policy)
    memory = InterventionMemory(capacity=100)
    
    state = AdaptationState(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 100, 100, 0)
    
    # Populate memory
    for _ in range(40):
        memory.push(state, "GROW", 1.0, state)
        
    initial_loss = trainer.update(memory, batch_size=32)
    assert initial_loss > 0.0
