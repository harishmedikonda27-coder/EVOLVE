import pytest

def test_one_brain_per_run():
    from brain.brain import ModularBrain
    from evolution.mutation import MutationManager
    
    brain = ModularBrain(16, 5)
    brain.add_module("mod1", 0, "TEST")
    initial_id = id(brain)
    
    # 1. Training / Evaluation
    # (Since these just call agent.select_action / agent.update, they don't replace the brain object)
    assert id(brain) == initial_id
    
    # 2. ADAPT (weights change, object same)
    import torch.optim as optim
    optimizer = optim.Adam(brain.parameters(), lr=0.001)
    optimizer.step()
    assert id(brain) == initial_id
    
    # 3. GROW
    mut = MutationManager()
    mut.execute_grow(brain, 100)
    assert id(brain) == initial_id
    
    # 4. REWIRE
    mod = brain.registry.get_module("mod1")
    mod.mask1.view(-1)[:50] = 0.0 # Force sparsity
    mut.execute_rewire(brain)
    assert id(brain) == initial_id
    
    # 5. PRUNE
    brain.add_module("mod2", 100, "TEST")
    mut.execute_prune(brain, "mod2")
    assert id(brain) == initial_id
    
    # 6. Level transition
    # (Level loop uses the same brain object initialized at run start)
    assert id(brain) == initial_id
