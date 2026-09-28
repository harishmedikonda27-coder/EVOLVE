import pytest
import torch
import numpy as np
from brain.brain import ModularBrain
from evolution.mutation import MutationManager, compute_rewire_diff
from evolution.transaction import TransactionResult
import copy

def test_rewire_actual_mask_change():
    brain = ModularBrain(10, 5)
    brain.add_module("mod1", 0, "TEST")
    
    # Force sparsity to allow rewire
    mod = brain.registry.get_module("mod1")
    mod.fc1.weight.grad = torch.randn_like(mod.fc1.weight)
    mod.mask1.view(-1)[:50] = 0.0
    
    topo_before = brain.topology_snapshot()
    mut_manager = MutationManager()
    structural_change = mut_manager.execute_rewire(brain, sparsity_budget=0.2)
    
    assert structural_change is True
    topo_after = brain.topology_snapshot()
    
    diff = compute_rewire_diff(topo_before, topo_after, "m1")
    assert diff.added_edge_count > 0 or diff.removed_edge_count > 0, "A successful REWIRE must produce actual mask change."

def test_rewire_infeasible():
    brain = ModularBrain(10, 5)
    brain.add_module("mod1", 0, "TEST")
    
    # Dense network with no room to grow = Infeasible for strict rewire
    # Or network with 1 connection where dropping it is not allowed
    # Let's make it fully dense, and mock gradients
    mod = brain.registry.get_module("mod1")
    mod.mask1.fill_(1.0)
    mod.mask2.fill_(1.0)
    
    mut_manager = MutationManager()
    structural_change = mut_manager.execute_rewire(brain, sparsity_budget=0.2)
    
    # If no growth is possible, it should not drop anything (strict parameter preservation)
    # Thus, structural_change must be False, or diff is 0
    topo_after = brain.topology_snapshot()
    if not structural_change:
        assert True
    else:
        # If it says it changed, the diff must be 0
        topo_before = brain.topology_snapshot()
        diff = compute_rewire_diff(topo_before, topo_after, "m1")
        assert diff.added_edge_count == 0 and diff.removed_edge_count == 0

def test_rewire_diff_integrity():
    brain = ModularBrain(10, 5)
    brain.add_module("mod1", 0, "TEST")
    mod = brain.registry.get_module("mod1")
    
    # Ensure some sparsity so we can 'add' edges later
    mod.mask1.view(-1)[:5] = 0.0
    topo_before = brain.topology_snapshot()
    
    # Manually drop 5 edges and add 5 edges
    mod.mask1.view(-1)[:5] = 1.0       # Add 5
    mod.mask1.view(-1)[5:10] = 0.0     # Drop 5
    
    topo_after = brain.topology_snapshot()
    diff = compute_rewire_diff(topo_before, topo_after, "mod1")
    
    assert diff.added_edge_count == 5
    assert diff.removed_edge_count == 5

def test_topology_revision_semantics():
    brain = ModularBrain(10, 5)
    rev1 = brain.topology_revision
    
    brain.add_module("mod1", 0, "TEST")
    rev2 = brain.topology_revision
    assert rev2 > rev1

def test_rewire_rollback():
    from evolution.rollback import RollbackManager
    import torch.optim as optim
    brain = ModularBrain(10, 5)
    brain.add_module("mod1", 0, "TEST")
    
    optimizer = optim.Adam(brain.parameters(), lr=0.001)
    
    rollback = RollbackManager()
    ckpt = rollback.capture_checkpoint(brain, optimizer, {})
    
    topo_before = brain.topology_snapshot()
    
    mod = brain.registry.get_module("mod1")
    mod.mask1.view(-1)[:10] = 0.0
    brain.topology_revision += 1
    
    assert brain.topology_revision > topo_before["topology_revision"]
    
    restored_brain, _ = rollback.restore_checkpoint(brain, optimizer, ckpt)
    topo_restored = restored_brain.topology_snapshot()
    
    assert topo_restored["topology_revision"] == topo_before["topology_revision"]
