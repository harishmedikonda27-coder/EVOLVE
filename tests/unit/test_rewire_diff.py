import pytest
import torch
import numpy as np
from brain.brain import ModularBrain
from evolution.mutation import compute_rewire_diff, MutationManager

def test_rewire_with_actual_mask_changes():
    brain = ModularBrain(10, 5)
    mod_id = "test_mod"
    brain.add_module(mod_id, created_step=0, origin_event="TEST")
    mod = brain.registry.get_module(mod_id)
    mod.fc1.weight.grad = torch.randn_like(mod.fc1.weight)
    mod.fc2.weight.grad = torch.randn_like(mod.fc2.weight)
    mod.mask1.view(-1)[:50] = 0.0
    mod.mask2.view(-1)[:20] = 0.0
    
    topo_before = brain.topology_snapshot()
    mut_manager = MutationManager()
    mut_manager.execute_rewire(brain, sparsity_budget=0.2)
    topo_after = brain.topology_snapshot()
    diff = compute_rewire_diff(topo_before, topo_after, "mut_1")
    
    assert diff.added_edge_count > 0 or diff.removed_edge_count > 0

def test_no_op_rewire():
    brain = ModularBrain(10, 5)
    mod_id = "test_mod"
    brain.add_module(mod_id, created_step=0, origin_event="TEST")
    
    for m in brain.registry.list_active_ids():
        mod = brain.registry.get_module(m)
        mod.mask1.data.zero_()
        mod.mask2.data.zero_()
        
    topo_before = brain.topology_snapshot()
    rev_before = brain.topology_revision
    
    mut_manager = MutationManager()
    structural_change = mut_manager.execute_rewire(brain, sparsity_budget=0.0)
    
    topo_after = brain.topology_snapshot()
    rev_after = brain.topology_revision
    
    diff = compute_rewire_diff(topo_before, topo_after, "mut_1")
    assert diff.added_edge_count == 0
    assert diff.removed_edge_count == 0
    assert rev_before == rev_after
    assert not structural_change

def test_parameter_count_unchanged():
    brain = ModularBrain(10, 5)
    mod_id = "test_mod"
    brain.add_module(mod_id, created_step=0, origin_event="TEST")
    mod = brain.registry.get_module(mod_id)
    mod.fc1.weight.grad = torch.randn_like(mod.fc1.weight)
    mod.fc2.weight.grad = torch.randn_like(mod.fc2.weight)
    mod.mask1.view(-1)[:50] = 0.0
    mod.mask2.view(-1)[:20] = 0.0
    
    topo_before = brain.topology_snapshot()
    p_before = sum(p.numel() for p in brain.parameters())
    
    mut_manager = MutationManager()
    mut_manager.execute_rewire(brain, sparsity_budget=0.2)
    
    topo_after = brain.topology_snapshot()
    p_after = sum(p.numel() for p in brain.parameters())
    
    diff = compute_rewire_diff(topo_before, topo_after, "mut_1")
    assert p_before == p_after
    assert topo_before["total_parameter_count"] == topo_after["total_parameter_count"]

def test_module_count_unchanged():
    brain = ModularBrain(10, 5)
    mod_id = "test_mod"
    brain.add_module(mod_id, created_step=0, origin_event="TEST")
    mod = brain.registry.get_module(mod_id)
    mod.fc1.weight.grad = torch.randn_like(mod.fc1.weight)
    mod.fc2.weight.grad = torch.randn_like(mod.fc2.weight)
    mod.mask1.view(-1)[:50] = 0.0
    mod.mask2.view(-1)[:20] = 0.0
    
    topo_before = brain.topology_snapshot()
    
    mut_manager = MutationManager()
    mut_manager.execute_rewire(brain, sparsity_budget=0.2)
    
    topo_after = brain.topology_snapshot()
    
    diff = compute_rewire_diff(topo_before, topo_after, "mut_1")
    # Using ModularBrain creates the first module at init. Adding test_mod makes it 2 active.
    assert diff.module_count_before == 2
    assert diff.module_count_after == 2

def test_added_edge_diff_exactly():
    brain = ModularBrain(10, 5)
    mod_id = "test_mod"
    brain.add_module(mod_id, created_step=0, origin_event="TEST")
    mod = brain.registry.get_module(mod_id)
    mod.fc1.weight.grad = torch.randn_like(mod.fc1.weight)
    mod.mask1.view(-1)[:50] = 0.0
    mod.mask2.view(-1)[:20] = 0.0
    
    topo_before = brain.topology_snapshot()
    mut_manager = MutationManager()
    mut_manager.execute_rewire(brain, sparsity_budget=0.2)
    topo_after = brain.topology_snapshot()
    
    diff = compute_rewire_diff(topo_before, topo_after, "mut_1")
    assert diff.added_edge_count >= 0
    assert diff.added_edge_count == diff.after_edge_count - diff.unchanged_edge_count

def test_removed_edge_diff_exactly():
    brain = ModularBrain(10, 5)
    mod_id = "test_mod"
    brain.add_module(mod_id, created_step=0, origin_event="TEST")
    mod = brain.registry.get_module(mod_id)
    mod.fc1.weight.grad = torch.randn_like(mod.fc1.weight)
    mod.mask1.view(-1)[:50] = 0.0
    mod.mask2.view(-1)[:20] = 0.0
    
    topo_before = brain.topology_snapshot()
    mut_manager = MutationManager()
    mut_manager.execute_rewire(brain, sparsity_budget=0.2)
    topo_after = brain.topology_snapshot()
    
    diff = compute_rewire_diff(topo_before, topo_after, "mut_1")
    assert diff.removed_edge_count >= 0
    assert diff.removed_edge_count == diff.before_edge_count - diff.unchanged_edge_count

def test_rollback_restores_exact():
    brain = ModularBrain(10, 5)
    mod_id = "test_mod"
    brain.add_module(mod_id, created_step=0, origin_event="TEST")
    mod = brain.registry.get_module(mod_id)
    mod.fc1.weight.grad = torch.randn_like(mod.fc1.weight)
    mod.mask1.view(-1)[:50] = 0.0
    mod.mask2.view(-1)[:20] = 0.0
    
    import copy
    from evolution.rollback import RollbackManager
    optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)
    rm = RollbackManager()
    
    ckpt = rm.capture_checkpoint(brain, optimizer, {})
    
    topo_before = brain.topology_snapshot()
    
    mut_manager = MutationManager()
    mut_manager.execute_rewire(brain, sparsity_budget=0.2)
    
    brain, optimizer = rm.restore_checkpoint(brain, optimizer, ckpt)
    
    topo_rollback = brain.topology_snapshot()
    diff = compute_rewire_diff(topo_before, topo_rollback, "mut_2")
    
    assert diff.added_edge_count == 0
    assert diff.removed_edge_count == 0
    assert diff.before_edge_count == diff.after_edge_count
