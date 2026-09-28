"""Integration test: Verify that GROW → Verify → Rollback restores exact model state,
and GROW → Verify → Accept produces a new topology."""
import pytest
import torch
import numpy as np
import copy
from brain.brain import ModularBrain
from evolution.controller import EvolutionController
from evolution.rollback import RollbackManager
from training.agent import DQNAgent


def test_grow_and_rollback_restores_exact_state():
    """After a rejected GROW, rollback must restore model weights, topology, and router dimensions."""
    torch.manual_seed(42)
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4, initial_module_id="module_A")
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)
    rollback = RollbackManager()

    # Capture the state before mutation
    checkpoint = rollback.capture_checkpoint(brain, optimizer, {"phase": "B"})

    # Record original state
    original_params = {k: v.clone() for k, v in brain.state_dict().items()}
    original_active_ids = brain.registry.list_active_ids().copy()
    original_param_count = brain.parameter_count()
    original_router_count = brain.router.num_modules

    # Perform GROW mutation
    brain.add_module("module_B", created_step=100, origin_event="GROW")
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)

    # Verify mutation actually changed things
    assert len(brain.registry.list_active_ids()) == 2
    assert brain.router.num_modules == 2
    assert brain.parameter_count() > original_param_count

    # Do some forward passes to change weights
    obs = torch.randn(4, 16)
    for _ in range(10):
        logits, _ = brain(obs)
        loss = logits.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Restore checkpoint
    rollback.restore_checkpoint(brain, optimizer, checkpoint)

    # Verify restoration
    assert brain.registry.list_active_ids() == original_active_ids
    assert brain.router.num_modules == original_router_count

    # Check that model weights match original
    restored_params = brain.state_dict()
    for key in original_params:
        if key in restored_params:
            assert torch.allclose(original_params[key], restored_params[key], atol=1e-6), \
                f"Parameter {key} not restored correctly"


def test_grow_and_accept_changes_topology():
    """After an accepted GROW, the brain should have a new module and updated router."""
    torch.manual_seed(42)
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4, initial_module_id="module_A")

    assert len(brain.registry.list_active_ids()) == 1
    assert brain.router.num_modules == 1

    # GROW
    brain.add_module("module_B", created_step=100, origin_event="GROW")
    assert len(brain.registry.list_active_ids()) == 2
    assert brain.router.num_modules == 2

    # Forward pass should still work with new topology
    obs = torch.randn(4, 16)
    logits, weights = brain(obs)
    assert logits.shape == (4, 4)
    assert weights.shape == (4, 2)

    # Topology snapshot should reflect the change
    snap = brain.topology_snapshot()
    assert snap["active_module_count"] == 2
    assert snap["total_module_count"] == 2


def test_multiple_grow_and_prune_cycle():
    """Test GROW -> GROW -> PRUNE cycle maintains consistency."""
    torch.manual_seed(42)
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4, initial_module_id="module_A")

    # Two GROWs
    brain.add_module("module_B", created_step=100)
    brain.add_module("module_C", created_step=200)
    assert len(brain.registry.list_active_ids()) == 3

    # PRUNE one
    brain.remove_module("module_B")
    active = brain.registry.list_active_ids()
    assert "module_B" not in active
    assert len(active) == 2

    # Forward pass should work
    obs = torch.randn(2, 16)
    logits, weights = brain(obs)
    assert logits.shape == (2, 4)
    assert weights.shape == (2, 2)


def test_controller_end_to_end_decision():
    """Test the full controller pipeline: signals -> need score -> decision."""
    controller = EvolutionController()

    # Simulate a high-need scenario (large performance gap, stagnation, high novelty)
    decision = controller.evaluate_step(
        current_performance=0.1,
        target_performance=1.0,
        recent_history=[0.1, 0.1, 0.1, 0.1, 0.1],
        task_profile_distance=0.8,
        interference_value=0.3,
        module_utilizations=[0.9],
        current_param_count=5000,
        max_param_budget=50000,
        current_step=500,
        active_module_count=1,
    )

    assert decision.action in ["GROW", "ADAPT", "REUSE", "PRUNE"]
    assert 0.0 <= decision.need_score <= 1.0
    assert decision.budget_remaining >= 0
