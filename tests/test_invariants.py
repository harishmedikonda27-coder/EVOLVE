"""
Test suite for EVOLVE structural invariants.

Aggressively mutates the ModularBrain and verifies that invariants hold
after every operation (GROW, PRUNE, REWIRE, rollback).
"""

import pytest
import torch
import copy

from brain.brain import ModularBrain
from training.agent import DQNAgent
from evolution.mutation import MutationManager
from evolution.rollback import RollbackManager
from backend.invariants import validate_brain_state, InvariantError


@pytest.fixture
def brain():
    return ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4)


@pytest.fixture
def agent(brain):
    return DQNAgent(brain=brain)


class TestInvariantsAfterInit:
    def test_fresh_brain_passes(self, brain, agent):
        validate_brain_state(brain, agent.optimizer, label="fresh")

    def test_fresh_brain_has_one_module(self, brain):
        assert len(brain.registry.list_active_ids()) == 1

    def test_router_matches_modules(self, brain):
        assert brain.router.num_modules == len(brain.registry.list_active_ids())
        assert brain.router.gate.out_features == len(brain.registry.list_active_ids())


class TestInvariantsAfterGrow:
    def test_grow_preserves_invariants(self, brain, agent):
        mut = MutationManager()
        new_id = mut.execute_grow(brain, step=100)
        agent.rebuild_optimizer()
        validate_brain_state(brain, agent.optimizer, label="post-grow")

    def test_multiple_grows(self, brain, agent):
        mut = MutationManager()
        for i in range(5):
            mut.execute_grow(brain, step=100 + i)
            agent.rebuild_optimizer()
            validate_brain_state(brain, agent.optimizer, label=f"grow-{i}")

        assert len(brain.registry.list_active_ids()) == 6  # 1 initial + 5 grown
        assert brain.router.num_modules == 6

    def test_forward_pass_after_grow(self, brain, agent):
        mut = MutationManager()
        mut.execute_grow(brain, step=100)
        agent.rebuild_optimizer()
        
        x = torch.randn(4, 16)
        logits, weights = brain(x)
        assert logits.shape == (4, 4)
        assert weights.shape == (4, 2)  # 2 modules now

    def test_new_module_gets_routing_weight(self, brain):
        """The zero-gain fix: new module should get a meaningful routing probability."""
        mut = MutationManager()
        mut.execute_grow(brain, step=100)
        
        x = torch.randn(8, 16)
        with torch.no_grad():
            _, weights = brain(x)
        
        # The new module (index 1) should have a non-trivial mean weight
        new_module_weight = weights[:, 1].mean().item()
        assert new_module_weight > 0.05, (
            f"New module routing weight is {new_module_weight:.4f}, "
            f"expected > 0.05 due to bias boost."
        )


class TestInvariantsAfterPrune:
    def test_prune_preserves_invariants(self, brain, agent):
        mut = MutationManager()
        new_id = mut.execute_grow(brain, step=100)
        agent.rebuild_optimizer()
        
        # Now prune the new module
        mut.execute_prune(brain, new_id)
        agent.rebuild_optimizer()
        validate_brain_state(brain, agent.optimizer, label="post-prune")

    def test_cannot_prune_last_module(self, brain):
        with pytest.raises(RuntimeError, match="Cannot prune the last active module"):
            brain.remove_module(brain.registry.list_active_ids()[0])


class TestInvariantsAfterRewire:
    def test_rewire_preserves_invariants(self, brain, agent):
        mut = MutationManager()
        mut.execute_rewire(brain)
        validate_brain_state(brain, agent.optimizer, label="post-rewire")

    def test_rewire_preserves_mask_shape(self, brain):
        mut = MutationManager()
        mut.execute_rewire(brain)
        
        for mod_id in brain.registry.list_active_ids():
            mod = brain.registry.get_module(mod_id)
            assert mod.mask1.shape == mod.fc1.weight.shape
            assert mod.mask2.shape == mod.fc2.weight.shape


class TestInvariantsAfterRollback:
    def test_rollback_restores_invariants(self, brain, agent):
        rollback = RollbackManager()
        mut = MutationManager()
        
        # Capture pre-mutation state
        ckpt = rollback.capture_checkpoint(brain, agent.optimizer, {})
        initial_param_count = brain.parameter_count()
        initial_active = len(brain.registry.list_active_ids())
        
        # Mutate
        mut.execute_grow(brain, step=100)
        agent.rebuild_optimizer()
        assert len(brain.registry.list_active_ids()) == initial_active + 1
        
        # Rollback
        brain, agent.optimizer = rollback.restore_checkpoint(brain, agent.optimizer, ckpt)
        agent.target_brain = copy.deepcopy(brain)
        agent.target_brain.eval()
        
        validate_brain_state(brain, agent.optimizer, label="post-rollback")
        assert len(brain.registry.list_active_ids()) == initial_active
        assert brain.parameter_count() == initial_param_count

    def test_double_rollback(self, brain, agent):
        """Test that rollback works correctly even if called twice."""
        rollback = RollbackManager()
        mut = MutationManager()
        
        ckpt = rollback.capture_checkpoint(brain, agent.optimizer, {})
        
        # Grow + rollback twice
        for _ in range(2):
            mut.execute_grow(brain, step=100)
            agent.rebuild_optimizer()
            brain, agent.optimizer = rollback.restore_checkpoint(brain, agent.optimizer, ckpt)
            agent.target_brain = copy.deepcopy(brain)
            agent.target_brain.eval()
            validate_brain_state(brain, agent.optimizer, label="double-rollback")


class TestInvariantDetection:
    """Test that the invariant checker actually catches violations."""

    def test_detects_router_mismatch(self, brain):
        # Artificially break the router
        brain.router.num_modules = 999
        with pytest.raises(InvariantError, match="Router num_modules"):
            validate_brain_state(brain, label="broken-router")

    def test_detects_no_active_modules(self, brain):
        # Mark all modules as pruned
        for mod_id in list(brain.registry.modules_dict.keys()):
            brain.registry.modules_dict[mod_id].status = "pruned"
        with pytest.raises(InvariantError, match="No active modules"):
            validate_brain_state(brain, label="no-active")


class TestConvergenceTracker:
    """Smoke tests for the ConvergenceTracker integration."""

    def test_converged_detection(self):
        from training.convergence import ConvergenceTracker, ConvergenceStatus
        tracker = ConvergenceTracker(target_score=0.8, window_size=5, min_evaluations=3)
        history = [0.85, 0.88, 0.90, 0.92, 0.91]
        result = tracker.check(history, current_step=500)
        assert result.status == ConvergenceStatus.CONVERGED

    def test_improving_detection(self):
        from training.convergence import ConvergenceTracker, ConvergenceStatus
        tracker = ConvergenceTracker(target_score=0.8, window_size=5, min_evaluations=3)
        history = [0.1, 0.3, 0.5]
        result = tracker.check(history, current_step=500)
        assert result.status == ConvergenceStatus.IMPROVING

    def test_exhausted_detection(self):
        from training.convergence import ConvergenceTracker, ConvergenceStatus
        tracker = ConvergenceTracker(target_score=0.8, window_size=5, min_evaluations=3, max_steps=1000)
        history = [0.3, 0.3, 0.3, 0.3, 0.3]
        result = tracker.check(history, current_step=1000)
        assert result.status == ConvergenceStatus.EXHAUSTED

    def test_stagnating_detection(self):
        from training.convergence import ConvergenceTracker, ConvergenceStatus
        tracker = ConvergenceTracker(target_score=0.8, window_size=5, min_evaluations=3, max_steps=20000)
        history = [0.50, 0.50, 0.50, 0.50, 0.50]
        result = tracker.check(history, current_step=5000)
        assert result.status == ConvergenceStatus.STAGNATING
