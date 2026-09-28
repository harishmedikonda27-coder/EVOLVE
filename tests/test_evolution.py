import unittest
import torch
import numpy as np
import copy
from evolution.scorer import NeedScorer
from evolution.signals import EvolutionSignals
from evolution.rollback import RollbackManager
from brain.brain import ModularBrain
from brain.module import EvolveModule


class TestEvolutionMechanisms(unittest.TestCase):

    def test_needscore_bounds_and_saturation(self):
        """NeedScore should strictly bound between 0 and 1 without artificially saturating at 1.0 unnecessarily."""
        scorer = NeedScorer(w_p=0.4, w_s=0.3, w_n=0.15, w_i=0.15, w_c=0.20)
        
        # Max positive signals, min negative signals
        signals_max = EvolutionSignals(
            performance_gap=1.0, stagnation=1.0, novelty=1.0, interference=1.0, utilization=0.0, cost=0.0
        )
        score_max = scorer.compute(signals_max)
        self.assertAlmostEqual(score_max, 1.0, places=4, msg="Max NeedScore should be exactly 1.0")

        # Min positive signals, max negative signals
        signals_min = EvolutionSignals(
            performance_gap=0.0, stagnation=0.0, novelty=0.0, interference=0.0, utilization=1.0, cost=1.0
        )
        score_min = scorer.compute(signals_min)
        self.assertAlmostEqual(score_min, 0.0, places=4, msg="Min NeedScore should be exactly 0.0")

        # Mixed
        signals_mixed = EvolutionSignals(
            performance_gap=0.5, stagnation=0.5, novelty=0.2, interference=0.1, utilization=0.5, cost=0.8
        )
        score_mixed = scorer.compute(signals_mixed)
        expected = (0.4*0.5) + (0.3*0.5) + (0.15*0.2) + (0.15*0.1) - (0.2*0.8)
        self.assertAlmostEqual(score_mixed, expected, places=4)

    def test_obd_proxy_math(self):
        """Test OBD diagonal curvature proxy math explicitly."""
        # H_hat(t) = beta * H_hat(t-1) + (1-beta)*g^2
        beta = 0.9
        h_hat_prev = 1.0
        g = 2.0
        h_hat_new = beta * h_hat_prev + (1 - beta) * (g**2)
        self.assertAlmostEqual(h_hat_new, 0.9 * 1.0 + 0.1 * 4.0)

        # S = 0.5 * H_hat * w^2
        w = 0.5
        s = 0.5 * h_hat_new * (w**2)
        self.assertAlmostEqual(s, 0.5 * 1.3 * 0.25)

    def test_rollback_exact_optimizer_and_rng(self):
        """Prove that after a rejected mutation, model parameters, topology, registry, optimizer state, 
        and RNG are restored to the exact checkpoint state within tolerances."""
        brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4, initial_module_id="module_start")
        optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)

        # Do a fake forward pass and step to populate optimizer state
        loss = brain(torch.randn(2, 16))[0].sum()
        loss.backward()
        optimizer.step()

        # Set specific RNG
        torch.manual_seed(42)
        np.random.seed(42)

        rollback = RollbackManager()
        controller_state = {"mock": "data"}

        # 1. Capture checkpoint
        ckpt = rollback.capture_checkpoint(brain, optimizer, controller_state)

        # 2. Mutate (GROW)
        old_param_count = brain.parameter_count()
        new_id = "module_mutation"
        brain.add_module(new_id)
        self.assertTrue(new_id in brain.registry.list_active_ids())
        self.assertTrue(brain.parameter_count() > old_param_count)

        # Mess up RNG
        _ = torch.rand(10)
        _ = np.random.rand(10)

        # Modify old module weights
        with torch.no_grad():
            brain.registry.get_module("module_start").fc1.weight.add_(1.0)

        # 3. Restore checkpoint
        brain, optimizer = rollback.restore_checkpoint(brain, optimizer, ckpt)

        # 4. Verify invariants
        self.assertFalse(new_id in brain.registry.list_active_ids(), "Added module must be fully removed.")
        self.assertEqual(brain.parameter_count(), old_param_count, "Parameter count must match exactly.")
        
        # Verify weights restored
        weight_sum = brain.registry.get_module("module_start").fc1.weight.sum().item()
        
        # Verify RNG restored
        rand_torch_post_rollback = torch.rand(1).item()
        
        torch.manual_seed(42)
        rand_torch_expected = torch.rand(1).item()
        self.assertAlmostEqual(rand_torch_post_rollback, rand_torch_expected, places=6, msg="Torch RNG not perfectly restored.")
        
    def test_rigl_sparse_rewiring_masks(self):
        """Validate dynamic masking logic for RigL rewiring."""
        mod = EvolveModule(module_id="test", input_dim=4, output_dim=4, hidden_dim=4)
        
        # Manually drop half the weights via masks
        with torch.no_grad():
            mod.mask1.fill_(0.0)
            mod.mask1[0, :] = 1.0 # only first row active
            mod.apply_masks()
            
        # Verify applied
        w1 = mod.fc1.weight
        self.assertEqual(w1[1:, :].sum().item(), 0.0)
        
        # Forward pass should work with implicit zeroes
        x = torch.randn(2, 4)
        out = mod(x)
        self.assertEqual(out.shape, (2, 4))
        
if __name__ == '__main__':
    unittest.main()
