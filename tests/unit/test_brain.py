import pytest
import torch
from brain.encoder import Encoder
from brain.module import EvolveModule
from brain.router import Router
from brain.head import OutputHead
from brain.registry import ModuleRegistry
from brain.brain import ModularBrain


def test_encoder_shape():
    encoder = Encoder(obs_dim=16, latent_dim=32)
    obs = torch.randn(4, 16)
    out = encoder(obs)
    assert out.shape == (4, 32)


def test_evolve_module_shape_and_metadata():
    mod = EvolveModule("module_A", input_dim=32, output_dim=64)
    x = torch.randn(4, 32)
    out = mod(x)
    assert out.shape == (4, 64)
    meta = mod.metadata()
    assert meta["module_id"] == "module_A"
    assert meta["parameter_count"] > 0


def test_router_softmax_sum():
    router = Router(latent_dim=32, num_modules=3)
    latent = torch.randn(5, 32)
    weights = router(latent)
    assert weights.shape == (5, 3)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(5), atol=1e-5)


def test_module_registry_operations():
    registry = ModuleRegistry()
    mod_a = EvolveModule("module_A", input_dim=32, output_dim=64)
    registry.add_module(mod_a)
    assert registry.list_active_ids() == ["module_A"]

    mod_b = EvolveModule("module_B", input_dim=32, output_dim=64)
    registry.add_module(mod_b)
    assert set(registry.list_active_ids()) == {"module_A", "module_B"}

    with pytest.raises(ValueError):
        # Reject duplicate ID
        registry.add_module(EvolveModule("module_A", input_dim=32, output_dim=64))


def test_modular_brain_forward_and_add_module():
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4)
    obs = torch.randn(2, 16)
    action_logits, routing_weights = brain(obs)
    assert action_logits.shape == (2, 4)
    assert routing_weights.shape == (2, 1)

    # Add module via GROW operation
    added_id = brain.add_module("module_B", created_step=100)
    assert added_id == "module_B"
    assert len(brain.registry.list_active_ids()) == 2

    # Forward after GROW
    action_logits2, routing_weights2 = brain(obs)
    assert action_logits2.shape == (2, 4)
    assert routing_weights2.shape == (2, 2)
