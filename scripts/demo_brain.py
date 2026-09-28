import torch
from brain.brain import ModularBrain


def main():
    print("=== EVOLVE Member 1 Standalone Brain Demo ===")
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4)
    print("Initial topology snapshot:")
    print(brain.topology_snapshot())

    # Forward pass
    dummy_obs = torch.randn(1, 16)
    action_logits, routing_weights = brain(dummy_obs)
    print(f"\nForward output shape: {action_logits.shape}, Routing weights: {routing_weights.tolist()}")

    # Trigger GROW
    print("\n--- Triggering GROW mutation ---")
    new_id = brain.add_module("module_B", created_step=100, origin_event="GROW")
    print(f"Added module: {new_id}")
    print("Updated topology snapshot:")
    print(brain.topology_snapshot())

    # Forward pass 2
    action_logits2, routing_weights2 = brain(dummy_obs)
    print(f"Post-GROW forward output shape: {action_logits2.shape}, Routing weights: {routing_weights2.tolist()}")

    print("\nStandalone Brain demo completed successfully!")


if __name__ == "__main__":
    main()
