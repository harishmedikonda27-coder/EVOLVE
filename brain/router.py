import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, latent_dim: int, num_modules: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modules = num_modules
        self.gate = nn.Linear(latent_dim, num_modules)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        logits = self.gate(latent)
        weights = F.softmax(logits, dim=-1)
        return weights

    def update_module_count(self, new_count: int):
        """Expands or rebuilds the routing linear layer for new module count while preserving existing weights.
        
        When growing, the bias of new gate outputs is initialised to +1.0 so the
        softmax assigns a meaningful routing probability to the freshly added
        module.  Without this, the random-near-zero initialisation causes the
        new module to receive almost no gradient during the verification budget,
        producing the characteristic gain=0.0 rollback cycle.
        """
        if new_count == self.num_modules:
            return
        
        old_gate = self.gate
        device = old_gate.weight.device
        new_gate = nn.Linear(self.latent_dim, new_count).to(device)
        
        with torch.no_grad():
            # Zero-init new weights/biases first so they're clean
            nn.init.zeros_(new_gate.weight)
            nn.init.zeros_(new_gate.bias)

            # Copy over existing weights/biases
            min_count = min(self.num_modules, new_count)
            new_gate.weight[:min_count, :] = old_gate.weight[:min_count, :]
            new_gate.bias[:min_count] = old_gate.bias[:min_count]

            # Give new modules a routing bias boost so softmax assigns them
            # a meaningful initial probability (≈ uniform with existing modules)
            if new_count > self.num_modules:
                existing_mean_bias = old_gate.bias.mean().item()
                boost = max(existing_mean_bias + 1.0, 1.0)
                new_gate.bias[self.num_modules:new_count] = boost
                # Small-magnitude Xavier-like init for new weights
                nn.init.xavier_uniform_(new_gate.weight[self.num_modules:new_count, :])

        self.gate = new_gate
        self.num_modules = new_count
