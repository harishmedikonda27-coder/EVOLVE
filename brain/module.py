import torch
import torch.nn as nn
from typing import Dict, Any


class EvolveModule(nn.Module):
    def __init__(
        self,
        module_id: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        created_step: int = 0,
        origin_event: str = "initial",
        status: str = "active",
    ):
        super().__init__()
        self.module_id = module_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.created_step = created_step
        self.origin_event = origin_event
        self.status = status

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # RigL-inspired masks (1.0 = active, 0.0 = dropped)
        self.register_buffer("mask1", torch.ones_like(self.fc1.weight))
        self.register_buffer("mask2", torch.ones_like(self.fc2.weight))

    def apply_masks(self):
        """Zero out pruned weights."""
        with torch.no_grad():
            self.fc1.weight.mul_(self.mask1)
            self.fc2.weight.mul_(self.mask2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We apply masks implicitly or explicitly. For safety during eval/forward:
        # In a real heavy RigL, we'd hook the gradient. Here we just zero out the weights 
        # before forward pass, and typically after optimizer step (in agent).
        w1 = self.fc1.weight * self.mask1
        w2 = self.fc2.weight * self.mask2
        x = nn.functional.linear(x, w1, self.fc1.bias)
        x = self.relu(x)
        x = nn.functional.linear(x, w2, self.fc2.bias)
        return x

    def _aggregate_mask(self, mask: torch.Tensor, n_target_groups: int, n_source_groups: int):
        out_dim, in_dim = mask.shape
        out_chunk = out_dim / n_target_groups
        in_chunk = in_dim / n_source_groups
        
        aggregates = []
        for i in range(n_target_groups):
            for j in range(n_source_groups):
                out_start = int(i * out_chunk)
                out_end = int((i + 1) * out_chunk) if i < n_target_groups - 1 else out_dim
                in_start = int(j * in_chunk)
                in_end = int((j + 1) * in_chunk) if j < n_source_groups - 1 else in_dim
                
                submask = mask[out_start:out_end, in_start:in_end]
                active = int(submask.sum().item())
                total = submask.numel()
                density = float(active / total) if total > 0 else 0.0
                
                aggregates.append({
                    "target_idx": i,
                    "source_idx": j,
                    "active_connections": active,
                    "possible_connections": total,
                    "density": density
                })
        return aggregates

    def metadata(self) -> Dict[str, Any]:
        # Count only active parameters
        param_count = int(self.mask1.sum().item() + self.mask2.sum().item())
        param_count += self.fc1.bias.numel() + self.fc2.bias.numel()

        return {
            "module_id": self.module_id,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "created_step": self.created_step,
            "origin_event": self.origin_event,
            "status": self.status,
            "parameter_count": param_count,
            "mask1_aggregates": self._aggregate_mask(self.mask1, 3, 4),
            "mask2_aggregates": self._aggregate_mask(self.mask2, 4, 3),
        }


