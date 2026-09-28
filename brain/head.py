import torch
import torch.nn as nn


class OutputHead(nn.Module):
    def __init__(self, module_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.module_dim = module_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(module_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
