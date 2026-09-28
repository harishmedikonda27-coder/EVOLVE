import torch
import torch.nn as nn
import random
from typing import List, Dict, Optional
from evolution.state import AdaptationState

class FeasibilityMask:
    """
    Zeros out impossible actions. It is NOT the final safety authority, 
    but it helps the policy avoid trivially rejected actions.
    Actions: REUSE(0), ADAPT(1), GROW(2), PRUNE(3), REWIRE(4)
    """
    def __init__(self, max_modules: int = 8, min_modules: int = 1):
        self.max_modules = max_modules
        self.min_modules = min_modules
        
    def get_mask(self, module_count: int, budget_remaining: int) -> torch.Tensor:
        # 1 means feasible, 0 means infeasible
        mask = torch.ones(5, dtype=torch.float32)
        
        # Structural mutations need budget
        if budget_remaining <= 0:
            mask[2] = 0 # GROW
            mask[3] = 0 # PRUNE
            mask[4] = 0 # REWIRE
            
        if module_count >= self.max_modules:
            mask[2] = 0 # GROW
            
        if module_count <= self.min_modules:
            mask[3] = 0 # PRUNE
            
        return mask

class LearnedPolicy(nn.Module):
    """
    A small DQN that predicts Q-values for structural evolution actions.
    It takes an AdaptationState vector as input and outputs Q-values for the 5 discrete actions.
    """
    def __init__(self, state_dim: int = 12, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.action_map = {0: "REUSE", 1: "ADAPT", 2: "GROW", 3: "PRUNE", 4: "REWIRE"}
        self.mask_generator = FeasibilityMask()
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass for Q-values"""
        return self.net(state)
        
    def select_action(self, state_vec: torch.Tensor, module_count: int, budget_remaining: int, epsilon: float = 0.0) -> str:
        """
        Selects an action using epsilon-greedy, masking out infeasible actions.
        state_vec should be shape (12,) or (1, 12).
        """
        device = next(self.parameters()).device
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
            
        state_vec = state_vec.to(device)
        mask = self.mask_generator.get_mask(module_count, budget_remaining).to(device)
        
        if torch.rand(1).item() < epsilon:
            feasible_indices = torch.nonzero(mask).squeeze(-1).tolist()
            if not feasible_indices:
                return "REUSE" # Safe fallback
            action_idx = random.choice(feasible_indices)
        else:
            with torch.no_grad():
                q_values = self(state_vec).squeeze(0)
                # Apply mask: set infeasible actions to large negative value
                q_values[mask == 0] = -float('inf')
                # If everything is masked, fallback
                if torch.isinf(q_values).all():
                    action_idx = 0
                else:
                    action_idx = int(torch.argmax(q_values).item())
                
        return self.action_map.get(action_idx, "REUSE")
