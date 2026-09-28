import random
import torch
import numpy as np
from typing import Tuple, List, Dict
from collections import deque
from evolution.state import AdaptationState

class InterventionMemory:
    """
    Experience replay for structural evolution policy.
    Stores (state, action, delayed_consequence, next_state) transitions.
    
    IMPORTANT: We store both ACCEPTED and ROLLED-BACK mutations.
    For rolled-back mutations, next_state is the state *after* the rollback 
    (which is functionally similar to the initial state, but time has advanced).
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        # Action space mapping for discrete policy
        self.action_map = {"REUSE": 0, "ADAPT": 1, "GROW": 2, "PRUNE": 3, "REWIRE": 4}
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        
    def push(self, state: AdaptationState, action: str, consequence: float, next_state: AdaptationState):
        """
        Push a completed transition with computed long-horizon consequence into memory.
        """
        if action not in self.action_map:
            raise ValueError(f"Unknown action {action}")
        self.buffer.append((state, action, consequence, next_state))
        
    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = []
        actions = []
        consequences = []
        next_states = []
        
        for s, a, c, ns in batch:
            states.append(s.to_vector())
            actions.append(self.action_map[a])
            consequences.append(c)
            next_states.append(ns.to_vector())
            
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(consequences, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        )
        
    def get_all_experiences(self) -> List[Dict]:
        """Utility for inspection/serialization"""
        return [
            {
                "state": s.to_dict(),
                "action": a,
                "consequence": c,
                "next_state": ns.to_dict()
            } for s, a, c, ns in self.buffer
        ]
        
    def __len__(self):
        return len(self.buffer)
