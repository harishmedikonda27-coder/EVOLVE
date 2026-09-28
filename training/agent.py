import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque
from typing import Tuple, Dict, Any, List, Optional


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.long, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        brain: nn.Module,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.998,
        buffer_capacity: int = 10000,
        target_update_freq: int = 100,
    ):
        self.brain = brain
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stability

        # Target network for stable Q-learning
        self.target_brain = copy.deepcopy(brain)
        self.target_brain.eval()

    def rebuild_optimizer(self):
        """Rebuilds optimizer parameter groups after structural mutation so new parameters are trained.
        
        Strategy: create a fresh Adam optimizer for *all* current brain parameters.
        Transfer momentum buffers (exp_avg, exp_avg_sq) from the old optimizer
        only for parameters that still exist (matched by data_ptr identity).
        New parameters start with clean optimizer state, which is correct because
        they have no meaningful history.
        """
        old_state = self.optimizer.state
        # Map old param data_ptr → old state
        old_param_state = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in old_state:
                    old_param_state[p.data_ptr()] = old_state[p]

        # Create fresh optimizer
        self.optimizer = optim.Adam(self.brain.parameters(), lr=self.learning_rate)

        # Transfer momentum buffers for surviving parameters
        device = next(self.brain.parameters()).device
        for group in self.optimizer.param_groups:
            for p in group['params']:
                ptr = p.data_ptr()
                if ptr in old_param_state:
                    state_copy = {}
                    for k, v in old_param_state[ptr].items():
                        if isinstance(v, torch.Tensor):
                            state_copy[k] = v.to(device)
                        else:
                            state_copy[k] = v
                    self.optimizer.state[p] = state_copy

        # Also update target network to match new architecture
        self.target_brain = copy.deepcopy(self.brain)
        self.target_brain.eval()

    def _sync_target(self):
        """Copy weights from online brain to target brain."""
        self.target_brain.load_state_dict(self.brain.state_dict())

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> Tuple[int, Optional[torch.Tensor]]:
        if not evaluate and random.random() < self.epsilon:
            action = random.randint(0, self.brain.action_dim - 1)
            weights = None
        else:
            with torch.no_grad():
                device = next(self.brain.parameters()).device
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_logits, weights = self.brain(obs_t)
                action = int(torch.argmax(action_logits, dim=-1).item())
        return action, weights

    def update(self, batch_size: int = 32) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None

        device = next(self.brain.parameters()).device
        obs, action, reward, next_obs, done = self.buffer.sample(batch_size, device)

        # Current Q values from online network
        action_logits, _ = self.brain(obs)
        state_action_values = action_logits.gather(1, action.unsqueeze(1)).squeeze(1)

        # Next Q values from target network (Double DQN style)
        with torch.no_grad():
            # Use online network to select best action
            next_online_logits, _ = self.brain(next_obs)
            best_actions = next_online_logits.argmax(dim=1, keepdim=True)
            # Use target network to evaluate that action
            next_target_logits, _ = self.target_brain(next_obs)
            next_state_values = next_target_logits.gather(1, best_actions).squeeze(1)
            expected_state_action_values = reward + (self.gamma * next_state_values * (1.0 - done))

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Periodically sync target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self._sync_target()

        return float(loss.item())
