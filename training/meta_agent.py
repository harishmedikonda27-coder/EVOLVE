import torch
import torch.nn as nn
import torch.optim as optim
import copy
from evolution.memory import InterventionMemory
from evolution.learned_policy import LearnedPolicy

class MetaTrainer:
    """
    Trains the LearnedPolicy on InterventionMemory using off-policy Q-learning (DQN).
    This runs periodically (or in the background) to learn the structural evolution policy.
    """
    def __init__(self, policy: LearnedPolicy, learning_rate: float = 1e-3, gamma: float = 0.99, target_update_freq: int = 10):
        self.policy = policy
        self.target_policy = copy.deepcopy(policy)
        self.target_policy.eval()
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.updates = 0
        self.epsilon = 0.1
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
    def get_state(self):
        return {
            "policy": self.policy.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "epsilon": self.epsilon
        }
        
    def load_state(self, state):
        self.policy.load_state_dict(state["policy"])
        self.target_policy.load_state_dict(state["target_policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        self.epsilon = state.get("epsilon", 0.1)
        
    def update(self, memory: InterventionMemory, batch_size: int = 32) -> float:
        if len(memory) < batch_size:
            return 0.0
            
        device = next(self.policy.parameters()).device
        states, actions, rewards, next_states = memory.sample(batch_size, device)
        
        # Current Q-values
        q_values = self.policy(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_policy(next_states)
            # We don't strictly apply FeasibilityMask here to keep the target simple, 
            # but we could if we wanted to enforce bounded future expectations.
            next_state_values = next_q_values.max(1)[0]
            
        expected_state_action_values = rewards + (self.gamma * next_state_values)
        
        loss = self.criterion(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
            
        return float(loss.item())
