import torch
import numpy as np
import copy
from typing import Dict, Any, Tuple
from environment.adaptive_frontier import AdaptiveFrontierWorld, get_level_spec
from training.agent import DQNAgent
from utils.rng_isolation import IsolatedRNGContext

class PlasticityMetrics:
    def __init__(self, initial_perf: float, final_perf: float, auc: float, slope: float,
                 task_id: str, requested_seed: int, resolved_seed: int, fingerprint: str):
        self.initial_perf = initial_perf
        self.final_perf = final_perf
        self.improvement = final_perf - initial_perf
        self.auc = auc
        self.slope = slope
        self.task_id = task_id
        self.requested_seed = requested_seed
        self.resolved_seed = resolved_seed
        self.task_fingerprint = fingerprint
        self.evaluation_type = "PLASTICITY"
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_perf": self.initial_perf,
            "final_perf": self.final_perf,
            "improvement": self.improvement,
            "auc": self.auc,
            "slope": self.slope,
            "task_id": self.task_id,
            "requested_seed": self.requested_seed,
            "resolved_seed": self.resolved_seed,
            "task_fingerprint": self.task_fingerprint,
            "evaluation_type": self.evaluation_type
        }

class PlasticityProbeEngine:
    """
    Measures structural plasticity observationally using an unseen 'probe task'.
    
    The engine isolates the system state, spins up a fresh MemoryObservation task
    at a scaled difficulty, trains for a fixed budget, measures improvement, 
    and restores the EXACT RNG and model state afterwards.
    """
    def __init__(self, budget: int = 1000, eval_freq: int = 100, probe_level: int = 1):
        self.budget = budget
        self.eval_freq = eval_freq
        self.probe_level = probe_level
        


    def run_probe(self, agent: DQNAgent, rollback_manager) -> PlasticityMetrics:
        """
        Runs the plasticity probe and returns the metrics.
        Guarantees that the agent's state is completely restored.
        """
        from backend.experiments.run import evaluate_agent
        
        with IsolatedRNGContext():
            # Snapshot state
            ckpt_state = rollback_manager.capture_checkpoint(agent.brain, agent.optimizer, {})
            
            # We need to snapshot the target network separately because rollback_manager might not cover it directly if it just captures brain.
            target_brain_state = copy.deepcopy(agent.target_brain.state_dict())
            epsilon_snapshot = agent.epsilon
            update_count_snapshot = agent.update_count
            
            # We also want to clear the agent's replay buffer temporarily so it learns only from the probe task.
            buffer_snapshot = list(agent.buffer.buffer)
            agent.buffer.buffer.clear()
            
            # Create explicitly held-out probe task
            spec = get_level_spec(999, 99999, distribution="plasticity_probe")
            probe_env = AdaptiveFrontierWorld(spec)
            profile = probe_env.get_task_profile()
            min_r, max_r = profile["reward_spec"]["R_min"], profile["reward_spec"]["R_max"]
            
            def normalize(perf: float) -> float:
                return max(0.0, min(1.0, (perf - min_r) / (max_r - min_r)))
                
            learning_curve = []
            
            # Initial Eval
            initial_raw, _ = evaluate_agent(agent, probe_env, episodes=5)
            initial_norm = normalize(initial_raw)
            learning_curve.append(initial_norm)
            
            obs, _ = probe_env.reset()
            
            # Adaptation Loop
            for step in range(1, self.budget + 1):
                action, _ = agent.select_action(obs, evaluate=False)
                next_obs, reward, terminated, truncated, _ = probe_env.step(action)
                agent.buffer.push(obs, action, reward, next_obs, terminated)
                agent.update()
                
                if terminated or truncated:
                    obs, _ = probe_env.reset()
                else:
                    obs = next_obs
                    
                if step % self.eval_freq == 0:
                    perf_raw, _ = evaluate_agent(agent, probe_env, episodes=5)
                    learning_curve.append(normalize(perf_raw))
                    
            final_norm = learning_curve[-1] if learning_curve else initial_norm
            
            # Compute AUC and Slope (Normalized to [0, 1])
            raw_auc = float(np.trapz(learning_curve, dx=self.eval_freq))
            auc = min(1.0, max(0.0, raw_auc / self.budget))
            
            x = np.arange(len(learning_curve)) * self.eval_freq
            y = np.array(learning_curve)
            if len(x) > 1:
                slope, _ = np.polyfit(x, y, 1)
            else:
                slope = 0.0
                
            metrics = PlasticityMetrics(
                initial_perf=initial_norm,
                final_perf=final_norm,
                auc=auc,
                slope=float(slope),
                task_id=spec.task_id,
                requested_seed=spec.requested_seed,
                resolved_seed=spec.resolved_seed,
                fingerprint=probe_env.get_fingerprint()
            )
            
            # Restore State
            agent.brain, agent.optimizer = rollback_manager.restore_checkpoint(agent.brain, agent.optimizer, ckpt_state)
            agent.target_brain.load_state_dict(target_brain_state)
            agent.epsilon = epsilon_snapshot
            agent.update_count = update_count_snapshot
            
            # Restore replay buffer
            agent.buffer.buffer.clear()
            agent.buffer.buffer.extend(buffer_snapshot)
            
            return metrics
