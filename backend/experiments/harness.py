import os
import json
import uuid
import datetime
import torch
import random
import numpy as np
from typing import Dict, Any, List

from environment.world import UnifiedEvolveWorld, LevelConfig
from backend.experiments.config import ExperimentConfig, BaselineConfig
from backend.events import EventBus
from brain.brain import ModularBrain
from training.agent import DQNAgent
from training.checkpoints import CheckpointManager
from evolution.controller import EvolutionController
from evolution.mutation import MutationManager
from evolution.rollback import RollbackManager
from evolution.verify import Verifier
from evolution.transaction import TransactionRunner
from backend.tasks import TaskRegistry
from backend.invariants import validate_system_state
from training.convergence import ConvergenceTracker, ConvergenceStatus

from evolution.ood import OODMonitor
from evolution.plasticity import PlasticityProbeEngine
from evolution.memory import InterventionMemory
from evolution.consequence import LongHorizonTracker
from evolution.learned_policy import LearnedPolicy
from training.meta_agent import MetaTrainer

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def evaluate_agent(agent: DQNAgent, env: UnifiedEvolveWorld, episodes: int = 5, eval_seed: int = 42):
    agent.brain.eval()
    total_reward = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=eval_seed + ep)
        done = False
        while not done:
            action, _ = agent.select_action(obs, evaluate=True)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            done = term or trunc
    agent.brain.train()
    return total_reward / max(1, episodes), [1.0 for _ in range(len(agent.brain.registry.list_active_ids()))]

class ExperimentHarness:
    def __init__(self, config_path: str):
        self.config = ExperimentConfig.load(config_path)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def execute(self):
        print(f"Starting Experiment: {self.config.experiment_id}")
        for baseline in self.config.baselines:
            if not baseline.enabled:
                continue
            for seed in self.config.seeds:
                self.run_condition_seed(baseline, seed)

    def run_condition_seed(self, baseline: BaselineConfig, seed: int):
        print(f"\n{'='*60}\nCondition: {baseline.condition_name} | Seed: {seed}\n{'='*60}")
        set_seed(seed)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4).to(device)
        agent = DQNAgent(brain=brain)
        agent.target_brain = agent.target_brain.to(device)
        
        # Adaptation state objects
        ood_monitor = OODMonitor(obs_dim=16)
        plasticity_engine = PlasticityProbeEngine(budget=self.config.plasticity_probe_budget, eval_freq=100)
        intervention_memory = InterventionMemory(capacity=1000)
        consequence_tracker = LongHorizonTracker(memory=intervention_memory, horizon_tasks=self.config.consequence_horizon)
        
        if baseline.adaptation_mode in ["learned", "shadow"]:
            learned_policy = LearnedPolicy(state_dim=12, action_dim=5).to(device)
            meta_trainer = MetaTrainer(policy=learned_policy)
        else:
            learned_policy = None
            meta_trainer = None
            
        # Overwrite mode for random
        act_mode = baseline.adaptation_mode
        if baseline.random_valid_action:
            act_mode = "random"
            
        controller = EvolutionController(learned_policy=learned_policy, adaptation_mode=act_mode)
        mut_manager = MutationManager()
        verifier = Verifier()
        rollback = RollbackManager()
        ckpt_manager = CheckpointManager()
        tasks = TaskRegistry()
        
        if baseline.condition_name == "FIXED":
            # For FIXED, we disable all structural mutation logic by setting budget to 0 or overriding policy
            pass # Usually handled by max_param_budget or explicit policy wrapper
            
        global_step = 0
        last_plasticity_auc = 0.0
        
        # Datasets to persist
        run_data = {
            "experiment_id": self.config.experiment_id,
            "experiment_generation": "post_harness_repair_v1",
            "condition": baseline.condition_name,
            "seed": seed,
            "tasks": [],
            "interventions": [],
            "ood_events": [],
            "plasticity_probes": [],
            "eval_matrix": {} # global_step -> {task_id: perf}
        }
        
        for task_idx, t_conf in enumerate(self.config.task_sequence):
            l_config = LevelConfig(**t_conf)
            task_id = l_config.level_id
            env = UnifiedEvolveWorld(config=l_config, max_steps=300)
            
            tasks.register_task(task_id, l_config.task_variant, env.get_task_profile(), l_config.seed, l_config.difficulty_level)
            
            # 1. Evaluate baseline of old tasks BEFORE starting new task (Forward-Learning / Interference baseline)
            before_scores = {}
            for past_id in tasks.tasks.keys():
                if past_id == task_id: continue
                past_task = tasks.get_task(past_id)
                past_env = UnifiedEvolveWorld(LevelConfig(past_id, past_task.family, past_task.level, past_task.seed), 300)
                raw_p, _ = evaluate_agent(agent, past_env, eval_seed=past_task.seed)
                min_r, max_r = past_env.get_task_profile()["reward_spec"]["R_min"], past_env.get_task_profile()["reward_spec"]["R_max"]
                before_scores[past_id] = max(0.0, min(1.0, (raw_p - min_r) / (max_r - min_r)))
                
            # Note: fresh baseline evaluation would happen here if we run a second fresh agent, 
            # but usually it's approximated by epoch 0 or run independently.
            
            target_score = 0.8
            convergence_tracker = ConvergenceTracker(target_score=target_score, window_size=5, max_steps=self.config.max_steps_per_task)
            
            obs, _ = env.reset(seed=l_config.seed)
            level_step = 0
            eval_history_list = []
            
            # ...
            # We would run the full training loop here (similar to run.py)
            # This is a skeleton. The actual step-by-step logic should match run.py perfectly.
            # ...
            
            while level_step < self.config.max_steps_per_task:
                # Agent step
                if act_mode in ["learned", "shadow"]:
                    ood_monitor.add_observation(obs)
                    
                action, _ = agent.select_action(obs, evaluate=False)
                next_obs, reward, term, trunc, _ = env.step(action)
                agent.buffer.push(obs, action, reward, next_obs, term)
                agent.update()
                
                obs = next_obs if not (term or trunc) else env.reset()[0]
                global_step += 1
                level_step += 1
                
                # Evaluation
                if level_step % self.config.eval_freq == 0:
                    current_perf, _ = evaluate_agent(agent, env, eval_seed=l_config.seed)
                    profile = env.get_task_profile()
                    min_r, max_r = profile["reward_spec"]["R_min"], profile["reward_spec"]["R_max"]
                    norm_perf = max(0.0, min(1.0, (current_perf - min_r) / (max_r - min_r)))
                    
                    eval_history_list.append(norm_perf)
                    tasks.update_performance(task_id, norm_perf, global_step)
                    
                    # Store to eval matrix
                    if global_step not in run_data["eval_matrix"]:
                        run_data["eval_matrix"][global_step] = {}
                    run_data["eval_matrix"][global_step][task_id] = norm_perf
                    
                    conv_result = convergence_tracker.check(eval_history_list, level_step)
                    
                    if conv_result.status == ConvergenceStatus.CONVERGED or conv_result.status == ConvergenceStatus.EXHAUSTED:
                        break
                        
                    # Evolution logic for NON-FIXED
                    if baseline.condition_name != "FIXED":
                        decision = controller.evaluate_step(
                            current_performance=norm_perf,
                            target_performance=target_score,
                            recent_history=eval_history_list[-5:],
                            current_param_count=brain.parameter_count(),
                            max_param_budget=self.config.max_param_budget,
                            current_step=global_step,
                            active_module_count=len(brain.registry.list_active_ids()),
                            youngest_module_age=global_step
                        )
                        
                        if decision.action not in ["REUSE", "ADAPT"]:
                            if act_mode in ["learned", "shadow"]:
                                inv_id = consequence_tracker.register_intervention(decision.adaptation_state, decision.action, global_step, task_id)
                            else:
                                inv_id = None
                                
                            def create_env_fn(fam, lvl):
                                return UnifiedEvolveWorld(LevelConfig("test", fam, lvl, 42), 300)
                                
                            runner = TransactionRunner(mut_manager, rollback, verifier, None, "exp")
                            
                            topology_before = brain.parameter_count() # simplified topological tracking metric
                            modules_before = len(brain.registry.list_active_ids())
                            
                            class MockControl:
                                def __init__(self):
                                    import threading
                                    self.stop_requested = threading.Event()
                            class MockCkpt:
                                def save_checkpoint(self, *args, **kwargs): pass
                            def val_system_state(a, label): pass
                            
                            txn = runner.run_transaction(
                                decision=decision,
                                brain=brain,
                                agent=agent,
                                env=env,
                                obs=obs,
                                tasks=tasks,
                                controller=controller,
                                global_step=global_step,
                                level_step=level_step,
                                task_id=task_id,
                                family=l_config.task_variant,
                                before_scores={},
                                min_r=0.0,
                                max_r=1.0,
                                seed=seed,
                                control=MockControl(),
                                create_environment_fn=create_env_fn,
                                evaluate_agent_fn=evaluate_agent,
                                validate_system_state_fn=val_system_state,
                                norm_perf=norm_perf,
                                ckpt_manager=MockCkpt(),
                                results={}
                            )
                            
                            if hasattr(txn, "new_global_step") and txn.new_global_step is not None:
                                global_step = txn.new_global_step
                            if hasattr(txn, "new_level_step") and txn.new_level_step is not None:
                                level_step = txn.new_level_step
                            if hasattr(txn, "new_obs") and txn.new_obs is not None:
                                obs = txn.new_obs
                            
                            topology_after = brain.parameter_count()
                            modules_after = len(brain.registry.list_active_ids())
                            
                            intervention_record = {
                                "task_id": task_id,
                                "step": global_step,
                                "action": decision.action,
                                "status": txn.status,
                                "topology_before": topology_before,
                                "topology_after": topology_after,
                                "modules_before": modules_before,
                                "modules_after": modules_after
                            }
                            if inv_id is not None:
                                intervention_record["intervention_id"] = inv_id
                                
                            run_data["interventions"].append(intervention_record)

            # End of task logic
            # Execute plasticity probe for ALL conditions to ensure comparability
            # But we respect the config budget
            p_metrics = None
            if self.config.plasticity_probe_budget > 0:
                p_metrics = plasticity_engine.run_probe(agent, rollback)
                run_data["plasticity_probes"].append(p_metrics.to_dict())
                
            last_plasticity_auc = p_metrics.auc if p_metrics else None
            
            final_perf = eval_history_list[-1] if eval_history_list else 0.0
            resolved = consequence_tracker.on_task_complete(task_id, final_perf, last_plasticity_auc or 0.0, 1.0)
            if act_mode == "learned" and meta_trainer:
                meta_trainer.update(intervention_memory)
                    
            # Compute Continual Learning metrics for this task transition
            forgetting = tasks.calculate_forgetting(task_id)
            retention = tasks.calculate_retention(global_step)
            
            run_data["tasks"].append({
                "task_id": task_id,
                "steps_taken": level_step,
                "final_performance": eval_history_list[-1] if eval_history_list else 0.0,
                "forgetting": forgetting,
                "retention": retention
            })
            
        # Export data
        file_path = os.path.join(self.config.output_dir, f"{self.config.experiment_id}_{baseline.condition_name}_seed_{seed}.json")
        with open(file_path, "w") as f:
            json.dump(run_data, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    args = parser.parse_args()
    
    harness = ExperimentHarness(args.config)
    harness.execute()
