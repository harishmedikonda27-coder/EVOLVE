import argparse
import json
import os
import torch
import numpy as np
import random
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

import copy
import uuid
import datetime

from environment.adaptive_frontier import AdaptiveFrontierWorld, get_level_spec
from brain.brain import ModularBrain

from training.agent import DQNAgent
from training.checkpoints import CheckpointManager
from evolution.controller import EvolutionController
from evolution.mutation import MutationManager
from evolution.rollback import RollbackManager
from evolution.verify import Verifier
from evolution.transaction import TransactionRunner
from backend.invariants import validate_system_state
from backend.tasks import TaskRegistry
from backend.events import EventBus, Event
from training.convergence import ConvergenceTracker, ConvergenceStatus

# Phase 2 imports
from evolution.ood import OODMonitor
from evolution.plasticity import PlasticityProbeEngine
from evolution.memory import InterventionMemory
from evolution.consequence import LongHorizonTracker
from evolution.learned_policy import LearnedPolicy
from training.meta_agent import MetaTrainer
from evolution.state import AdaptationState

import threading

class FailedToConverge(Exception):
    pass

class RunControl:
    def __init__(self):
        self.stop_requested = threading.Event()

run_controls: Dict[str, RunControl] = {}

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_environment(family: str, level: int, seed: int):
    human_level = level + 1
    distribution = "ood" if family == "OOD" else "train"
    spec = get_level_spec(human_level, seed, distribution)
    return AdaptiveFrontierWorld(spec)

def evaluate_agent(agent: DQNAgent, eval_env, episodes: int = 5) -> Tuple[float, List[float]]:
    agent.brain.eval()
    total_reward = 0.0
    routing_weights_list = []
    for ep in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, weights = agent.select_action(obs, evaluate=True)
            if weights is not None:
                routing_weights_list.append(weights.cpu().numpy())
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
    agent.brain.train()
    
    if routing_weights_list:
        avg_weights = np.mean(np.concatenate(routing_weights_list, axis=0), axis=0).tolist()
    else:
        avg_weights = [1.0 for _ in range(len(agent.brain.registry.list_active_ids()))]
        
    return total_reward / max(1, episodes), avg_weights

def publish_event(event_bus: Optional[EventBus], run_id: str, event_type: str, phase: str, step: int, payload: Dict[str, Any]):
    if event_bus:
        event_bus.publish(Event(
            event_type=event_type,
            run_id=run_id,
            step=step,
            phase=phase,
            payload=payload
        ))

def run_continuous_experiment(
    run_id: str,
    event_bus: Optional[EventBus] = None,
    num_levels: int = 30,
    seed: int = 42,
    max_steps: int = 20000,
    eval_freq: int = 500,
    adaptation_mode: str = "deterministic",
    plasticity_threshold_active_override: Optional[float] = None,
    plasticity_threshold_degraded_override: Optional[float] = None
) -> Dict[str, Any]:
    if run_id not in run_controls:
        run_controls[run_id] = RunControl()
    control = run_controls[run_id]

    set_seed(seed)
    print(f"[EVOLVE] Starting continuous experiment {run_id} - Seed: {seed}")
    # Device detection and logging
    if torch.cuda.is_available():
        device_name = "cuda"
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_available": True,
            "pytorch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "selected_device": "cuda",
            "gpu_memory_allocated": torch.cuda.memory_allocated(0),
            "gpu_memory_reserved": torch.cuda.memory_reserved(0),
        }
    else:
        device_name = "cpu"
        gpu_info = {
            "gpu_name": "N/A",
            "cuda_available": False,
            "pytorch_version": torch.__version__,
            "cuda_runtime_version": "N/A",
            "selected_device": "cpu",
            "gpu_memory_allocated": 0,
            "gpu_memory_reserved": 0,
        }
    
    device = torch.device(device_name)
    print(f"[EVOLVE][RUN] id={run_id} device={device_name} gpu={gpu_info['gpu_name']} seed={seed} mode={adaptation_mode} world=AdaptiveFrontier")

    publish_event(event_bus, run_id, "RUN_STARTED", "INITIAL", 0, {
        "num_levels": num_levels,
        "seed": seed,
        "gpu_info": gpu_info
    })

    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=5).to(device)
    agent = DQNAgent(brain=brain, epsilon_decay=0.9995)
    
    # Force agent target brain to match device immediately
    agent.target_brain = agent.target_brain.to(device)
    gpu_info["model_parameter_count"] = sum(p.numel() for p in brain.parameters())
    
    from evolution.scorer import NeedScorer
    from evolution.policies import DecisionPolicy
    import json
    import os
    
    # Load explicit policy configuration
    policy_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "policy.json")
    if os.path.exists(policy_path):
        with open(policy_path, "r") as f:
            policy_cfg = json.load(f)
    else:
        policy_cfg = {}
        
    p_weights = policy_cfg.get("need_weights", {})
    scorer = NeedScorer(
        w_p=p_weights.get("performance_gap", 0.40),
        w_s=p_weights.get("stagnation", 0.30),
        w_n=p_weights.get("novelty", 0.15),
        w_i=p_weights.get("interference", 0.15),
        w_c=p_weights.get("cost", 0.20)
    )
    
    p_thresh = policy_cfg.get("mutation_thresholds", {})
    d_policy = DecisionPolicy(
        adapt_threshold=p_thresh.get("adapt", 0.25),
        grow_threshold=p_thresh.get("grow", 0.55),
        prune_threshold=p_thresh.get("prune", 0.15),
        cooldown_steps=policy_cfg.get("mutation_cooldown", 150),
        max_modules=policy_cfg.get("structural_budgets", {}).get("max_modules", 5),
        max_parameters=policy_cfg.get("structural_budgets", {}).get("max_parameters", 50000),
        min_module_lifetime=policy_cfg.get("structural_budgets", {}).get("min_module_age", 1000),
        need_score_delta_threshold=policy_cfg.get("evidence_change_thresholds", {}).get("need_score_delta", 0.05)
    )
    
    plasticity_threshold_active = plasticity_threshold_active_override if plasticity_threshold_active_override is not None else policy_cfg.get("plasticity_thresholds", {}).get("active", 0.5)
    plasticity_threshold_degraded = plasticity_threshold_degraded_override if plasticity_threshold_degraded_override is not None else policy_cfg.get("plasticity_thresholds", {}).get("degraded", 0.1)

    # Phase 2 Components
    ood_monitor = OODMonitor(obs_dim=16, threshold=policy_cfg.get("ood_threshold", 2.0))
    plasticity_engine = PlasticityProbeEngine(budget=policy_cfg.get("verification_budget", 500), eval_freq=100)
    intervention_memory = InterventionMemory(capacity=1000)
    consequence_tracker = LongHorizonTracker(memory=intervention_memory, horizon_tasks=1)
    
    if adaptation_mode in ["learned", "shadow"]:
        learned_policy = LearnedPolicy(state_dim=12, action_dim=5).to(device)
        meta_trainer = MetaTrainer(policy=learned_policy)
    else:
        learned_policy = None
        meta_trainer = None
        
    controller = EvolutionController(scorer=scorer, policy=d_policy, learned_policy=learned_policy, adaptation_mode=adaptation_mode)
    mut_manager = MutationManager()
    verifier = Verifier(
        min_new_task_gain=policy_cfg.get("acceptance_thresholds", {}).get("min_new_task_gain", 0.10),
        min_old_task_retention=policy_cfg.get("acceptance_thresholds", {}).get("min_old_task_retention", 0.85)
    )
    rollback = RollbackManager()
    ckpt_manager = CheckpointManager()
    tasks = TaskRegistry()

    global_step = 0
    last_plasticity_auc = None
    last_task_perf = 0.0
    last_accepted_action = "STABLE — NO STRUCTURAL CHANGE"
    results = {"task_outcomes": {}}

    family = "AdaptiveFrontier"
    for level in range(num_levels):
        human_level = level + 1
        task_id = f"AdaptiveFrontier_L{human_level:02d}"
        
        level_base_seed = seed + level * 10000
        # The canonical level seed IS the task
        env = create_environment(family, level, level_base_seed)
        prof = env.get_task_profile()
        tasks.register_task(task_id, family, prof, env.spec.resolved_seed, level)
        
        # We evaluate on the EXACT SAME canonical task environment to track performance.
        eval_env = create_environment(family, level, level_base_seed)
        
        print(f"[EVOLVE][LEVEL] L{human_level:02d} START task={task_id} obj={env.spec.objective_type} grid={env.spec.grid_size}x{env.spec.grid_size} obs={env.spec.obstacle_density:.2f} hazard={env.spec.hazard_density:.2f} uncert={env.spec.transition_uncertainty:.2f} modules={len(brain.registry.list_active_ids())} params={brain.parameter_count()} rev={brain.topology_revision}")
        
        publish_event(event_bus, run_id, "LEVEL_STARTED", family, level, {
            "task_id": task_id,
            "family": family,
            "level": level
        })
        publish_event(event_bus, run_id, "TOPOLOGY_SNAPSHOT", family, global_step, brain.topology_snapshot())
        publish_event(event_bus, run_id, "TOPOLOGY_HISTORY_ENTRY", family, global_step, {
            "label": "INITIAL",
            "level_id": task_id,
            "level_index": level,
            "event_type": "LEVEL_STARTED",
            "topology_revision": brain.topology_revision,
            "parameter_count": brain.parameter_count(),
            "active_module_count": len(brain.registry.list_active_ids()),
            "topology_snapshot": brain.topology_snapshot()
        })
        
        # Baseline evaluation on previous tasks
        before_scores = {}
        for past_id, past_task in tasks.tasks.items():
            if past_id != task_id and past_task.final_performance > 0.5:
                past_env = create_environment(past_task.family, past_task.level, past_task.seed)
                raw_p, _ = evaluate_agent(agent, past_env)
                past_profile = past_env.get_task_profile()
                past_reward_spec = past_profile["reward_spec"]
                min_r, max_r = past_reward_spec["R_min"], past_reward_spec["R_max"]
                before_scores[past_id] = max(0.0, min(1.0, (raw_p - min_r) / (max_r - min_r)))
                
        # Step Accounting Definitions:
        # global_step: The total number of environment transitions across ALL tasks and phases (training + verification).
        # training_step (formerly level_step): The number of environment transitions spent strictly optimizing the CURRENT task. Verification steps are explicitly excluded from this budget.
        # verification_step: The number of transitions spent evaluating a structural mutation candidate. This advances global_step but NOT training_step.
        # decision_step: Implicitly, the points where training_step % eval_freq == 0 and a structural mutation may be proposed.
        target_score = policy_cfg.get("convergence_threshold", 0.95)
        convergence_tracker = ConvergenceTracker(
            target_score=target_score,
            window_size=policy_cfg.get("convergence_window", 5),
            min_evaluations=3,
            variance_threshold=0.005,
            stagnation_slope=0.001,
            max_steps=max_steps,
        )
        eval_history_list: List[float] = []
        training_step = 0
        
        obs, _ = env.reset()
            
        controller.policy.reset_level_stats()
        
        level_converged = False
        movement_batch = []
        
        while True:
            if control.stop_requested.is_set():
                print(f"[{run_id}] Stop requested. Halting run.")
                publish_event(event_bus, run_id, "RUN_STOPPING", family, global_step, {})
                ckpt_id = f"ckpt_{task_id}_stopped"
                ckpt_manager.save_checkpoint(
                    checkpoint_id=ckpt_id, 
                    brain=brain, 
                    optimizer=agent.optimizer, 
                    metadata={"task": task_id},
                    meta_trainer=meta_trainer,
                    consequence_tracker=consequence_tracker,
                    intervention_memory=intervention_memory,
                    ood_monitor=ood_monitor
                )
                publish_event(event_bus, run_id, "RUN_STOPPED", "COMPLETE", global_step, results)
                return results

            import time
            # Yield slightly to prevent starving the websocket thread and causing frontend lag
            if training_step % 10 == 0:
                time.sleep(0.01)
                
            # 1. Training Step
            prev_pos = env.agent_pos
            ood_monitor.add_observation(obs)
                
            action, _ = agent.select_action(obs, evaluate=False)
            next_obs, reward, term, trunc, info = env.step(action)
            
            new_pos = env.agent_pos
            action_val = int(action) if hasattr(action, 'item') else int(action)
            action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}
            
            movement_batch.append({
                "step": training_step + 1,
                "previous_position": prev_pos,
                "position": new_pos,
                "action": action_map.get(action_val, "UNKNOWN"),
                "blocked": prev_pos == new_pos
            })
            
            agent.buffer.push(obs, action, reward, next_obs, term)
            agent.update()
            
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
                
            global_step += 1
            training_step += 1
            
            # Emit environment state less frequently to avoid spamming the websocket if needed,
            # but for this experiment, let's emit every 10 steps so the frontend has live motion
            if training_step % 20 == 0:
                payload = env.get_state()
                payload.update({
                    "global_step": global_step,
                    "training_step": training_step,
                    "topology_version": brain.topology_version if hasattr(brain, "topology_version") else 1,
                    "movements": movement_batch
                })
                payload.update(info)
                publish_event(event_bus, run_id, "ENVIRONMENT_STATE_UPDATE", family, global_step, payload)
                movement_batch = []
            
            # 2. Evaluation & Convergence Check
            if training_step % eval_freq == 0:
                current_perf, current_utils = evaluate_agent(agent, eval_env)
                profile = env.get_task_profile()
                reward_spec = profile["reward_spec"]
                min_r, max_r = reward_spec["R_min"], reward_spec["R_max"]
                
                norm_perf = max(0.0, min(1.0, (current_perf - min_r) / (max_r - min_r)))
                
                eval_history_list.append(norm_perf)
                tasks.update_performance(task_id, norm_perf)
                
                # --- Convergence check via ConvergenceTracker ---
                conv_result = convergence_tracker.check(eval_history_list, training_step)

                # 3. Structural Evolution Evaluation
                after_scores = {}
                forgetting_scores = []
                for past_id in tasks.tasks.keys():
                    if past_id == task_id:
                        continue
                    past_env = create_environment(tasks.tasks[past_id].family, tasks.tasks[past_id].level, tasks.tasks[past_id].seed)
                    raw_p, _ = evaluate_agent(agent, past_env)
                    past_profile = past_env.get_task_profile()
                    past_reward_spec = past_profile["reward_spec"]
                    min_r_past, max_r_past = past_reward_spec["R_min"], past_reward_spec["R_max"]
                    current_p = max(0.0, min(1.0, (raw_p - min_r_past) / (max_r_past - min_r_past)))
                    after_scores[past_id] = current_p
                    
                    reference_p = tasks.tasks[past_id].final_performance
                    forgetting_scores.append(max(0.0, reference_p - current_p))
                
                ood_shift_val = 0.0
                ood_metrics = ood_monitor.compute_distance()
                ood_shift_val = ood_metrics["mean_distance"]
                publish_event(event_bus, run_id, "OOD_SHIFT_DETECTED", family, global_step, {
                    "shift_distance": ood_shift_val,
                    "raw_distance": ood_metrics["raw_distance"],
                    "normalization_factor": ood_metrics["ref_std_mean"]
                })
                    
                interference = tasks.calculate_interference(before_scores, after_scores)
                retention = sum(after_scores.values()) / len(after_scores) if after_scores else None
                forgetting = sum(forgetting_scores) / len(forgetting_scores) if forgetting_scores else None

                # OOD Recovery Calculation
                ood_recovery = None
                ood_denominator = None
                structural_change_accepted = 1 if last_accepted_action not in ["REUSE", "STABLE — NO STRUCTURAL CHANGE"] else 0
                if ood_shift_val > 0.0 and len(eval_history_list) > 0:
                    shifted_perf = eval_history_list[0]
                    # The OOD monitor's reference distribution is the FIRST task (L01).
                    # Therefore, the valid baseline performance for OOD recovery must be L01's final performance.
                    reference_task_id = "AdaptiveFrontier_L01"
                    if reference_task_id in tasks.tasks and tasks.tasks[reference_task_id].final_performance > 0:
                        baseline_perf = tasks.tasks[reference_task_id].final_performance
                        ood_denominator = baseline_perf - shifted_perf
                        if ood_denominator > 0:
                            recovery_val = (norm_perf - shifted_perf) / ood_denominator
                            ood_recovery = max(0.0, min(1.0, recovery_val))
                        else:
                            ood_recovery = 1.0 # No drop = fully recovered or didn't impact

                publish_event(event_bus, run_id, "METRICS_UPDATE", family, global_step, {
                    "step": global_step,
                    "training_step": training_step,
                    "reward": reward,
                    "reward_raw": current_perf,
                    "performance": norm_perf,
                    "normalization_min": min_r,
                    "normalization_max": max_r,
                    "topology_version": brain.topology_version if hasattr(brain, "topology_version") else 1,
                    "eval_episodes": 5,
                    "eval_seed": level_base_seed,
                    "convergence_status": conv_result.status.value,
                    "convergence_variance": conv_result.variance,
                    "convergence_slope": conv_result.slope,
                    "retention": retention,
                    "forgetting": forgetting,
                    "ood_shift": ood_shift_val,
                    "ood_recovery": ood_recovery,
                    "last_action": last_accepted_action,
                    "plasticity_status": (
                        "NOT TESTED" if last_plasticity_auc is None else
                        "ACTIVE" if last_plasticity_auc >= plasticity_threshold_active else 
                        "DEGRADED" if last_plasticity_auc > plasticity_threshold_degraded else 
                        "LOST" if last_plasticity_auc > 0.0 else "NOT TESTED"
                    )
                })
                
                print(f"[EVOLVE][EVAL] L{human_level:02d} S{training_step} raw={current_perf:.3f} perf={norm_perf:.3f} success={1 if norm_perf >= 0.8 else 0} retention={f'{retention:.3f}' if retention is not None else 'NA'} forgetting={f'{forgetting:.3f}' if forgetting is not None else 'NA'}")
                
                if conv_result.status == ConvergenceStatus.CONVERGED:
                    print(f"\n[CONVERGENCE DECISION]")
                    print(f"  Status:    {conv_result.status.value}")
                    print(f"  Variance:  {conv_result.variance:.5f}")
                    print(f"  Slope:     {conv_result.slope:.5f}")
                    print(f"  Reason:    {conv_result.reason}")
                    print(f">>> Task {task_id} Converged in {training_step} steps! <<<")
                    level_converged = True
                    last_task_perf = norm_perf
                    publish_event(event_bus, run_id, "CONVERGENCE", family, global_step, {
                        "task_id": task_id,
                        "training_step": training_step,
                        "final_performance": norm_perf,
                        "convergence_reason": conv_result.reason,
                    })
                    publish_event(event_bus, run_id, "TOPOLOGY_HISTORY_ENTRY", family, global_step, {
                        "label": "FINAL",
                        "level_id": task_id,
                        "level_index": level,
                        "event_type": "CONVERGENCE",
                        "topology_revision": brain.topology_revision,
                        "parameter_count": brain.parameter_count(),
                        "active_module_count": len(brain.registry.list_active_ids()),
                        "topology_snapshot": brain.topology_snapshot()
                    })
                    break
                    
                if conv_result.status == ConvergenceStatus.EXHAUSTED:
                    print(f"\n[CONVERGENCE DECISION]")
                    print(f"  Status:    {conv_result.status.value}")
                    print(f"  Variance:  {conv_result.variance:.5f}")
                    print(f"  Slope:     {conv_result.slope:.5f}")
                    print(f"  Reason:    {conv_result.reason}")
                    msg = f"Task {task_id} failed to converge within {max_steps} steps."
                    publish_event(event_bus, run_id, "ERROR", family, global_step, {"error": msg, "reason": "FAILED_TO_CONVERGE"})
                    print(msg)
                    level_converged = False
                    break
                    
                novelty = tasks.calculate_novelty(family, env.get_task_profile())
                utils = current_utils
                
                active_mods = [brain.registry.get_module(mid) for mid in brain.registry.list_active_ids()]
                y_age = global_step - max((m.created_step for m in active_mods), default=0)
                
                state_box = []
                decision = controller.evaluate_step(
                    current_performance=norm_perf,
                    target_performance=target_score,
                    recent_history=eval_history_list[-5:],
                    task_profile_distance=novelty,
                    interference_value=interference,
                    module_utilizations=utils,
                    current_param_count=brain.parameter_count(),
                    max_param_budget=policy_cfg.get("structural_budgets", {}).get("max_parameters", 50000),
                    current_step=global_step,
                    active_module_count=len(brain.registry.list_active_ids()),
                    youngest_module_age=y_age,
                    ood_shift=ood_shift_val,
                    plasticity_auc=last_plasticity_auc,
                    adaptation_state_out=state_box,
                    brain=brain
                )
                
                publish_event(event_bus, run_id, "DECISION_MADE", family, global_step, {
                    "action": decision.action,
                    "need_score": decision.need_score,
                    "signals": {
                        "performance_gap": decision.signals.performance_gap,
                        "stagnation": decision.signals.stagnation,
                        "novelty": decision.signals.novelty,
                        "interference": decision.signals.interference,
                        "utilization": decision.signals.utilization,
                        "cost": decision.signals.cost,
                    },
                    "reason": decision.reason,
                    "budget_remaining": decision.budget_remaining,
                    "cooldown_active": decision.cooldown_active,
                    "q_values": decision.q_values,
                    "adaptation_state": decision.adaptation_state.to_dict() if hasattr(decision, "adaptation_state") and decision.adaptation_state else None
                })
                
                util_str = f"{decision.signals.utilization:.3f}" if decision.signals.utilization is not None else "NA"
                plast_str = f"{last_plasticity_auc:.3f}" if last_plasticity_auc is not None else "NA"
                print(f"[EVOLVE][SIGNALS] L{human_level:02d} S{training_step} gap={decision.signals.performance_gap:.3f} stag={decision.signals.stagnation:.3f} nov={decision.signals.novelty:.3f} int={decision.signals.interference:.3f} util={util_str} ood={ood_shift_val:.3f} plast={plast_str} cost={decision.signals.cost:.3f} need={decision.need_score:.3f}")
                # Log any actions that were preferred but blocked
                for act in ["GROW", "REWIRE", "PRUNE"]:
                    if act in decision.blocked_actions:
                        # Determine if this action would have been selected based on thresholds
                        is_preferred = False
                        if act == "GROW" and decision.need_score >= controller.policy.grow_threshold: is_preferred = True
                        if act == "REWIRE" and decision.need_score >= controller.policy.rewire_threshold: is_preferred = True
                        if act == "PRUNE" and decision.signals.utilization is not None and decision.signals.utilization < controller.policy.prune_threshold: is_preferred = True
                        
                        if is_preferred:
                            reason_blocked = decision.blocked_reasons.get(act, "Unknown")
                            publish_event(event_bus, run_id, "DECISION_MADE", family, global_step, {
                                "action": act,
                                "need_score": decision.need_score,
                                "signals": {
                                    "performance_gap": decision.signals.performance_gap,
                                    "stagnation": decision.signals.stagnation,
                                    "novelty": decision.signals.novelty,
                                    "interference": decision.signals.interference,
                                    "utilization": decision.signals.utilization,
                                    "cost": decision.signals.cost,
                                },
                                "reason": reason_blocked,
                                "budget_remaining": decision.budget_remaining,
                                "cooldown_active": decision.cooldown_active,
                                "q_values": decision.q_values,
                                "feasible": False,
                                "blocked_reason": reason_blocked
                            })
                            print(f"[EVOLVE][DECISION] L{human_level:02d} S{training_step} action={act} need={decision.need_score:.3f} feasible=0 cooldown={1 if decision.cooldown_active else 0} reason=\"{reason_blocked}\"")

                print(f"[EVOLVE][DECISION] L{human_level:02d} S{training_step} action={decision.action} need={decision.need_score:.3f} feasible={0 if decision.cooldown_active else 1} cooldown={1 if decision.cooldown_active else 0} reason=\"{decision.reason}\"")
                
                if decision.action not in ["REUSE", "ADAPT"]:
                    publish_event(event_bus, run_id, "MUTATION_PROPOSED", family, global_step, {
                        "level": human_level,
                        "task_id": task_id,
                        "action": decision.action,
                        "need_score": decision.need_score,
                        "performance_gap": decision.signals.performance_gap,
                        "stagnation": decision.signals.stagnation,
                        "novelty": decision.signals.novelty,
                        "interference": decision.signals.interference,
                        "utilization": decision.signals.utilization,
                        "ood_shift": ood_shift_val,
                        "plasticity": last_plasticity_auc,
                        "structural_cost": decision.signals.cost,
                        "active_modules_before": len(brain.registry.list_active_ids()),
                        "parameters_before": brain.parameter_count(),
                        "topology_revision_before": brain.topology_revision,
                        "reason": decision.reason,
                        "feasibility_result": not decision.cooldown_active
                    })
                    
                    if adaptation_mode in ["learned", "shadow"] and len(state_box) > 0:
                        inv_id = consequence_tracker.register_intervention(
                            state_box[0], decision.action, global_step, task_id
                        )
                    else:
                        inv_id = None
                        
                    runner = TransactionRunner(mut_manager, rollback, verifier, event_bus, run_id)
                    res = runner.run_transaction(
                        decision=decision,
                        brain=brain,
                        agent=agent,
                        env=env,
                        obs=obs,
                        tasks=tasks,
                        controller=controller,
                        global_step=global_step,
                        training_step=training_step,
                        task_id=task_id,
                        family=family,
                        before_scores=before_scores,
                        min_r=min_r,
                        max_r=max_r,
                        seed=seed,
                        eval_env=eval_env,
                        control=control,
                        create_environment_fn=create_environment,
                        evaluate_agent_fn=evaluate_agent,
                        validate_system_state_fn=validate_system_state,
                        norm_perf=norm_perf,
                        ckpt_manager=ckpt_manager,
                        results=results,
                        verification_budget=policy_cfg.get("verification_budget", 500)
                    )
                    
                    if res.stopped:
                        return results
                        
                    if inv_id is not None and res.status != "ERROR":
                        next_state_box = []
                        _ = controller.evaluate_step(
                            current_performance=norm_perf,
                            target_performance=target_score,
                            recent_history=eval_history_list[-5:],
                            task_profile_distance=novelty,
                            interference_value=interference,
                            module_utilizations=utils,
                            current_param_count=brain.parameter_count(),
                            max_param_budget=policy_cfg.get("structural_budgets", {}).get("max_parameters", 50000),
                            current_step=global_step,
                            active_module_count=len(brain.registry.list_active_ids()),
                            youngest_module_age=y_age,
                            ood_shift=ood_shift_val,
                            plasticity_auc=last_plasticity_auc,
                            adaptation_state_out=next_state_box
                        )
                        consequence_tracker.register_next_state(inv_id, next_state_box[0])
                        
                    obs = res.new_obs
                    global_step = res.new_global_step
                    training_step = res.new_training_step
                    if res.status == "ACCEPT":
                        last_accepted_action = res.action
        # Phase 2 Plasticity & Consequence (Always execute now for metric accuracy)
        p_metrics = plasticity_engine.run_probe(agent, rollback)
        last_plasticity_auc = p_metrics.auc
        publish_event(event_bus, run_id, "PLASTICITY_PROBE", family, global_step, p_metrics.to_dict())
        
        print(f"[EVOLVE][PLASTICITY] L{human_level:02d} probe={task_id}_probe budget={plasticity_engine.budget} initial={p_metrics.initial_perf:.3f} final={p_metrics.final_perf:.3f} auc={p_metrics.auc:.3f} slope={p_metrics.slope:.4f} state={'ACTIVE' if p_metrics.auc >= plasticity_threshold_active else 'DEGRADED' if p_metrics.auc > plasticity_threshold_degraded else 'LOST'}")
        
        baseline_str = f"{baseline_perf:.3f}" if 'baseline_perf' in locals() else "NA"
        shifted_str = f"{shifted_perf:.3f}" if 'shifted_perf' in locals() else "NA"
        denom_str = f"{ood_denominator:.3f}" if 'ood_denominator' in locals() and ood_denominator is not None else "NA"
        post_perf = eval_history_list[-1] if eval_history_list else 0.0
        
        raw_dist_str = f"{ood_metrics.get('raw_distance', 0.0):.3f}" if 'ood_metrics' in locals() else "NA"
        norm_factor_str = f"{ood_metrics.get('ref_std_mean', 0.0):.3f}" if 'ood_metrics' in locals() else "NA"
        
        # If we didn't meet min samples, OOD measurement is invalid
        if len(ood_monitor.reference_window) < ood_monitor.min_samples or len(ood_monitor.current_window) < ood_monitor.min_samples:
            shift_str = "UNAVAILABLE"
            shift_status = "UNAVAILABLE"
        else:
            shift_str = f"{ood_shift_val:.3f}"
            shift_status = "DETECTED" if ood_shift_val > ood_monitor.threshold else "NORMAL"
        
        print(f"[EVOLVE][OOD] L{human_level:02d} raw_dist={raw_dist_str} norm_factor={norm_factor_str} shift={shift_str} threshold={ood_monitor.threshold:.3f} status={shift_status} baseline={baseline_str} shifted={shifted_str} post={post_perf:.3f} denominator={denom_str} recovery={f'{ood_recovery:.3f}' if ood_recovery is not None else 'NA'} adaptation_action={last_accepted_action} accepted_structural_change={'YES' if structural_change_accepted else 'NO'}")
            
        final_retention = 1.0
        final_forgetting = 0.0
        if len(before_scores) > 0:
            ret_scores = []
            fgt_scores = []
            for past_id in tasks.tasks.keys():
                if past_id == task_id: continue
                if past_id not in before_scores: continue
                p_env = create_environment(tasks.tasks[past_id].family, tasks.tasks[past_id].level, tasks.tasks[past_id].seed)
                raw_p, _ = evaluate_agent(agent, p_env)
                p_prof = p_env.get_task_profile()
                min_r_past, max_r_past = p_prof["reward_spec"]["R_min"], p_prof["reward_spec"]["R_max"]
                curr_p = max(0.0, min(1.0, (raw_p - min_r_past)/(max_r_past - min_r_past)))
                ret_scores.append(curr_p)
                fgt_scores.append(max(0.0, before_scores[past_id] - curr_p))
            if ret_scores:
                final_retention = sum(ret_scores) / len(ret_scores)
            if fgt_scores:
                final_forgetting = sum(fgt_scores) / len(fgt_scores)
                
        final_perf = eval_history_list[-1] if len(eval_history_list) > 0 else 0.0
        resolved = consequence_tracker.on_task_complete(task_id, final_perf, last_plasticity_auc, final_retention)
        
        for r in resolved:
            publish_event(event_bus, run_id, "CONSEQUENCE_RESOLVED", family, global_step, r)
        
        if adaptation_mode == "learned":
            meta_loss = meta_trainer.update(intervention_memory)
            if meta_loss > 0:
                publish_event(event_bus, run_id, "META_TRAIN_UPDATE", family, global_step, {"loss": meta_loss})
                    
        ckpt_id = f"ckpt_{task_id}"
        ckpt_manager.save_checkpoint(
            checkpoint_id=ckpt_id, 
            brain=brain, 
            optimizer=agent.optimizer, 
            metadata={"task": task_id},
            meta_trainer=meta_trainer,
            consequence_tracker=consequence_tracker,
            intervention_memory=intervention_memory,
            ood_monitor=ood_monitor
        )
        tasks.save_checkpoint_info(task_id, ckpt_id, "hash")
        
        snapshot = {
            "task_id": task_id,
            "family": family,
            "converged": level_converged,
            "steps": training_step,
            "snapshot_id": ckpt_id,
            "topology": brain.topology_snapshot(),
            "final_performance": eval_history_list[-1] if len(eval_history_list)>0 else 0,
            "global_step": global_step,
            "topology_version": brain.topology_version if hasattr(brain, "topology_version") else 1,
            "task_profile": env.get_task_profile()
        }
        publish_event(event_bus, run_id, "LEVEL_COMPLETED", family, global_step, snapshot)
        
        results["task_outcomes"][task_id] = {
            "converged": level_converged,
            "steps": training_step,
            "performance": eval_history_list[-1] if len(eval_history_list)>0 else 0
        }
        
        print(f"[EVOLVE][LEVEL_END] L{human_level:02d} perf={eval_history_list[-1] if len(eval_history_list)>0 else 0.0:.3f} retention={final_retention:.3f} forgetting={final_forgetting:.3f} plasticity={last_plasticity_auc:.3f} ood={ood_shift_val:.3f} recovery={f'{ood_recovery:.3f}' if ood_recovery is not None else 'NA'} modules={len(brain.registry.list_active_ids())} params={brain.parameter_count()} rev={brain.topology_revision}")

    print("Experiment Completed Successfully!")
    publish_event(event_bus, run_id, "RUN_COMPLETED", "COMPLETE", global_step, results)
    return results

def main():
    parser = argparse.ArgumentParser(description="EVOLVE Experiment Runner")
    parser.add_argument("--families", nargs="+", default=["HierarchicalNavigation"], help="Environment families to cycle through")
    parser.add_argument("--levels", type=int, default=3, help="Number of levels per family")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-freq", type=int, default=500)
    args = parser.parse_args()
    
    # Run in CLI mode without an EventBus
    try:
        run_continuous_experiment(
        run_id="CLI_RUN_001",
        event_bus=None,
        num_levels=args.levels,
        seed=args.seed,
        max_steps=args.max_steps,
        eval_freq=args.eval_freq
        )
    except Exception as e:
        print(f"Run failed: {e}")

if __name__ == "__main__":
    main()
