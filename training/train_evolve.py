import torch
import numpy as np
from typing import Dict, Any, List, Optional

from brain.brain import ModularBrain
from evolution.controller import EvolutionController
from environment.scenarios import EnvironmentA, EnvironmentB
from environment.evaluation import IsolatedEvaluator
from training.agent import DQNAgent
from training.metrics import MetricsCollector, MetricPoint
from backend.events import EventBus, Event

def run_evolve_experiment(run_id: str, steps_a: int = 3000, steps_b: int = 3000, seed: int = 42, event_bus: EventBus = None) -> Dict[str, Any]:
    print(f"[EVOLVE System] Starting evidence-driven A -> B -> A run (seed={seed})...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_a = EnvironmentA()
    env_b = EnvironmentB()
    evaluator = IsolatedEvaluator(num_episodes=10, seed=seed)

    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=4, initial_module_id="module_A")
    controller = EvolutionController()
    agent = DQNAgent(brain, learning_rate=0.001)
    collector = MetricsCollector()

    mutation_log: List[Dict[str, Any]] = []

    def publish_topology(phase: str, step: int):
        if event_bus:
            snap = brain.topology_snapshot()
            event_bus.publish(Event(
                event_type="TOPOLOGY_SNAPSHOT",
                run_id=run_id,
                step=step,
                phase=phase,
                payload=snap
            ))

    def publish_env(phase: str, step: int, env_obj, info: dict):
        if event_bus:
            payload = {
                "grid_size": env_obj.grid_size,
                "agent_pos": info.get("agent_pos", env_obj.agent_pos),
                "goal_pos": info.get("goal_pos", env_obj.goal_pos),
                "obstacles": info.get("obstacles", list(env_obj.obstacles)),
                "hazards": info.get("hazards", list(env_obj.hazards) if hasattr(env_obj, 'hazards') else []),
            }
            event_bus.publish(Event(
                event_type="ENVIRONMENT_STATE_UPDATE",
                run_id=run_id,
                step=step,
                phase=phase,
                payload=payload
            ))

    def publish_metrics(phase: str, step: int, reward: float, perf: float):
        if event_bus:
            event_bus.publish(Event(
                event_type="METRICS_UPDATE",
                run_id=run_id,
                step=step,
                phase=phase,
                payload={
                    "step": step,
                    "phase": phase,
                    "reward": reward,
                    "performance": perf,
                }
            ))
            
    def publish_decision(phase: str, step: int, decision):
        if event_bus:
            event_bus.publish(Event(
                event_type="DECISION_MADE",
                run_id=run_id,
                step=step,
                phase=phase,
                payload={
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
                    "cooldown_active": decision.cooldown_active
                }
            ))

    # Initial snapshot
    publish_topology("A", 0)

    # ===== Phase 1: Train on Environment A =====
    print("[EVOLVE System] Phase A: Training on Environment A...")
    obs, info = env_a.reset(seed=seed)
    publish_env("A", 0, env_a, info)
    
    recent_rewards: List[float] = []
    episode_rewards: List[float] = []
    ep_reward = 0.0

    for step in range(steps_a):
        action, _ = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env_a.step(action)
        agent.buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        recent_rewards.append(reward)
        ep_reward += reward
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # Publish environment state every few steps to avoid flooding if it runs fast
        if step % 5 == 0:
            publish_env("A", step, env_a, info)
            avg_r = np.mean(recent_rewards) if recent_rewards else 0.0
            publish_metrics("A", step, reward, avg_r)

        if terminated or truncated:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, info = env_a.reset()
            publish_env("A", step, env_a, info)
            
        agent.update()

        if step > 0 and step % 500 == 0:
            avg = np.mean(episode_rewards[-5:]) if episode_rewards else 0.0
            print(f"  [Phase A] Step {step}, Avg reward (last 5 eps): {avg:.3f}, Epsilon: {agent.epsilon:.3f}")

    eval_a_before = evaluator.evaluate(brain, env_a)
    print(f"[EVOLVE System] Pre-B Environment A Score: {eval_a_before['mean_reward']:.3f} "
          f"(Success rate: {eval_a_before['success_rate']:.1%})")

    # ===== Phase 2: Train on Environment B with Controller =====
    print("[EVOLVE System] Phase B: Training on Environment B with Evolution Controller...")
    obs, info = env_b.reset(seed=seed + 1)
    publish_env("B", steps_a, env_b, info)
    
    growth_counter = 0
    episode_rewards_b: List[float] = []
    ep_reward = 0.0
    recent_rewards = []
    last_eval_a_score = eval_a_before["mean_reward"]

    for step in range(steps_b):
        global_step = steps_a + step
        action, routing_weights = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env_b.step(action)
        agent.buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        recent_rewards.append(reward)
        ep_reward += reward
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        if global_step % 5 == 0:
            publish_env("B", global_step, env_b, info)
            avg_r = np.mean(recent_rewards) if recent_rewards else 0.0
            publish_metrics("B", global_step, reward, avg_r)

        if terminated or truncated:
            episode_rewards_b.append(ep_reward)
            ep_reward = 0.0
            obs, info = env_b.reset()
            publish_env("B", global_step, env_b, info)

        loss = agent.update()

        if step > 0 and step % 200 == 0:
            current_perf = float(np.mean(recent_rewards[-50:])) if recent_rewards else 0.0
            task_prof_dist = 0.75

            if routing_weights is not None and routing_weights.dim() >= 1:
                utilization = routing_weights.squeeze().tolist()
                if isinstance(utilization, float):
                    utilization = [utilization]
            else:
                utilization = [1.0]

            if step % 600 == 0:
                eval_a_current = evaluator.evaluate(brain, env_a)
                last_eval_a_score = eval_a_current["mean_reward"]

            decision = controller.evaluate_step(
                current_performance=current_perf,
                target_performance=0.5,
                recent_history=recent_rewards[-30:],
                task_profile_distance=task_prof_dist,
                interference_value=0.3,
                module_utilizations=utilization,
                current_param_count=brain.parameter_count(),
                max_param_budget=50000,
                current_step=global_step,
                active_module_count=len(brain.registry.list_active_ids()),
            )
            
            publish_decision("B", global_step, decision)

            avg_b = np.mean(episode_rewards_b[-5:]) if episode_rewards_b else 0.0
            print(f"  [Phase B] Step {global_step} -> Action: {decision.action}, "
                  f"Need Score: {decision.need_score:.3f}, Avg B reward: {avg_b:.3f}")

            if decision.action == "GROW":
                growth_counter += 1
                mod_name = f"module_{chr(65 + growth_counter)}"
                print(f"  [EVOLVE System] Executing evidence-driven GROW: adding {mod_name}")

                ckpt = controller.rollback_manager.capture_checkpoint(brain, agent.optimizer, {})
                brain.add_module(mod_name, created_step=global_step, origin_event="GROW")
                agent.rebuild_optimizer()
                controller.policy.record_mutation_outcome(current_step=global_step, action="GROW", accepted=True)
                
                # Publish the new topology immediately so the UI animates the addition
                publish_topology("B", global_step)

                for v_step in range(150):
                    v_action, _ = agent.select_action(obs)
                    v_next_obs, v_reward, v_term, v_trunc, v_info = env_b.step(v_action)
                    agent.buffer.push(obs, v_action, v_reward, v_next_obs, float(v_term))
                    obs = v_next_obs
                    if v_term or v_trunc:
                        obs, v_info = env_b.reset()
                    agent.update()

                eval_b_temp = evaluator.evaluate(brain, env_b)
                eval_a_temp = evaluator.evaluate(brain, env_a)

                verif = controller.verifier.evaluate(
                    new_task_score_before=current_perf,
                    new_task_score_after=eval_b_temp["mean_reward"],
                    old_task_score_before=eval_a_before["mean_reward"],
                    old_task_score_after=eval_a_temp["mean_reward"],
                    structural_cost=brain.parameter_count() / 50000.0,
                )

                mutation_entry = {
                    "module": mod_name,
                    "step": global_step,
                    "need_score": decision.need_score,
                    "accepted": verif.accepted,
                    "reason": verif.reason,
                    "b_score_after": eval_b_temp["mean_reward"],
                    "a_retention": eval_a_temp["mean_reward"],
                }

                if verif.accepted:
                    print(f"  [EVOLVE System] Mutation {mod_name} ACCEPTED: {verif.reason}")
                    last_eval_a_score = eval_a_temp["mean_reward"]
                else:
                    print(f"  [EVOLVE System] Mutation {mod_name} REJECTED: {verif.reason}. Rolling back...")
                    brain, restored_optimizer = controller.rollback_manager.restore_checkpoint(
                        brain, agent.optimizer, ckpt
                    )
                    agent.optimizer = restored_optimizer
                    agent.rebuild_optimizer()
                    # Revert topology in UI
                    publish_topology("B", global_step)

                mutation_log.append(mutation_entry)

            elif decision.action == "PRUNE":
                active = brain.registry.list_active_ids()
                if len(active) > 1:
                    least_used = active[-1]
                    print(f"  [EVOLVE System] PRUNE: deactivating {least_used}")
                    brain.remove_module(least_used)
                    agent.rebuild_optimizer()
                    controller.policy.record_mutation_applied(current_step=global_step, action="PRUNE")
                    publish_topology("B", global_step)

    # ===== Phase 3: Final Evaluation =====
    eval_b_after = evaluator.evaluate(brain, env_b)
    eval_a_after = evaluator.evaluate(brain, env_a)

    retention_ratio = eval_a_after["mean_reward"] / max(eval_a_before["mean_reward"], 1e-5)
    return {
        "experiment": "evolve",
        "a_before": eval_a_before["mean_reward"],
        "b_after": eval_b_after["mean_reward"],
        "a_after": eval_a_after["mean_reward"],
        "a_success_before": eval_a_before["success_rate"],
        "b_success_after": eval_b_after["success_rate"],
        "a_success_after": eval_a_after["success_rate"],
        "retention_ratio": retention_ratio,
        "parameter_count": brain.parameter_count(),
        "active_modules": len(brain.registry.list_active_ids()),
        "mutations": mutation_log,
    }
