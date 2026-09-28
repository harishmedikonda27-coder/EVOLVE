import copy
from typing import Dict, Any, Optional, Tuple, Callable
import torch

class TransactionResult:
    def __init__(self, status: str, v_perf: float, action: str, record: Any, new_obs: Any = None,
                 new_global_step: int = 0, new_training_step: int = 0, stopped: bool = False):
        self.status = status
        self.v_perf = v_perf
        self.action = action
        self.record = record
        self.new_obs = new_obs
        self.new_global_step = new_global_step
        self.new_training_step = new_training_step
        self.stopped = stopped

class TransactionRunner:
    def __init__(self, mut_manager, rollback_manager, verifier, event_bus, run_id):
        self.mut_manager = mut_manager
        self.rollback = rollback_manager
        self.verifier = verifier
        self.event_bus = event_bus
        self.run_id = run_id

    def publish_event(self, event_type: str, phase: str, step: int, payload: Dict[str, Any]):
        from backend.events import Event
        if self.event_bus:
            self.event_bus.publish(Event(
                event_type=event_type,
                run_id=self.run_id,
                step=step,
                phase=phase,
                payload=payload
            ))

    def run_transaction(
        self,
        decision,
        brain,
        agent,
        env,
        obs,
        tasks,
        controller,
        global_step: int,
        training_step: int,
        task_id: str,
        family: str,
        before_scores: Dict[str, float],
        min_r: float,
        max_r: float,
        seed: int,
        eval_env: Any,
        control,
        create_environment_fn: Callable,
        evaluate_agent_fn: Callable,
        validate_system_state_fn: Callable,
        norm_perf: float,
        ckpt_manager,
        results: Dict[str, Any],
        verification_budget: int = 500
    ) -> TransactionResult:
        
        param_count_before = sum(p.numel() for p in brain.parameters())
        topology_before = brain.topology_snapshot()
        active_modules_before = len(brain.registry.list_active_ids())
        
        mut_record = self.mut_manager.create_mutation_record(
            operation=decision.action,
            level_id=tasks.tasks[task_id].level,
            task_profile=env.get_task_profile(),
            need_score=decision.need_score,
            component_scores={
                "performance_gap": decision.signals.performance_gap,
                "stagnation": decision.signals.stagnation,
                "novelty": decision.signals.novelty,
                "interference": decision.signals.interference,
                "utilization": decision.signals.utilization,
                "cost": decision.signals.cost,
            },
            reason=decision.reason,
            topology_before=topology_before,
            parameter_count_before=param_count_before,
            budget_before=decision.budget_remaining
        )
        mut_record.new_task_metric_before = norm_perf
        
        # We need after_scores from just before mutation for retention_metric_before
        after_scores = {}
        for past_id in tasks.tasks.keys():
            if past_id == task_id:
                continue
            past_env = create_environment_fn(tasks.tasks[past_id].family, tasks.tasks[past_id].level, tasks.tasks[past_id].seed)
            raw_p, _ = evaluate_agent_fn(agent, past_env)
            past_profile = past_env.get_task_profile()
            past_reward_spec = past_profile["reward_spec"]
            min_r_past, max_r_past = past_reward_spec["R_min"], past_reward_spec["R_max"]
            after_scores[past_id] = max(0.0, min(1.0, (raw_p - min_r_past) / (max_r_past - min_r_past)))

        mut_record.retention_metric_before = sum(after_scores.values()) / len(after_scores) if after_scores else 1.0
        mut_record.verification_steps = verification_budget
        ckpt_state = self.rollback.capture_checkpoint(brain, agent.optimizer, controller.__dict__)
        buffer_snapshot = list(agent.buffer.buffer)
        epsilon_snapshot = agent.epsilon
        update_count_snapshot = agent.update_count
        
        try:
            if decision.action == "GROW":
                self.mut_manager.execute_grow(brain, global_step)
            elif decision.action == "PRUNE":
                target = brain.registry.list_active_ids()[-1]
                self.mut_manager.execute_prune(brain, target)
            elif decision.action == "REWIRE":
                structural_change = self.mut_manager.execute_rewire(brain)
                if not structural_change:
                    print(f"[EVOLVE][ROLLBACK] L{tasks.tasks[task_id].level + 1:02d} action=REWIRE reason=\"REWIRE_INFEASIBLE: No valid rewire exists.\"")
                    self.publish_event("MUTATION_REJECTED", family, global_step, {"action": decision.action, "reason": "REWIRE_INFEASIBLE"})
                    return TransactionResult(status="INFEASIBLE", v_perf=0.0, action=decision.action, record=mut_record, new_obs=obs, new_global_step=global_step, new_training_step=training_step)
                
            agent.rebuild_optimizer()
            
            
            # If REWIRE, compute edges diff
            if decision.action == "REWIRE":
                from evolution.mutation import compute_rewire_diff
                mut_record.rewire_diff = compute_rewire_diff(topology_before, brain.topology_snapshot(), mut_record.mutation_id)
                if mut_record.rewire_diff.added_edge_count == 0 and mut_record.rewire_diff.removed_edge_count == 0:
                    print(f"[EVOLVE][ROLLBACK] L{tasks.tasks[task_id].level + 1:02d} action=REWIRE reason=\"REWIRE_INFEASIBLE: Zero delta mask change.\"")
                    
                    # Restore state immediately without consuming budget
                    brain, agent.optimizer = self.rollback.restore_checkpoint(brain, agent.optimizer, ckpt_state)
                    agent.target_brain.load_state_dict(brain.state_dict())
                    agent.buffer.buffer.clear()
                    agent.buffer.buffer.extend(buffer_snapshot)
                    agent.epsilon = epsilon_snapshot
                    agent.update_count = update_count_snapshot
                    
                    self.publish_event("MUTATION_REJECTED", family, global_step, {"action": decision.action, "reason": "REWIRE_INFEASIBLE"})
                    return TransactionResult(status="INFEASIBLE", v_perf=0.0, action=decision.action, record=mut_record, new_obs=obs, new_global_step=global_step, new_training_step=training_step)
            
            rewire_add = mut_record.rewire_diff.added_edge_count if mut_record.rewire_diff else 0
            rewire_rm = mut_record.rewire_diff.removed_edge_count if mut_record.rewire_diff else 0
            
            print(f"[EVOLVE][MUTATION] L{tasks.tasks[task_id].level + 1:02d} S{global_step} action={decision.action} modules={active_modules_before}->{len(brain.registry.list_active_ids())} params={param_count_before}->{sum(p.numel() for p in brain.parameters())} rev={topology_before['topology_revision']}->{brain.topology_revision} add_edges={rewire_add} remove_edges={rewire_rm}")
            
            self.publish_event("MUTATION_VERIFYING", family, global_step, {
                "action": decision.action,
                "level": tasks.tasks[task_id].level + 1,
                "task_id": task_id,
                "parameters_before": param_count_before,
                "parameters_after": sum(p.numel() for p in brain.parameters()),
                "modules_before": active_modules_before,
                "modules_after": len(brain.registry.list_active_ids()),
                "topology_revision_before": topology_before["topology_revision"],
                "topology_revision_after": brain.topology_revision,
                "new_task_before": norm_perf,
                "new_task_after": None,
                "retention_before": mut_record.retention_metric_before,
                "retention_after": None,
                "structural_cost": decision.signals.cost,
                "rewire_added_edges": rewire_add,
                "rewire_removed_edges": rewire_rm
            })
            
            validate_system_state_fn(agent, label="POST_MUTATION")
            
            snap = brain.topology_snapshot()
            if decision.action == "REWIRE" and mut_record.rewire_diff:
                snap["rewire_diff"] = mut_record.rewire_diff.to_dict()
                
            self.publish_event("TOPOLOGY_SNAPSHOT", family, global_step, snap)
            
            self.publish_event("TOPOLOGY_HISTORY_ENTRY", family, global_step, {
                "label": f"{decision.action} · BEFORE",
                "level_id": task_id,
                "mutation_type": decision.action,
                "mutation_status": "BEFORE",
                "event_type": "MUTATION_VERIFYING",
                "topology_revision": brain.topology_revision,
                "parameter_count": param_count_before,
                "active_module_count": len(brain.registry.list_active_ids()),
                "topology_snapshot": topology_before
            })
            
            # Verify for 500 steps
            verification_step = 0
            for _ in range(verification_budget):
                if control.stop_requested.is_set():
                    print(f"[{self.run_id}] Stop requested during verification. Aborting mutation and halting run.")
                    brain, agent.optimizer = self.rollback.restore_checkpoint(brain, agent.optimizer, ckpt_state)
                    agent.target_brain = copy.deepcopy(brain)
                    agent.target_brain.eval()
                    controller.__dict__.update(ckpt_state["controller_state"])
                    
                    validate_system_state_fn(agent, label="POST_ROLLBACK_STOP")
                    
                    self.publish_event("RUN_STOPPING", family, global_step, {})
                    ckpt_id = f"ckpt_{task_id}_stopped"
                    ckpt_manager.save_checkpoint(ckpt_id, brain, agent.optimizer, {"task": task_id})
                    self.publish_event("RUN_STOPPED", "COMPLETE", global_step, results)
                    
                    return TransactionResult(status="STOPPED", v_perf=0.0, action=decision.action, record=mut_record, stopped=True)

                a, _ = agent.select_action(obs, evaluate=False)
                nobs, r, tm, tr, _ = env.step(a)
                agent.buffer.push(obs, a, r, nobs, tm)
                agent.update()
                obs = nobs if not (tm or tr) else env.reset()[0]
                global_step += 1
                verification_step += 1
                
            print(f"[{task_id}] (Trained for {verification_budget} verification steps)")
                
            v_raw, _ = evaluate_agent_fn(agent, eval_env)
            v_perf = max(0.0, min(1.0, (v_raw - min_r) / (max_r - min_r)))

            # Evaluate old tasks
            after_scores_post_verif = {}
            if len(before_scores) > 0:
                for past_id in before_scores.keys():
                    past_env = create_environment_fn(tasks.tasks[past_id].family, tasks.tasks[past_id].level, tasks.tasks[past_id].seed)
                    raw_p, _ = evaluate_agent_fn(agent, past_env)
                    past_profile = past_env.get_task_profile()
                    past_reward_spec = past_profile["reward_spec"]
                    min_r_past, max_r_past = past_reward_spec["R_min"], past_reward_spec["R_max"]
                    after_scores_post_verif[past_id] = max(0.0, min(1.0, (raw_p - min_r_past) / (max_r_past - min_r_past)))
                old_task_score_before = sum(after_scores.values()) / len(after_scores)
                old_task_score_after = sum(after_scores_post_verif.values()) / len(after_scores_post_verif)
            else:
                old_task_score_before = 1.0
                old_task_score_after = 1.0

            v_res = self.verifier.evaluate(norm_perf, v_perf, old_task_score_before, old_task_score_after, decision.signals.cost)
            
            mut_record.topology_after = brain.topology_snapshot()
            mut_record.parameter_count_after = sum(p.numel() for p in brain.parameters())
            mut_record.new_task_metric_after = v_perf
            mut_record.retention_metric_after = old_task_score_after
            
            
            print(f"[EVOLVE][VERIFY] L{tasks.tasks[task_id].level + 1:02d} action={decision.action} task={norm_perf:.3f}->{v_perf:.3f} gain={v_perf - norm_perf:.3f} retention={old_task_score_before:.3f}->{old_task_score_after:.3f} cost={decision.signals.cost:.3f} result={'ACCEPT' if v_res.accepted else 'REJECT'}")
            
            if v_res.accepted:
                # [REPLAY SEMANTICS - ACCEPTED]
                # Because the structural mutation is verified and committed, the 500 transitions
                # collected during verification using the new topology are retained in the replay buffer.
                # These transitions are valid on-policy/off-policy experience for the committed architecture.
                # Rejected branches have their replay buffers correctly cleared and rolled back below.
                controller.policy.record_mutation_outcome(global_step, decision.action, accepted=True)
                mut_record.status = "ACCEPT"
                
                self.publish_event("MUTATION_ACCEPTED", family, global_step, {
                    "action": decision.action,
                    "level": tasks.tasks[task_id].level + 1,
                    "task_id": task_id,
                    "performance_gain": v_perf - norm_perf,
                    "retention": old_task_score_after,
                    "structural_cost": decision.signals.cost,
                    "modules_before": active_modules_before,
                    "modules_after": len(brain.registry.list_active_ids()),
                    "parameters_before": param_count_before,
                    "parameters_after": mut_record.parameter_count_after,
                    "topology_revision_before": topology_before["topology_revision"],
                    "topology_revision_after": brain.topology_revision
                })
                self.publish_event("TOPOLOGY_HISTORY_ENTRY", family, global_step, {
                    "label": f"{decision.action} · ACCEPTED",
                    "level_id": task_id,
                    "mutation_type": decision.action,
                    "mutation_status": "ACCEPTED",
                    "event_type": "MUTATION_ACCEPTED",
                    "topology_revision": brain.topology_revision,
                    "parameter_count": mut_record.parameter_count_after,
                    "active_module_count": len(brain.registry.list_active_ids()),
                    "topology_snapshot": mut_record.topology_after
                })
                return TransactionResult(status="ACCEPT", v_perf=v_perf, action=decision.action, record=mut_record, new_obs=obs, new_global_step=global_step, new_training_step=training_step)
            else:
                mut_record.status = "ROLLBACK"
                mut_record.rollback_reason = v_res.reason
                self.publish_event("MUTATION_REJECTED", family, global_step, {"action": decision.action, "reason": v_res.reason})
                
                self.publish_event("ROLLBACK", family, global_step, {
                    "action": decision.action,
                    "level": tasks.tasks[task_id].level + 1,
                    "task_id": task_id,
                    "rollback_reason": v_res.reason,
                    "modules_before": active_modules_before,
                    "modules_after_attempt": len(brain.registry.list_active_ids()),
                    "modules_after_restore": active_modules_before,
                    "parameters_before": param_count_before,
                    "parameters_after_attempt": mut_record.parameter_count_after,
                    "parameters_after_restore": param_count_before,
                    "topology_revision_before": topology_before["topology_revision"],
                    "topology_revision_restored": topology_before["topology_revision"],
                    "verification_result": v_res.accepted
                })
                print(f"[EVOLVE][ROLLBACK] L{tasks.tasks[task_id].level + 1:02d} action={decision.action} reason=\"{v_res.reason}\" modules={active_modules_before} params={param_count_before} rev={topology_before['topology_revision']}")
                
                brain, agent.optimizer = self.rollback.restore_checkpoint(brain, agent.optimizer, ckpt_state)
                agent.target_brain = copy.deepcopy(brain)
                agent.target_brain.eval()
                controller.__dict__.update(ckpt_state["controller_state"])
                
                # Record the failure on the restored controller policy so it is not erased
                controller.policy.record_mutation_outcome(global_step, decision.action, accepted=False)
                
                # Restore agent state (buffer, epsilon, update_count)
                agent.buffer.buffer.clear()
                agent.buffer.buffer.extend(buffer_snapshot)
                agent.epsilon = epsilon_snapshot
                agent.update_count = update_count_snapshot
                
                validate_system_state_fn(agent, label="POST_ROLLBACK_REJECT")
                
                self.publish_event("TOPOLOGY_SNAPSHOT", family, global_step, brain.topology_snapshot())
                
                self.publish_event("TOPOLOGY_HISTORY_ENTRY", family, global_step, {
                    "label": f"{decision.action} · ROLLED BACK",
                    "level_id": task_id,
                    "mutation_type": decision.action,
                    "mutation_status": "ROLLED_BACK",
                    "event_type": "ROLLBACK",
                    "topology_revision": brain.topology_revision,
                    "parameter_count": sum(p.numel() for p in brain.parameters()),
                    "active_module_count": len(brain.registry.list_active_ids()),
                    "topology_snapshot": brain.topology_snapshot()
                })
                
                return TransactionResult(status="ROLLBACK", v_perf=v_perf, action=decision.action, record=mut_record, new_obs=obs, new_global_step=global_step, new_training_step=training_step)
                
        except Exception as e:
            controller.policy.record_mutation_outcome(global_step, decision.action, accepted=False)
            mut_record.status = "ROLLBACK"
            mut_record.rollback_reason = str(e)
            self.publish_event("MUTATION_REJECTED", family, global_step, {"action": decision.action, "reason": str(e)})
            self.publish_event("ROLLBACK", family, global_step, {
                "action": decision.action,
                "level": tasks.tasks[task_id].level + 1,
                "task_id": task_id,
                "rollback_reason": str(e),
                "modules_before": active_modules_before,
                "modules_after_attempt": len(brain.registry.list_active_ids()),
                "modules_after_restore": active_modules_before,
                "parameters_before": param_count_before,
                "parameters_after_attempt": sum(p.numel() for p in brain.parameters()),
                "parameters_after_restore": param_count_before,
                "topology_revision_before": topology_before["topology_revision"],
                "topology_revision_restored": topology_before["topology_revision"],
                "verification_result": False
            })
            print(f"[EVOLVE][ROLLBACK] L{tasks.tasks[task_id].level + 1:02d} action={decision.action} reason=\"{str(e)}\" modules={active_modules_before} params={param_count_before} rev={topology_before['topology_revision']}")
            
            brain, agent.optimizer = self.rollback.restore_checkpoint(brain, agent.optimizer, ckpt_state)
            agent.target_brain = copy.deepcopy(brain)
            agent.target_brain.eval()
            controller.__dict__.update(ckpt_state["controller_state"])
            
            # Restore replay buffer
            agent.buffer.buffer.clear()
            agent.buffer.buffer.extend(buffer_snapshot)
            
            validate_system_state_fn(agent, label="POST_ROLLBACK_ERROR")
            
            self.publish_event("TOPOLOGY_SNAPSHOT", family, global_step, brain.topology_snapshot())
            
            self.publish_event("TOPOLOGY_HISTORY_ENTRY", family, global_step, {
                "label": f"{decision.action} · ROLLED BACK",
                "level_id": task_id,
                "mutation_type": decision.action,
                "mutation_status": "ROLLED_BACK",
                "event_type": "ROLLBACK",
                "topology_revision": brain.topology_revision,
                "parameter_count": sum(p.numel() for p in brain.parameters()),
                "active_module_count": len(brain.registry.list_active_ids()),
                "topology_snapshot": brain.topology_snapshot()
            })
            
            return TransactionResult(status="ERROR", v_perf=0.0, action=decision.action, record=mut_record, new_obs=obs, new_global_step=global_step, new_training_step=training_step)
