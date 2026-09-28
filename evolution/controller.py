from typing import Dict, Any, List, Optional
from evolution.signals import SignalCalculator, EvolutionSignals
from evolution.scorer import NeedScorer
from evolution.policies import DecisionPolicy, EvolutionDecision
from evolution.mutation import MutationManager, MutationRecord
from evolution.verify import Verifier, VerificationResult
from evolution.rollback import RollbackManager
from evolution.state import AdaptationState
import torch

class EvolutionController:
    def __init__(
        self,
        signal_calculator: Optional[SignalCalculator] = None,
        scorer: Optional[NeedScorer] = None,
        policy: Optional[DecisionPolicy] = None,
        verifier: Optional[Verifier] = None,
        learned_policy = None,
        adaptation_mode: str = "deterministic"
    ):
        self.signal_calculator = signal_calculator or SignalCalculator()
        self.scorer = scorer or NeedScorer()
        self.policy = policy or DecisionPolicy()
        self.verifier = verifier or Verifier()
        self.mutation_manager = MutationManager()
        self.rollback_manager = RollbackManager()
        self.learned_policy = learned_policy
        self.adaptation_mode = adaptation_mode

    def evaluate_step(
        self,
        current_performance: float,
        target_performance: float = 1.0,
        recent_history: Optional[List[float]] = None,
        task_profile_distance: float = 0.0,
        interference_value: float = 0.0,
        module_utilizations: Optional[List[float]] = None,
        current_param_count: int = 1000,
        max_param_budget: int = 10000,
        current_step: int = 0,
        active_module_count: int = 1,
        youngest_module_age: int = 999999,
        ood_shift: float = 0.0,
        plasticity_auc: float = 0.0,
        adaptation_state_out: Optional[List[AdaptationState]] = None,
        brain: Optional[Any] = None
    ) -> EvolutionDecision:
        signals = self.signal_calculator.compute(
            current_performance=current_performance,
            target_performance=target_performance,
            recent_history=recent_history,
            task_profile_distance=task_profile_distance,
            interference_value=interference_value,
            module_utilizations=module_utilizations,
            current_param_count=current_param_count,
            max_param_budget=max_param_budget,
        )

        need_score = self.scorer.compute(signals)

        decision = self.policy.evaluate(
            need_score=need_score,
            signals=signals,
            current_step=current_step,
            active_module_count=active_module_count,
            current_parameters=current_param_count,
            youngest_module_age=youngest_module_age,
            brain=brain
        )
        
        # Phase 2: Construct AdaptationState
        util_mean = float(sum(module_utilizations)/len(module_utilizations)) if module_utilizations else 0.0
        util_var = float(sum((x - util_mean)**2 for x in module_utilizations)/len(module_utilizations)) if module_utilizations else 0.0
        
        state = AdaptationState(
            current_performance=current_performance,
            stagnation_slope=signals.stagnation,
            novelty=signals.novelty,
            ood_shift=ood_shift,
            interference=signals.interference,
            plasticity_auc=plasticity_auc,
            utilization_mean=util_mean,
            utilization_variance=util_var,
            module_count=active_module_count,
            parameter_count=current_param_count,
            budget_remaining=decision.budget_remaining,
            youngest_module_age=youngest_module_age
        )
        if adaptation_state_out is not None:
            adaptation_state_out.append(state)
            
        decision.adaptation_state = state

        if self.adaptation_mode == "random":
            # Phase 2: Random Valid Action Baseline
            from evolution.learned_policy import FeasibilityMask
            import random
            
            mask_generator = FeasibilityMask()
            mask = mask_generator.get_mask(active_module_count, decision.budget_remaining)
            feasible_indices = torch.nonzero(mask).squeeze(-1).tolist()
            if not feasible_indices:
                random_action = "REUSE"
            else:
                action_map = {0: "REUSE", 1: "ADAPT", 2: "GROW", 3: "PRUNE", 4: "REWIRE"}
                action_idx = random.choice(feasible_indices)
                random_action = action_map.get(action_idx, "REUSE")
                
            # Random valid action baseline completely overrides the deterministic/legacy policy if not on cooldown
            if not decision.cooldown_active:
                decision.action = random_action
                decision.reason = f"RandomValidAction baseline selected {random_action}"

        elif self.learned_policy is not None and self.adaptation_mode in ["learned", "shadow"]:
            # Query LearnedPolicy
            state_vec = torch.tensor(state.to_vector(), dtype=torch.float32)
            
            with torch.no_grad():
                device = next(self.learned_policy.parameters()).device
                q_tensor = self.learned_policy(state_vec.to(device)).squeeze(0)
                decision.q_values = {self.learned_policy.action_map[i]: float(q_tensor[i].item()) for i in range(5)}

            epsilon = 0.1 if self.adaptation_mode == "learned" else 0.0
            learned_action = self.learned_policy.select_action(
                state_vec, 
                module_count=active_module_count, 
                budget_remaining=decision.budget_remaining,
                epsilon=epsilon
            )
            
            # In shadow mode, we just log. In learned mode, we override the action.
            if self.adaptation_mode == "learned":
                if not decision.cooldown_active:
                    decision.action = learned_action
                    decision.reason = f"LearnedPolicy recommended {learned_action}"

        return decision
