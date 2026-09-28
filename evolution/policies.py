from dataclasses import dataclass
from typing import Dict, Any, Optional
from evolution.signals import EvolutionSignals


@dataclass
class EvolutionDecision:
    action: str                        # REUSE, ADAPT, GROW, PRUNE, REWIRE
    need_score: float
    signals: EvolutionSignals
    reason: str
    budget_remaining: int
    cooldown_active: bool
    eligible_actions: list = None
    blocked_actions: list = None
    blocked_reasons: Dict[str, str] = None
    adaptation_state: Optional[Any] = None
    q_values: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "action": self.action,
            "need_score": self.need_score,
            "signals": self.signals.to_dict(),
            "reason": self.reason,
            "budget_remaining": self.budget_remaining,
            "cooldown_active": self.cooldown_active,
            "eligible_actions": self.eligible_actions or [],
            "blocked_actions": self.blocked_actions or [],
            "blocked_reasons": self.blocked_reasons or {},
        }
        if self.adaptation_state is not None:
            d["adaptation_state"] = self.adaptation_state.to_dict()
        if self.q_values is not None:
            d["q_values"] = self.q_values
        return d


class DecisionPolicy:
    def __init__(
        self,
        adapt_threshold: float = 0.25,
        grow_threshold: float = 0.55,
        prune_threshold: float = 0.15,
        rewire_threshold: float = 0.40,
        growth_budget_per_level: int = 2,
        total_growth_budget: int = 10,
        cooldown_steps: int = 200,
        min_module_lifetime: int = 500,
        max_modules: int = 10,
        max_parameters: int = 100000,
        need_score_delta_threshold: float = 0.05,
    ):
        self.adapt_threshold = adapt_threshold
        self.grow_threshold = grow_threshold
        self.prune_threshold = prune_threshold
        self.rewire_threshold = rewire_threshold
        self.growth_budget_per_level = growth_budget_per_level
        self.total_growth_budget = total_growth_budget
        
        self.cooldown_steps = cooldown_steps
        self.min_module_lifetime = min_module_lifetime
        self.max_modules = max_modules
        self.max_parameters = max_parameters
        self.need_score_delta_threshold = need_score_delta_threshold

        self.level_growth_count: int = 0
        self.total_growth_count: int = 0
        self.last_mutation_step: int = -999999
        self.last_failed_mutation_step: int = -999999
        self.failed_mutations: Dict[str, Dict[str, Any]] = {}  # Tracks {action: {"need_score": float, "fingerprint": str}}
        self._pending_need_score: float = 0.0
        self._pending_fingerprint: str = ""

    def reset_level_stats(self):
        self.level_growth_count = 0

    def evaluate(
        self,
        need_score: float,
        signals: EvolutionSignals,
        current_step: int = 0,
        active_module_count: int = 1,
        current_parameters: int = 0,
        youngest_module_age: int = 999999,
        brain: Optional[Any] = None
    ) -> EvolutionDecision:
        sig_dict = signals.to_dict()
        budget_remaining = max(0, self.growth_budget_per_level - self.level_growth_count)
        global_budget_remaining = max(0, self.total_growth_budget - self.total_growth_count)
        
        time_since_mutation = current_step - self.last_mutation_step
        time_since_failed = current_step - self.last_failed_mutation_step
        
        cooldown_active = (time_since_mutation < self.cooldown_steps) or (time_since_failed < (self.cooldown_steps * 2))

        current_fingerprint = f"m{active_module_count}_p{current_parameters}"

        # Engineering Heuristic: 0.05 NeedScore margin & Fingerprinting
        # If the structural state (modules + parameters) is identical to when the mutation last failed,
        # it will likely fail again. We require EITHER a different structural fingerprint OR a massive
        # new evidence surge (need_score) to try again.
        def is_repeated_failure(proposed_action: str) -> bool:
            if proposed_action in self.failed_mutations:
                failed_data = self.failed_mutations[proposed_action]
                if failed_data["fingerprint"] == current_fingerprint:
                    # If the structural state is exactly the same, block it unless there's a massive shift
                    if (need_score - failed_data["need_score"]) < self.need_score_delta_threshold:
                        return True
                    # Even if need_score increased, if it's the exact same state, we should probably block it
                    # but we allow a 0.05 margin to provide an "escape hatch" in case the environment radically changed.
            return False

        eligible_actions = ["REUSE"] # Always eligible
        blocked_actions = []
        blocked_reasons = {}

        if cooldown_active:
            blocked_actions.extend(["GROW", "PRUNE", "REWIRE"])
            blocked_reasons["GROW"] = "cooldown active"
            blocked_reasons["PRUNE"] = "cooldown active"
            blocked_reasons["REWIRE"] = "cooldown active"
        else:
            # GROW logic
            if budget_remaining <= 0 or global_budget_remaining <= 0:
                blocked_actions.append("GROW")
                blocked_reasons["GROW"] = "budget exhausted"
            elif active_module_count >= self.max_modules:
                blocked_actions.append("GROW")
                blocked_reasons["GROW"] = f"max modules ({self.max_modules}) reached"
            elif current_parameters >= self.max_parameters:
                blocked_actions.append("GROW")
                blocked_reasons["GROW"] = f"max parameters ({self.max_parameters}) reached"
            elif is_repeated_failure("GROW"):
                blocked_actions.append("GROW")
                fd = self.failed_mutations["GROW"]
                blocked_reasons["GROW"] = f"fingerprint {fd['fingerprint']} failed at need {fd['need_score']:.3f}"
            else:
                eligible_actions.append("GROW")

            # PRUNE logic
            if is_repeated_failure("PRUNE"):
                blocked_actions.append("PRUNE")
                fd = self.failed_mutations["PRUNE"]
                blocked_reasons["PRUNE"] = f"fingerprint {fd['fingerprint']} failed at need {fd['need_score']:.3f}"
            elif youngest_module_age < self.min_module_lifetime:
                blocked_actions.append("PRUNE")
                blocked_reasons["PRUNE"] = f"youngest module ({youngest_module_age}) < min lifetime ({self.min_module_lifetime})"
            elif active_module_count <= 1:
                blocked_actions.append("PRUNE")
                blocked_reasons["PRUNE"] = "cannot prune last module"
            else:
                eligible_actions.append("PRUNE")

            # REWIRE logic
            if is_repeated_failure("REWIRE"):
                blocked_actions.append("REWIRE")
                fd = self.failed_mutations["REWIRE"]
                blocked_reasons["REWIRE"] = f"fingerprint {fd['fingerprint']} failed at need {fd['need_score']:.3f}"
            else:
                from evolution.mutation import MutationManager
                mut_mgr = MutationManager()
                if brain is not None and not mut_mgr.can_rewire(brain):
                    blocked_actions.append("REWIRE")
                    blocked_reasons["REWIRE"] = "No valid rewire candidates under current sparsity constraints."
                else:
                    eligible_actions.append("REWIRE")
        
        eligible_actions.append("ADAPT")

        # Now pick the highest priority eligible action based on thresholds
        if "GROW" in eligible_actions and need_score >= self.grow_threshold:
            action = "GROW"
            reason = f"Evidence supports GROW. Need score ({need_score:.2f}) >= {self.grow_threshold:.2f}."
        elif "PRUNE" in eligible_actions and sig_dict["utilization"] is not None and sig_dict["utilization"] < self.prune_threshold:
            action = "PRUNE"
            reason = f"Utilization ({sig_dict['utilization']:.2f}) < prune threshold ({self.prune_threshold:.2f})."
        elif "REWIRE" in eligible_actions and need_score >= self.rewire_threshold:
            action = "REWIRE"
            reason = f"Need score ({need_score:.2f}) calls for REWIRE to optimize sparse connections."
        elif need_score >= self.grow_threshold or (sig_dict["utilization"] is not None and sig_dict["utilization"] < self.prune_threshold) or need_score >= self.rewire_threshold:
            # An action was preferred but blocked. Fallback to ADAPT if need score is high enough.
            action = "ADAPT"
            if need_score >= self.grow_threshold:
                reason = f"GROW needed but blocked: {blocked_reasons.get('GROW', 'unknown')}. Fallback ADAPT."
            elif sig_dict["utilization"] is not None and sig_dict["utilization"] < self.prune_threshold:
                reason = f"PRUNE needed but blocked: {blocked_reasons.get('PRUNE', 'unknown')}. Fallback ADAPT."
            else:
                reason = f"REWIRE needed but blocked: {blocked_reasons.get('REWIRE', 'unknown')}. Fallback ADAPT."
        elif "ADAPT" in eligible_actions and need_score >= self.adapt_threshold:
            action = "ADAPT"
            reason = f"Need score ({need_score:.2f}) calls for weight adaptation with existing capacity."
        else:
            action = "REUSE"
            reason = f"Need score ({need_score:.2f}) is low. Existing capacity is sufficient."

        self._pending_need_score = need_score
        self._pending_fingerprint = current_fingerprint

        return EvolutionDecision(
            action=action,
            need_score=need_score,
            signals=signals,
            reason=reason,
            budget_remaining=budget_remaining,
            cooldown_active=cooldown_active,
            eligible_actions=eligible_actions,
            blocked_actions=blocked_actions,
            blocked_reasons=blocked_reasons
        )

    def record_mutation_outcome(self, current_step: int, action: str, accepted: bool):
        self.last_mutation_step = current_step
        if accepted:
            if action == "GROW":
                self.level_growth_count += 1
                self.total_growth_count += 1
        else:
            self.last_failed_mutation_step = current_step
            # Record or update the highest failure score for this action and lock the fingerprint
            current_max = 0.0
            if action in self.failed_mutations:
                current_max = self.failed_mutations[action]["need_score"]
                
            self.failed_mutations[action] = {
                "need_score": max(current_max, self._pending_need_score),
                "fingerprint": self._pending_fingerprint
            }

