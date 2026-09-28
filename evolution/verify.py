from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class VerificationResult:
    accepted: bool
    reason: str
    new_task_gain: float
    old_task_retention: float
    structural_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "new_task_gain": self.new_task_gain,
            "old_task_retention": self.old_task_retention,
            "structural_cost": self.structural_cost,
        }


class Verifier:
    def __init__(
        self,
        min_new_task_gain: float = 0.10,
        min_old_task_retention: float = 0.85,
        max_allowed_cost: float = 0.80,
    ):
        self.min_new_task_gain = min_new_task_gain
        self.min_old_task_retention = min_old_task_retention
        self.max_allowed_cost = max_allowed_cost

    def evaluate(
        self,
        new_task_score_before: float,
        new_task_score_after: float,
        old_task_score_before: float,
        old_task_score_after: float,
        structural_cost: float,
    ) -> VerificationResult:
        new_gain = new_task_score_after - new_task_score_before
        retention = old_task_score_after / max(old_task_score_before, 1e-5)

        reasons = []
        accepted = True

        if new_gain < self.min_new_task_gain:
            accepted = False
            reasons.append(f"New task gain ({new_gain:.3f}) below threshold ({self.min_new_task_gain:.3f}).")

        if retention < self.min_old_task_retention:
            accepted = False
            reasons.append(f"Old task retention ({retention:.3f}) below threshold ({self.min_old_task_retention:.3f}).")

        if structural_cost > self.max_allowed_cost:
            accepted = False
            reasons.append(f"Structural cost ({structural_cost:.3f}) exceeds threshold ({self.max_allowed_cost:.3f}).")

        reason_str = "Mutation accepted." if accepted else " ; ".join(reasons)

        return VerificationResult(
            accepted=accepted,
            reason=reason_str,
            new_task_gain=new_gain,
            old_task_retention=retention,
            structural_cost=structural_cost,
        )
