from typing import Dict, Any
from evolution.signals import EvolutionSignals


class NeedScorer:
    def __init__(
        self,
        w_p: float = 0.40,
        w_s: float = 0.30,
        w_n: float = 0.15,
        w_i: float = 0.15,
        w_c: float = 0.20,
    ):
        # Positive terms sum to 1.0 to prevent artificial saturation > 1.0
        # If P=1, S=1, N=1, I=1 -> Sum = 1.0
        self.w_p = w_p
        self.w_s = w_s
        self.w_n = w_n
        self.w_i = w_i
        self.w_c = w_c

    def compute(self, signals: EvolutionSignals) -> float:
        sig_dict = signals.to_dict()
        p = sig_dict["performance_gap"]
        s = sig_dict["stagnation"]
        n = sig_dict["novelty"]
        i = sig_dict["interference"]
        c = sig_dict["cost"]

        score = (
            self.w_p * p +
            self.w_s * s +
            self.w_n * n +
            self.w_i * i -
            self.w_c * c
        )
        return float(max(0.0, min(1.0, score)))

    def compute_breakdown(self, signals: EvolutionSignals) -> Dict[str, Any]:
        sig_dict = signals.to_dict()
        score = self.compute(signals)
        return {
            "need_score": score,
            "terms": {
                "w_p * P": float(self.w_p * sig_dict["performance_gap"]),
                "w_s * S": float(self.w_s * sig_dict["stagnation"]),
                "w_n * N": float(self.w_n * sig_dict["novelty"]),
                "w_i * I": float(self.w_i * sig_dict["interference"]),
                "-w_c * C": float(-self.w_c * sig_dict["cost"]),
            },
            "raw_signals": {
                "P": sig_dict["raw_performance_gap"],
                "S": sig_dict["raw_stagnation"],
                "N": sig_dict["raw_novelty"],
                "I": sig_dict["raw_interference"],
                "C": sig_dict["raw_cost"],
                "U": sig_dict["raw_utilization"],
            },
            "weights": {
                "w_p": self.w_p,
                "w_s": self.w_s,
                "w_n": self.w_n,
                "w_i": self.w_i,
                "w_c": self.w_c,
            }
        }

