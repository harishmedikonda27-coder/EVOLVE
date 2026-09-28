"""
Convergence-driven training policies for EVOLVE.

Replaces naive fixed-window threshold checks with statistically principled
convergence detection.  The tracker inspects the recent evaluation history
and classifies the learning curve into one of four regimes:

    IMPROVING   – performance is still climbing at a meaningful rate
    STAGNATING  – performance is flat or declining despite continued training
    CONVERGED   – performance has met the target and is statistically stable
    EXHAUSTED   – step budget has been consumed without convergence

These regimes drive the outer training loop: CONVERGED advances to the next
task, STAGNATING may trigger a structural mutation, EXHAUSTED records the
failure and advances.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np


class ConvergenceStatus(str, Enum):
    IMPROVING = "IMPROVING"
    STAGNATING = "STAGNATING"
    CONVERGED = "CONVERGED"
    EXHAUSTED = "EXHAUSTED"


@dataclass
class ConvergenceResult:
    status: ConvergenceStatus
    variance: float
    slope: float
    mean_performance: float
    target_met: bool
    reason: str


class ConvergenceTracker:
    """Statistically principled convergence detector.

    Parameters
    ----------
    target_score : float
        Performance threshold that must be exceeded for convergence.
    window_size : int
        Number of recent evaluation points to consider.
    min_evaluations : int
        Minimum evaluations before convergence can be declared.
    variance_threshold : float
        Maximum allowed variance in the evaluation window for a stable result.
    stagnation_slope : float
        If the OLS slope of recent evaluations is below this value **and** the
        target has not been met, the learner is considered stagnating.
    max_steps : int
        Step budget.  If exhausted, the result is EXHAUSTED.
    """

    def __init__(
        self,
        target_score: float = 0.8,
        window_size: int = 5,
        min_evaluations: int = 3,
        variance_threshold: float = 0.005,
        stagnation_slope: float = 0.001,
        max_steps: int = 20000,
    ):
        self.target_score = target_score
        self.window_size = window_size
        self.min_evaluations = min_evaluations
        self.variance_threshold = variance_threshold
        self.stagnation_slope = stagnation_slope
        self.max_steps = max_steps

    def check(
        self,
        history: List[float],
        current_step: int,
    ) -> ConvergenceResult:
        """Evaluate the convergence status of a learning curve.

        Parameters
        ----------
        history : List[float]
            Chronological list of normalised performance scores (0–1).
        current_step : int
            Number of training steps taken so far on this task.

        Returns
        -------
        ConvergenceResult
        """
        n = len(history)

        # Not enough data yet → still improving by definition
        if n < self.min_evaluations:
            return ConvergenceResult(
                status=ConvergenceStatus.IMPROVING,
                variance=0.0,
                slope=0.0,
                mean_performance=float(np.mean(history)) if history else 0.0,
                target_met=False,
                reason=f"Only {n}/{self.min_evaluations} evaluations collected.",
            )

        window = history[-self.window_size:]
        variance = float(np.var(window))
        mean_perf = float(np.mean(window))

        # Ordinary least-squares slope over the window
        x = np.arange(len(window), dtype=np.float64)
        y = np.array(window, dtype=np.float64)
        if len(x) > 1:
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0

        target_met = all(v >= self.target_score for v in window)

        # --- Decision logic ---

        # Budget exhausted?
        if current_step >= self.max_steps:
            return ConvergenceResult(
                status=ConvergenceStatus.EXHAUSTED,
                variance=variance,
                slope=slope,
                mean_performance=mean_perf,
                target_met=target_met,
                reason=f"Step budget ({self.max_steps}) exhausted at step {current_step}.",
            )

        # Converged: target met AND variance is below threshold (stable)
        if target_met and variance <= self.variance_threshold:
            return ConvergenceResult(
                status=ConvergenceStatus.CONVERGED,
                variance=variance,
                slope=slope,
                mean_performance=mean_perf,
                target_met=True,
                reason=(
                    f"All {len(window)} recent evaluations >= {self.target_score:.2f} "
                    f"(var={variance:.5f} <= {self.variance_threshold:.5f})."
                ),
            )

        # Stagnating: not meeting target AND slope is flat/negative
        if not target_met and slope <= self.stagnation_slope and n >= self.window_size:
            return ConvergenceResult(
                status=ConvergenceStatus.STAGNATING,
                variance=variance,
                slope=slope,
                mean_performance=mean_perf,
                target_met=False,
                reason=(
                    f"Performance slope ({slope:.5f}) <= stagnation threshold "
                    f"({self.stagnation_slope:.5f}) without meeting target."
                ),
            )

        # Default: still improving
        return ConvergenceResult(
            status=ConvergenceStatus.IMPROVING,
            variance=variance,
            slope=slope,
            mean_performance=mean_perf,
            target_met=target_met,
            reason="Performance is still improving.",
        )
