import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evolution.policies import DecisionPolicy
from evolution.signals import EvolutionSignals

def test_grow_reeligibility():
    # Configure policy with delta threshold 0.05
    policy = DecisionPolicy(
        grow_threshold=0.5,
        prune_threshold=0.1,
        rewire_threshold=0.4,
        adapt_threshold=0.2,
        need_score_delta_threshold=0.05
    )

    signals = EvolutionSignals(
        performance_gap=0.5,
        stagnation=0.5,
        novelty=0.1,
        interference=0.1,
        utilization=0.8,
        cost=0.5
    )

    # First attempt: NeedScore 0.719 -> GROW
    print("=== INITIAL ATTEMPT ===")
    decision1 = policy.evaluate(
        need_score=0.719,
        signals=signals,
        current_step=0,
        active_module_count=2,
        current_parameters=10000,
        youngest_module_age=1000
    )
    print(f"Decision 1: {decision1.action} (Reason: {decision1.reason})")
    
    # Record failure
    policy.record_mutation_outcome(current_step=100, action="GROW", accepted=False)
    
    # Case A: NeedScore increases to 0.720 (Delta = 0.001 < 0.05). Should block GROW.
    print("\n=== CASE A: Delta < 0.05 ===")
    decision_A = policy.evaluate(
        need_score=0.720,
        signals=signals,
        current_step=600, # well past cooldown
        active_module_count=2,
        current_parameters=10000,
        youngest_module_age=1000
    )
    print(f"Eligible Actions: {decision_A.eligible_actions}")
    print(f"Decision A: {decision_A.action} (Reason: {decision_A.reason})")
    
    # Case B: NeedScore increases to 0.780 (Delta = 0.061 > 0.05). Should allow GROW.
    print("\n=== CASE B: Delta > 0.05 ===")
    decision_B = policy.evaluate(
        need_score=0.780,
        signals=signals,
        current_step=800,
        active_module_count=2,
        current_parameters=10000,
        youngest_module_age=1000
    )
    print(f"Decision B: {decision_B.action} (Reason: {decision_B.reason})")
    
    # Case C: Materially different state (different fingerprint).
    print("\n=== CASE C: Different State ===")
    decision_C = policy.evaluate(
        need_score=0.720,
        signals=signals,
        current_step=1200,
        active_module_count=3,
        current_parameters=12000,
        youngest_module_age=1000
    )
    print(f"Decision C: {decision_C.action} (Reason: {decision_C.reason})")

if __name__ == '__main__':
    test_grow_reeligibility()
