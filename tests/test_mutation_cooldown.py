import pytest
from evolution.policies import DecisionPolicy, EvolutionDecision
from evolution.signals import EvolutionSignals

def test_repeated_failure_blocked():
    policy = DecisionPolicy(cooldown_steps=0) # disable normal time-based cooldown for testing
    
    signals = EvolutionSignals(
        performance_gap=0.4,
        stagnation=0.0,
        novelty=0.0,
        interference=0.0,
        utilization=1.0,
        cost=0.0
    )
    
    # First time, need score is high (0.6)
    decision1 = policy.evaluate(need_score=0.6, signals=signals, current_step=1000)
    assert decision1.action == "GROW"
    
    # Record failure
    policy.record_mutation_outcome(current_step=1000, action="GROW", accepted=False)
    
    # Second time, need score hasn't grown materially (still 0.6)
    decision2 = policy.evaluate(need_score=0.6, signals=signals, current_step=1100)
    assert decision2.action == "REWIRE" # Blocked GROW, so fell back to next highest (REWIRE)
    assert "GROW" in decision2.blocked_actions
    assert "fingerprint" in decision2.blocked_reasons["GROW"]
    
    # Third time, need score increases to 0.70 (> 0.05 increase)
    decision3 = policy.evaluate(need_score=0.70, signals=signals, current_step=1200)
    assert decision3.action == "GROW" # Allowed!
