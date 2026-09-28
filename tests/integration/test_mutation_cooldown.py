import pytest
from evolution.policies import DecisionPolicy
from evolution.signals import EvolutionSignals

def test_failed_mutation_cooldown():
    policy = DecisionPolicy(cooldown_steps=100)
    
    # 1. Apply a GROW mutation successfully
    policy.record_mutation_outcome(current_step=500, action="GROW", accepted=True)
    
    # Step 550: Cooldown should be active (550 - 500 = 50 < 100)
    decision = policy.evaluate(need_score=0.9, signals=EvolutionSignals(0.5,0.0,0.0,0.0,0.0,0.0), current_step=550)
    assert decision.cooldown_active is True
    assert decision.action == "ADAPT"
    
    # Step 650: Cooldown should be over (650 - 500 = 150 > 100)
    decision = policy.evaluate(need_score=0.9, signals=EvolutionSignals(0.5,0.0,0.0,0.0,0.0,0.0), current_step=650)
    assert decision.cooldown_active is False
    assert decision.action == "GROW"
    
    # 2. Reject a GROW mutation at step 700
    policy.record_mutation_outcome(current_step=700, action="GROW", accepted=False)
    
    # Step 850: Normal cooldown is 100, so 150 > 100. BUT it failed, so cooldown is 2x = 200.
    # 850 - 700 = 150 < 200, so cooldown SHOULD BE ACTIVE!
    decision = policy.evaluate(need_score=0.9, signals=EvolutionSignals(0.5,0.0,0.0,0.0,0.0,0.0), current_step=850)
    assert decision.cooldown_active is True
    assert decision.action == "ADAPT"
    
    # Step 950: Failed cooldown over (950 - 700 = 250 >= 200)
    # 1. same evidence -> blocked
    decision_same = policy.evaluate(need_score=0.9, signals=EvolutionSignals(0.5,0.0,0.0,0.0,0.0,0.0), current_step=950)
    assert decision_same.action == "REWIRE"
    
    # 2. new evidence -> same mutation can become eligible
    decision_higher = policy.evaluate(need_score=0.96, signals=EvolutionSignals(0.5,0.0,0.0,0.0,0.0,0.0), current_step=950)
    assert decision_higher.action == "GROW"
    
    # 3. different action -> eligible immediately
    # Suppose need_score dropped but we try to REWIRE (which requires 0.40)
    # Wait, if we force the policy to pick REWIRE... 
    # Let's say need_score is 0.45. It doesn't hit GROW threshold, so it falls back.
    decision_diff = policy.evaluate(need_score=0.45, signals=EvolutionSignals(0.5,0.0,0.0,0.0,0.0,0.0), current_step=950)
    assert decision_diff.action == "REWIRE"
