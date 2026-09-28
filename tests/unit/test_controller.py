import pytest
from evolution.signals import SignalCalculator
from evolution.scorer import NeedScorer
from evolution.policies import DecisionPolicy
from evolution.controller import EvolutionController


def test_signal_calculator_bounds():
    calc = SignalCalculator()
    signals = calc.compute(
        current_performance=0.2,
        target_performance=1.0,
        recent_history=[0.1, 0.1, 0.1],
        task_profile_distance=0.8,
        interference_value=0.3,
        module_utilizations=[0.5, 0.6],
        current_param_count=2000,
        max_param_budget=10000,
    )
    sig_dict = signals.to_dict()
    for val in sig_dict.values():
        if val is not None:
            assert 0.0 <= val <= 1.0


def test_need_scorer():
    scorer = NeedScorer(w_p=0.35, w_s=0.25, w_n=0.20, w_i=0.30, w_c=0.15)
    calc = SignalCalculator()
    signals = calc.compute(
        current_performance=0.1,
        target_performance=1.0,
        recent_history=[0.1, 0.1, 0.1],
        task_profile_distance=0.9,
    )
    score = scorer.compute(signals)
    assert 0.0 <= score <= 1.0


def test_decision_policy():
    policy = DecisionPolicy(grow_threshold=0.60, growth_budget_per_level=2, cooldown_steps=100)
    calc = SignalCalculator()
    
    # High difficulty signals
    signals = calc.compute(
        current_performance=0.1,
        target_performance=1.0,
        recent_history=[0.1, 0.1, 0.1],
        task_profile_distance=0.9,
    )
    decision = policy.evaluate(need_score=0.80, signals=signals, current_step=0)
    assert decision.action == "GROW"

    policy.record_mutation_outcome(current_step=0, action="GROW", accepted=True)

    # Second check inside cooldown
    decision2 = policy.evaluate(need_score=0.80, signals=signals, current_step=10)
    assert decision2.action == "ADAPT"
    assert decision2.cooldown_active is True
def test_controller_rewire_feasibility_matches_executor():
    from brain.brain import ModularBrain
    from evolution.mutation import MutationManager
    
    controller = EvolutionController()
    controller.policy.rewire_threshold = -1.0 # Force REWIRE if eligible
    mut = MutationManager()
    
    brain = ModularBrain(10, 5)
    brain.add_module("mod1", 0, "TEST")
    
    # CASE A: Infeasible (Fully Dense)
    mod = brain.registry.get_module("mod1")
    mod.mask1.fill_(1.0)
    mod.mask2.fill_(1.0)
    
    decision_infeasible = controller.evaluate_step(
        current_performance=1.0, current_param_count=100,
        active_module_count=1, youngest_module_age=100, brain=brain
    )
    
    assert mut.can_rewire(brain) == False
    assert "REWIRE" in decision_infeasible.blocked_actions
    assert decision_infeasible.action != "REWIRE" # Should fall back (to ADAPT/REUSE)
    
    # Verification steps should be 0 because transaction won't be entered (verified by run.py logic)
    # But let's verify topology revision doesn't change on infeasible rewire execution
    topo_rev_before = brain.topology_revision
    mut.execute_rewire(brain)
    assert brain.topology_revision == topo_rev_before
    
    # CASE B: Feasible (Sufficient Sparsity)
    mod.mask1.view(-1)[:50] = 0.0
    mod.mask1.view(-1)[-50:] = 0.0
    
    decision_feasible = controller.evaluate_step(
        current_performance=1.0, current_param_count=100,
        active_module_count=1, youngest_module_age=100, brain=brain
    )
    
    assert mut.can_rewire(brain) == True
    assert "REWIRE" in decision_feasible.eligible_actions
    assert decision_feasible.action == "REWIRE"
