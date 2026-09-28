import pytest
import torch
from evolution.controller import EvolutionController
from evolution.state import AdaptationState
from evolution.policies import EvolutionDecision

def test_random_valid_action():
    controller = EvolutionController(adaptation_mode="random")
    
    # Active modules = 1 (Min modules, so PRUNE is infeasible)
    # Budget = 0 (GROW, PRUNE, REWIRE infeasible)
    # Feasible = REUSE, ADAPT
    
    class MockPolicy1:
        def evaluate(self, *args, **kwargs):
            from evolution.signals import EvolutionSignals
            return EvolutionDecision(
                action="REUSE",
                budget_remaining=0, # No budget
                need_score=0.0,
                signals=EvolutionSignals(0,0,0,0,0,0),
                reason="Mock",
                cooldown_active=False
            )
            
    controller.policy = MockPolicy1()
    
    decision = controller.evaluate_step(
        current_performance=0.5,
        current_param_count=1000,
        max_param_budget=1000,
        active_module_count=1,
    )
    
    # Since budget is 0, REWIRE, PRUNE, GROW are blocked.
    # Active modules = 1, PRUNE is blocked.
    assert decision.action in ["REUSE", "ADAPT"]

def test_random_valid_action_cooldown():
    controller = EvolutionController(adaptation_mode="random")
    # Simulate a state where cooldown is active. The legacy policy will set cooldown_active.
    # We pass it through `evaluate_step`. But legacy policy might decide REUSE automatically.
    # Let's mock the base policy to return a decision with cooldown_active=True.
    
    class MockPolicy:
        def evaluate(self, *args, **kwargs):
            from evolution.signals import EvolutionSignals
            dec = EvolutionDecision(
                action="REUSE",
                budget_remaining=1000,
                need_score=0.0,
                signals=EvolutionSignals(0,0,0,0,0,0),
                reason="Mock",
                cooldown_active=True
            )
            return dec

    controller.policy = MockPolicy()
    
    decision = controller.evaluate_step(
        current_performance=0.5,
        current_param_count=1000,
        max_param_budget=2000,
        active_module_count=4,
    )
    
    assert decision.action == "REUSE"
    assert decision.cooldown_active is True
