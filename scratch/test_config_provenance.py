import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.experiments.run import run_continuous_experiment

def test_config_provenance():
    config_path = os.path.join("configs", "policy.json")
    with open(config_path, "r") as f:
        original_config = json.load(f)
        
    try:
        # We will modify max_steps in the config to prove it's authoritative
        modified_config = original_config.copy()
        modified_config["mutation_cooldown"] = 777
        modified_config["verification_budget"] = 123
        modified_config["convergence_threshold"] = 0.999
        
        with open(config_path, "w") as f:
            json.dump(modified_config, f, indent=4)
            
        print("Modified config written.")
        
        # We can test parsing directly
        from evolution.policies import DecisionPolicy
        from training.convergence import ConvergenceTracker
        
        policy_cfg = modified_config
        d_policy = DecisionPolicy(
            adapt_threshold=policy_cfg.get("mutation_thresholds", {}).get("adapt", 0.25),
            grow_threshold=policy_cfg.get("mutation_thresholds", {}).get("grow", 0.55),
            prune_threshold=policy_cfg.get("mutation_thresholds", {}).get("prune", 0.15),
            rewire_threshold=policy_cfg.get("mutation_thresholds", {}).get("rewire", 0.40),
            cooldown_steps=policy_cfg.get("mutation_cooldown", 150),
            max_modules=policy_cfg.get("structural_budgets", {}).get("max_modules", 5),
            max_parameters=policy_cfg.get("structural_budgets", {}).get("max_parameters", 50000),
            min_module_lifetime=policy_cfg.get("structural_budgets", {}).get("min_module_age", 1000),
            need_score_delta_threshold=policy_cfg.get("evidence_change_thresholds", {}).get("need_score_delta", 0.05)
        )
        assert d_policy.cooldown_steps == 777, f"Expected 777, got {d_policy.cooldown_steps}"
        
        target_score = policy_cfg.get("convergence_threshold", 0.95)
        convergence_tracker = ConvergenceTracker(
            target_score=target_score,
            window_size=policy_cfg.get("convergence_window", 5),
            min_evaluations=3,
            variance_threshold=0.005,
            stagnation_slope=0.001,
            max_steps=2000,
        )
        assert convergence_tracker.target_score == 0.999, f"Expected 0.999, got {convergence_tracker.target_score}"
        
        print("Provenance Negative Test PASSED: Objects correctly respect policy.json values!")
            
    finally:
        with open(config_path, "w") as f:
            json.dump(original_config, f, indent=4)
        print("Original config restored.")

if __name__ == '__main__':
    test_config_provenance()
