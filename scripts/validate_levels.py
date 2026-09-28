import sys
import os

# Add the EVOLVE backend root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.families import HierarchicalNavigation, MemoryObservation, AdaptiveDynamics
from environment.base import LevelGenerationError

def validate_family(env_class, family_name: str, num_levels: int = 50):
    print(f"\n==================================================")
    print(f"Validating {family_name} ({num_levels} levels)")
    print(f"==================================================")
    
    passed = 0
    failed = 0
    
    for level in range(1, num_levels + 1):
        try:
            # Instantiate
            env = env_class(level=level)
            
            # Reset (generation)
            seed = 42 + level
            env.reset(seed=seed)
            
            # Validate solvability
            if not env.is_solvable():
                print(f"[FAIL] Level {level}: NOT SOLVABLE")
                failed += 1
                continue
                
            # Validate TaskProfile
            profile = env.get_task_profile()
            expected_keys = [
                "family", "level_id", "seed", "task_variant", "objective", 
                "difficulty", "observation_regime", "dynamics_regime", 
                "reward_spec", "constraints", "solvability_spec", "max_steps"
            ]
            missing_keys = [k for k in expected_keys if k not in profile]
            if missing_keys:
                print(f"[FAIL] Level {level}: Missing TaskProfile keys: {missing_keys}")
                failed += 1
                continue
                
            reward_spec = profile["reward_spec"]
            if "R_min" not in reward_spec or "R_max" not in reward_spec:
                print(f"[FAIL] Level {level}: Invalid reward_spec: {reward_spec}")
                failed += 1
                continue
                
            # Ensure R_min < R_max
            if reward_spec["R_min"] >= reward_spec["R_max"]:
                print(f"[FAIL] Level {level}: R_min ({reward_spec['R_min']}) >= R_max ({reward_spec['R_max']})")
                failed += 1
                continue
                
            passed += 1
            if level % 10 == 0:
                print(f"[OK] Level {level} passed.")
                
        except LevelGenerationError as e:
            print(f"[FAIL] Level {level}: LevelGenerationError: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] Level {level}: Unexpected Exception: {e}")
            failed += 1
            
    print(f"\n{family_name} Summary: {passed}/{num_levels} passed.")
    return passed == num_levels

if __name__ == "__main__":
    families = [
        (HierarchicalNavigation, "HierarchicalNavigation"),
        (MemoryObservation, "MemoryObservation"),
        (AdaptiveDynamics, "AdaptiveDynamics"),
    ]
    
    all_passed = True
    for env_cls, name in families:
        success = validate_family(env_cls, name, 50)
        if not success:
            all_passed = False
            
    print(f"\n==================================================")
    if all_passed:
        print("[SUCCESS] ALL 150 LEVELS VALIDATED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("[ERROR] LEVEL VALIDATION FAILED.")
        sys.exit(1)
