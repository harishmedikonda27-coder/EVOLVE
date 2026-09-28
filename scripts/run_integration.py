import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.experiments.run import run_continuous_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, default=5, help="Number of levels per family")
    args = parser.parse_args()
    
    families = ["HierarchicalNavigation", "MemoryObservation", "AdaptiveDynamics"]
    
    print(f"Starting Integration Experiment: {args.levels} levels x {len(families)} families")
    
    try:
        run_continuous_experiment(
            run_id=f"integration_{args.levels}x{len(families)}",
            families=families,
            levels=args.levels,
            seed=42,
            max_steps=2000, # fast convergence safety cutoff for tests
            eval_freq=100
        )
        print("==================================================")
        print(f"[SUCCESS] {args.levels}x{len(families)} INTEGRATION PASSED")
    except Exception as e:
        print("==================================================")
        print(f"[FAILED] {args.levels}x{len(families)} INTEGRATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
