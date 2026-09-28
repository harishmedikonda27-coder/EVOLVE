from training.train_fixed import run_fixed_experiment
from training.train_replay import run_replay_experiment
from training.train_always_grow import run_always_grow_experiment
from training.train_evolve import run_evolve_experiment


def main():
    print("=========================================================")
    print("=== EVOLVE Scientific Benchmark & Baseline Comparison ===")
    print("=========================================================\n")

    seed = 42
    steps_a = 3000
    steps_b = 3000

    res_fixed = run_fixed_experiment(steps_a=steps_a, steps_b=steps_b, seed=seed)
    print("---------------------------------------------------------")
    res_replay = run_replay_experiment(steps_a=steps_a, steps_b=steps_b, seed=seed)
    print("---------------------------------------------------------")
    res_always = run_always_grow_experiment(steps_a=steps_a, steps_b=steps_b, seed=seed)
    print("---------------------------------------------------------")
    res_evolve = run_evolve_experiment(steps_a=steps_a, steps_b=steps_b, seed=seed)
    print("---------------------------------------------------------")

    print("\n================ FINAL BENCHMARK SUMMARY ================")
    print(f"{'Method':<18} | {'A (Pre)':<8} | {'B (Post)':<8} | {'A (Post)':<8} | {'Retention':<10} | {'Params':<8}")
    print("-" * 74)
    for res in [res_fixed, res_replay, res_always, res_evolve]:
        print(
            f"{res['experiment']:<18} | "
            f"{res['a_before']:<8.3f} | "
            f"{res['b_after']:<8.3f} | "
            f"{res['a_after']:<8.3f} | "
            f"{res['retention_ratio']:<10.3f} | "
            f"{res['parameter_count']:<8}"
        )
    print("=========================================================")


if __name__ == "__main__":
    main()
