from evolution.controller import EvolutionController


def main():
    print("=== EVOLVE Member 2 Standalone Controller Demo ===")
    controller = EvolutionController()

    print("\n1. Evaluating Low Difficulty Scenario (Task A stable):")
    dec1 = controller.evaluate_step(
        current_performance=0.90,
        recent_history=[0.88, 0.89, 0.90],
        task_profile_distance=0.05,
    )
    print(f"Action: {dec1.action}, Need Score: {dec1.need_score:.3f}, Reason: {dec1.reason}")

    print("\n2. Evaluating High Difficulty Scenario (Shift to Task B):")
    dec2 = controller.evaluate_step(
        current_performance=0.15,
        recent_history=[0.15, 0.15, 0.15],
        task_profile_distance=0.85,
        old_task_score_before=0.90,
        old_task_score_current=0.40,
        current_step=100,
    )
    print(f"Action: {dec2.action}, Need Score: {dec2.need_score:.3f}, Reason: {dec2.reason}")

    print("\nStandalone Controller demo completed successfully!")


if __name__ == "__main__":
    main()
