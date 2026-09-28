# EVOLVE Backend Audit & Hardening Plan

This document represents the findings of the Phase 1 Discovery and Audit step. It identifies current architectural deficiencies, logical contradictions, and outlines the precise implementation steps for Phases 2 through 5.

## 1. Architectural Overview
The EVOLVE backend is structured to run self-adaptive continual learning experiments without frontend dependency. 
- **Experiment Loop**: `backend/experiments/run.py` orchestrates environments, agent actions, and structural evolution decisions.
- **Brain**: `brain/` defines the `ModularBrain` containing an encoder, dynamic module registry (`EvolveModule`), router, and output head.
- **Evolution**: `evolution/` calculates a `NeedScore` (based on Performance Gap, Stagnation, Novelty, Interference, Cost) to trigger `REUSE`, `ADAPT`, `GROW`, `PRUNE`, or `REWIRE`.
- **Agent**: `training/agent.py` implements DQN with experience replay.

## 2. Identified Deficiencies (The "Why")

### A. Environment Scaling & Verification
- **Current State**: The backend hardcodes `HierarchicalNavigation`, `MemoryObservation`, and `AdaptiveDynamics` with basic setups. In `run.py`, it loops over a fixed `levels=3` argument.
- **Deficiency**: There is no formalized 50-level scaling per family. Level difficulty scaling is superficial, and solvability validation (`is_solvable()`) for complex levels is not robustly generalized.
- **Scientific Goal**: We must generate exactly 50 levels per family (150 total tasks) deterministically from seeds, ensuring every single level is 100% solvable before presenting it to the agent.

### B. Convergence-Driven Training
- **Current State**: `run.py` checks convergence by taking 5 evaluation samples (every 500 steps) and checking if they all exceed a `target_score` (0.8).
- **Deficiency**: This is too rigid. It ignores variance, plateauing (stagnation below target), and doesn't explicitly separate convergence from simple threshold crossing.
- **Scientific Goal**: A principled `ConvergencePolicy` must evaluate the statistical stability of the learning curve, detecting true plateaus (for structural mutation triggers) and true convergence (for advancing to the next task).

### C. The "Zero Gain" Mutation Bug
- **Current State**: The user identified that `GROW` mutations often result in `gain=0.0` and rollback.
- **Deficiency**: When `execute_grow()` adds a new module, the `Router.update_module_count()` adds a new dimension to the linear routing gate. By default, this new weight might output small or negative logits compared to established modules, resulting in a near-zero softmax routing weight. During the 500-step verification, the new module receives almost zero gradient and therefore does not learn, yielding `0.0` performance gain.
- **Scientific Goal**: Newly grown modules must be given an explicit routing bias or exploration boost upon initialization to ensure they participate in the forward pass during verification.

### D. Reproducibility & Invariant Testing
- **Current State**: The system relies on `test_rollback.py` to check rollback fidelity, but lacks overarching state validation during the run.
- **Deficiency**: There is no runtime invariant checker that guarantees the `ModularBrain` parameter count exactly matches the `ModuleRegistry` active modules, and that the `Optimizer` parameter groups perfectly align with the model.
- **Scientific Goal**: Strict invariant checks must be inserted into the training loop and tested via a dedicated invariant testing suite.

---

## 3. Implementation Plan (Phases 2-5)

### Phase 2: Environment Families (50 Levels)
1.  **Refactor `families.py`**: Upgrade `HierarchicalNavigation`, `MemoryObservation`, and `AdaptiveDynamics` to accept a `difficulty` parameter (1 to 50) that parametrically scales grid size, obstacles, and dynamics.
2.  **Solvability Validator**: Implement deterministic A* or BFS in `generators.py` that absolutely guarantees a valid path exists for every generated level.
3.  **Task Registry**: Standardize `create_environment(family, level)` to cleanly iterate from 1 to 50.

### Phase 3: Convergence-Driven Training
1.  **Create `convergence.py`**: Build a `ConvergenceTracker` class.
2.  **Metrics**: Track `variance` (must be below a threshold for stability), `stagnation` (gradient of recent performance ~0), and `target_met`.
3.  **Integration**: Replace the naive 5-window check in `run.py` with `ConvergenceTracker.check_status(history)`.

### Phase 4: Hardening Mutation Logic (GROW/PRUNE/REWIRE)
1.  **Fix Router Bias**: In `Router.update_module_count()`, initialize the bias of the new gate output to a positive constant (e.g., `+1.0`) so the softmax gives the new module a meaningful initial routing weight.
2.  **Fix Optimizer Rebuilding**: Ensure `rebuild_optimizer` in `agent.py` does not leak momentum buffers and strictly assigns the correct state to the new module's parameters.
3.  **Verification Time**: Ensure the 500-step verification budget is sufficient for the new module to break symmetry.

### Phase 5: Reproducibility & Invariant Testing
1.  **Central Invariant Checker**: Create `backend/invariants.py` with `validate_brain_state(brain, optimizer)`.
2.  **Testing Suite**: Create `tests/test_invariants.py` to aggressively mutate the network and assert that parameter counts, router dimensions, and optimizer states remain perfectly synced.
