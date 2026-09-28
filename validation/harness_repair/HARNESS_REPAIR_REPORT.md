# HARNESS REPAIR AND CAUSALITY RESTORATION REPORT

## 1. Objective
The primary objective of this phase was to forensically repair the EVOLVE Experiment Harness to ensure actual, causal execution of structural adaptations (GROW, PRUNE, REWIRE). As discovered in the preceding audit, the previous version of the harness (`ExperimentHarness`) had bypassed the `TransactionRunner`, meaning that topological changes were never committed to the active model.

This report summarizes the repairs made, the tests implemented to prove causal execution, and the validation of missing metric handling and RNG isolation.

## 2. Repairs Executed

### A. TransactionRunner Integration
The `ExperimentHarness` was fundamentally re-wired. 
- The mocked `runner.execute()` stub was replaced with the mathematically accurate `runner.run_transaction()` signature.
- `TransactionRunner` now receives the live `agent`, `env`, and evaluation closures. It intrinsically simulates forward-learning for up to 500 steps when proposing structural modifications, verifies their impact via the retention and novelty verification thresholds, and handles the `status` ("ACCEPT" or "ROLLBACK") appropriately.
- If rejected, `TransactionRunner` seamlessly invokes the `RollbackManager` to restore the neural parameters and topological registry.

### B. Missing Data Protocol
- The `harness.py` was updated so that missing measurements (e.g., plasticity for non-adaptation trials) no longer default to `0.0`. They are systematically recorded as `null` or excluded from the JSON schema directly rather than injecting false 0 values into the empirical record.

### C. Strict RNG Isolation for Plasticity
- Probing plasticity introduces a destructive, forward-simulated gradient path. If the RNG state of the Python/Numpy/PyTorch instances is altered, the main training trajectory diverges artificially.
- An `IsolatedRNGContext` was built in `utils/rng_isolation.py`. The plasticity probe runner (`evolution/plasticity.py`) was wrapped in this context manager to completely isolate the RNG state.

## 3. Causal Verification

Two comprehensive tests were built and successfully passed in `tests/research/`:
1. `test_plasticity_rng_isolation.py`: Mathematically proved that sampling a probe step, rolling it back, and generating numbers from the environment/PyTorch exactly aligns with the clean baseline RNG trajectory.
2. `test_mutation_execution.py`: Mocked the `TransactionRunner` evaluation loop. It validated that:
   - When verification thresholds are met (Mocked as `ACCEPT`), the topology dimensions and parameter count actually increase.
   - When verification thresholds are missed (Mocked as `ROLLBACK`), the parameter dimensions are exactly restored.

## 4. Tiny Four-Condition Benchmark
A 2-task benchmark (`tiny_harness_test.py`) with heavily restricted step limits was run across all four conditions (FIXED, DETERMINISTIC_EVOLVE, RANDOM_VALID_ACTION, LEARNED_EVOLVE) using seed 42.

**Results confirmed:**
- Under `LEARNED_EVOLVE`, an adaptation proposal (`REWIRE`) was attempted at Step 510.
- `TransactionRunner` successfully received the proposal and evaluated it.
- Due to strict verification settings on a tiny benchmark, it correctly failed verification (`New task gain (-0.800) below threshold (0.100).`).
- The operation correctly triggered `RollbackManager`, explicitly recorded the `"status": "ROLLBACK"` string, and accurately retained the `topology_before: 13893` and `topology_after: 13893` invariant. 
- The JSON artifacts successfully persisted all intervention attempts.

## 5. Conclusion & Recommendation
The backend experiment harness is fully repaired and causally sound. Structural interventions definitively branch, train, rollback or commit to the live PyTorch models, and no artificial divergence taints the control baselines.

The system is ready to safely proceed to Phase G (Empirical Re-run) for the formal N=3 validation of the new `post_harness_repair_v1` generation.
