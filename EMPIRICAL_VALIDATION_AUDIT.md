# EMPIRICAL VALIDATION AUDIT

## 1. Executive Finding
**CRITICAL FAILURE**: The executed experiments did **not** perform any actual structural evolution. 
The experimental harness (`backend/experiments/harness.py`) mocked the structural mutation logic (bypassing the `TransactionRunner` completely). All topological modification attempts (GROW, PRUNE, REWIRE) were logged but never applied to the neural architecture. Consequently, all experimental conditions (`FIXED`, `DETERMINISTIC_EVOLVE`, `LEARNED_EVOLVE`, `RANDOM_VALID_ACTION`) operated under identical static topologies.

## 2. Actual Execution Path
The `ExperimentHarness` correctly instantiated the agents, environments, and policies. The `DecisionPolicy` (for `LEARNED_EVOLVE`) and `EvolutionController` (for `DETERMINISTIC_EVOLVE`) correctly proposed adaptation decisions based on evaluation metrics. 
However, the mutation execution pathway was stubbed out:
```python
# From harness.py:
runner = TransactionRunner(...)
# We omit the full runner args here for brevity, but they'd be populated
# ... (NO runner.execute() was ever called)
```

## 3. Mutation Evidence
A review of all raw JSON files confirms:
- **`interventions` array**: Empty (`[]`) for all experiments and all conditions.
- **`mutation_attempts`**: 0.
- **`accepted_mutations`**: 0.
- **Topology Changes**: False.

## 4. FIXED vs DETERMINISTIC Comparison
Because no mutations were ever executed, `FIXED` and `DETERMINISTIC_EVOLVE` were mathematically and functionally identical. Their final metrics matched exactly across all runs because they processed identical sequential environments with identical initial topologies and RNG states.

## 5. RNG Analysis
The measured performance discrepancies between `LEARNED_EVOLVE` and `FIXED` were caused strictly by uncontrolled RNG divergence, **not** structural adaptation. Specifically, the `PlasticityProbeEngine` cloned and trained temporary networks, which inadvertently advanced the PyTorch and NumPy random states. This RNG leakage caused future environmental sampling and action selection to diverge, creating the illusion of differentiated learning trajectories.

## 6. Plasticity Analysis
In the aggregate results, `FIXED` and `DETERMINISTIC_EVOLVE` reported plasticity AUCs of `0.0`. The audit confirms that this was an artifact of the harness only invoking the plasticity probe for the `LEARNED_EVOLVE` condition. Therefore, these values must explicitly be recorded as `not_measured`, and comparative claims regarding plasticity preservation are invalid.

## 7. Ablation Analysis
The ablation experiment (`ablation_no_memory`) executed successfully and recorded basic telemetry, but because interventions were never applied, the tracking of long-horizon consequences was meaningless. The ablation is technically **INCONCLUSIVE**.

## 8. Aggregation Analysis
The aggregation script (`analyze_results.py`) correctly identified files but failed to handle missing data constraints (e.g., averaging 0 for unmeasured plasticity probes). 

## 9. Reproducibility Analysis
The seeds and initial environments were fully reproducible. The failure lay not in reproducibility, but in the structural evolution mechanism itself being disabled.

## 10. Research-Claim Classification
- **1. EVOLVE performs structural evolution.** -> `NOT_SUPPORTED` (In current benchmark runs)
- **2. Deterministic EVOLVE performs structural evolution.** -> `NOT_SUPPORTED`
- **3. Learned EVOLVE performs structural evolution.** -> `NOT_SUPPORTED`
- **4. Learned policy actually controls structural decisions.** -> `NOT_SUPPORTED`
- **5. Learned policy benefits from InterventionMemory.** -> `INCONCLUSIVE`
- **6. Learned policy benefits from long-horizon consequences.** -> `INCONCLUSIVE`
- **7. Learned policy reduces catastrophic forgetting.** -> `NOT_SUPPORTED` (Differences due to RNG)
- **8. Learned policy improves OOD adaptation.** -> `NOT_SUPPORTED`
- **9. Learned policy preserves plasticity.** -> `NOT_TESTED` (Comparatively)
- **10. Learned policy generalizes to holdout tasks.** -> `NOT_SUPPORTED`
- **11. EVOLVE eliminates catastrophic forgetting.** -> `NOT_SUPPORTED`
- **12. EVOLVE learns an adaptive policy for its own structural evolution.** -> `INCONCLUSIVE`

## 11. Required Corrective Actions
- **Infrastructure Fix**: Connect `runner.execute()` properly within `harness.py`.
- **RNG Fix**: Wrap the plasticity probe in proper RNG state save/restore context managers to prevent leakage.
- **Rerun**: After verifying structural mutations actually occur, the entire benchmark suite must be re-executed.
