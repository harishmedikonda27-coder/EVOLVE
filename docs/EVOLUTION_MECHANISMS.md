# EVOLVE: Structural Evolution Mechanisms

This document formalizes the research-inspired structural operations and the EVOLVE-specific engineering designs. 
It explicitly distinguishes established literature from EVOLVE-specific heuristics.

## 1. ESTABLISHED RESEARCH MECHANISMS

### 1.1 DEN-Inspired Dynamic Capacity (GROW)
**Reference**: Yoon et al., "Lifelong Learning with Dynamically Expandable Networks" (DEN).
**Mechanism**: Dynamically expanding network capacity when learning new tasks that cannot be solved by the existing architecture.
**Implementation in EVOLVE**:
When the system identifies persistent structural insufficiency (high NeedScore, low performance, stable stagnation), it initiates a GROW mutation. A new computational module is instantiated and added to the network's active registry. The capacity is selectively expanded while keeping old parameters accessible.

### 1.2 OBD-Inspired Pruning (PRUNE)
**Reference**: LeCun et al., "Optimal Brain Damage" (OBD).
**Mechanism**: Removing parameters with low estimated impact on the training loss based on the diagonal Hessian.
**Implementation in EVOLVE**:
Exact Hessian computation is prohibitive due to VRAM constraints. We use an **OBD-inspired diagonal curvature proxy**.
The running curvature approximation is maintained via:
$H_{hat\_i}(t) = \beta H_{hat\_i}(t-1) + (1-\beta) g_i(t)^2$
The saliency of parameter $i$ is calculated as:
$S_i = 0.5 \cdot H_{hat\_i} \cdot w_i^2$
Parameters or modules with low aggregate saliency, combined with low empirical utilization and task-coverage redundancy, are pruned. **This is explicitly an approximation and not exact OBD.**

### 1.3 RigL-Inspired Dynamic Rewiring (REWIRE)
**Reference**: Evci et al., "Rigging the Lottery: Making All Tickets Winners" (RigL).
**Mechanism**: Maintaining sparse topologies by dropping low-magnitude weights and growing high-gradient inactive connections.
**Implementation in EVOLVE**:
For an active connection, the drop score is:
$D_{ij} = |w_{ij}|$
For an eligible inactive connection, the grow score is:
$G_{ij} = |\partial L / \partial w_{ij}|$
During a REWIRE event, a configurable per-layer sparsity budget is maintained. Connections with the lowest $D_{ij}$ are dropped, and inactive connections with the highest $G_{ij}$ are grown. Critical input/output/router connectivity is preserved to maintain structural validity.

### 1.4 SET (Sparse Evolutionary Training) Baseline
**Reference**: Mocanu et al.
**Mechanism**: Removing weak connections and replacing them with random sparse connections.
**Implementation in EVOLVE**:
SET acts purely as a baseline comparison mechanism, not the primary learning logic for the EVOLVE framework.

---

## 2. EVOLVE-SPECIFIC EXPERIMENTAL DESIGN

### 2.1 The EVOLVE Controller & Need Score
**Nature**: EVOLVE-specific experimental formulation.
The controller calculates an abstract `NeedScore` to determine structural pressure:
$\text{NeedScore} = w_P \cdot P + w_S \cdot S + w_N \cdot N + w_I \cdot I - w_C \cdot C$
Where:
- **P (Performance Gap)**: Normalized gap to target performance.
- **S (Stagnation)**: Based on slope of recent performance evaluation windows.
- **N (Novelty)**: Task profile distance from previously learned tasks.
- **I (Interference)**: Measured actual degradation on previous tasks ($R_{before} - R_{after}$).
- **C (Cost)**: Normalized structural compute burden.

This is **not** a published universal formula. It is EVOLVE's heuristic for balancing structural growth against stability and cost.

### 2.2 Transactional Mutations & Rollback
**Nature**: EVOLVE-specific engineering design.
All structural changes (GROW, PRUNE, REWIRE) are treated as **transactions**.
1. **Checkpoint**: Exact state (RNG, topology, optimizer, model) is saved.
2. **Mutate & Validate**: Operation applied, optimizer state rebuilt, numerical validity checked.
3. **Verify**: Trained for a bounded verification budget.
4. **Evaluate**: Checked for New Task Gain and Old Task Retention.
5. **Accept / Rollback**: If criteria are missed, the exact checkpoint state is restored, wiping any partial mutation state.

### 2.3 Structural Budgets & Anti-Oscillation
**Nature**: EVOLVE-specific engineering safeguard.
- **Cooldowns**: Minimum mutation intervals to prevent rapid cycling.
- **Minimum Module Lifetime**: Prevents immediate Pruning of a just-Grown module.
- **Limits**: Maximum modules, parameters, and mutations per level enforced globally.
