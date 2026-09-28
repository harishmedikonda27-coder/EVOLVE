# EVOLVE: Visual System Architecture

This document presents the definitive engineering architecture of the EVOLVE self-adaptive neural framework, mapping directly to the actual PyTorch and Next.js implementations.

## System Flow & Decision Pipeline

```mermaid
flowchart TD
    %% Styling
    classDef user fill:#09090b,stroke:#52525b,stroke-width:2px,color:#fafafa,font-weight:bold
    classDef frontend fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    classDef backend fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc
    classDef core fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    classDef brain fill:#022c22,stroke:#059669,stroke-width:2px,color:#f8fafc
    classDef controller fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#f8fafc
    classDef mutation fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#f8fafc
    classDef terminal fill:#171717,stroke:#404040,stroke-width:2px,color:#a3a3a3

    USER(["USER / JUDGE"]):::user

    subgraph FRONTEND ["EVOLVE FRONTEND (React / Next.js)"]
        direction TB
        F_DASH["Research Dashboard"]
        F_ENV["Live Environment View"]
        F_VIS["Topology & Actual PyTorch Vis"]
        F_TEL["Real-time Telemetry"]
    end
    FRONTEND:::frontend

    API["FASTAPI API LAYER"]:::backend
    ORCH["EXPERIMENT ORCHESTRATOR"]:::backend
    TRAIN["RL TRAINING ENGINE (PyTorch)"]:::core

    subgraph BRAIN ["DYNAMIC NEURAL BRAIN"]
        direction TB
        B_ENC["Encoder"] --> B_ROUT["Dynamic Router"]
        B_ROUT --> B_MOD["Sparse Modules (EvolveModule)"]
        B_MOD --> B_HEAD["Output Head"]
    end
    BRAIN:::brain

    ENV["ENVIRONMENT (Families & Levels)"]:::core
    REWARD(("REWARD & SIGNALS")):::controller
    CTRL["EVOLUTION CONTROLLER"]:::controller

    USER --> FRONTEND
    FRONTEND --> API
    API --> ORCH
    ORCH --> TRAIN
    TRAIN --> BRAIN
    BRAIN --> ENV
    ENV --> REWARD
    REWARD --> CTRL

    %% Mutation Decisions
    CTRL --> D_REUSE["REUSE"]:::mutation
    CTRL --> D_ADAPT["ADAPT"]:::mutation
    CTRL --> D_NOCHANGE["NO CHANGE"]:::mutation
    
    CTRL --> D_GROW["GROW"]:::mutation
    CTRL --> D_PRUNE["PRUNE"]:::mutation
    CTRL --> D_REWIRE["REWIRE"]:::mutation

    %% Direct to next step
    D_REUSE --> NEXT["NEXT TRAINING STEP"]:::terminal
    D_ADAPT --> NEXT
    D_NOCHANGE --> NEXT

    %% Verification Pipeline
    D_GROW --> VERIFY["VERIFICATION"]:::controller
    D_PRUNE --> VERIFY
    D_REWIRE --> VERIFY

    VERIFY --> V_ACCEPT["ACCEPT"]:::core
    VERIFY --> V_ROLLBACK["ROLLBACK"]:::mutation

    V_ACCEPT --> NEXT
    V_ROLLBACK --> RESTORE["RollbackManager"]:::controller
    RESTORE --> NEXT

    NEXT --> TRAIN
```

---

## Architecture Implementation Mapping

The diagram above translates directly into the following codebase implementations:

### 1. EVOLVE Frontend
**Location:** `frontend/app/` and `frontend/components/`
- **Dashboard (`page.tsx`):** Coordinates the overall research view.
- **Visualizers (`BrainTopology.tsx` & `ActualArchitectureView.tsx`):** Receives base64-encoded PyTorch float arrays and binary masks to render exact neural representations via HTML5 Canvas and SVG.
- **Telemetry & State (`useEvolveState.ts`):** Maintains synchronized state via WebSockets.

### 2. FastAPI API & Orchestrator
**Location:** `backend/main.py`, `backend/websocket.py`, `backend/events.py`
- **FastAPI Layer:** Handles REST initialization (`/runs`).
- **Orchestrator:** Spins up the experiment background thread (`_run_experiment_thread`).
- **Event Bus:** Streams state updates (`TOPOLOGY_SNAPSHOT`, `METRICS_UPDATE`) via `ConnectionManager` to the frontend.

### 3. Training Engine
**Location:** `training/train_evolve.py`, `training/agent.py`
- **Engine Loop:** Orchestrates the standard PyTorch RL loop (`DQNAgent`).
- **Convergence (`training/convergence.py`):** Utilizes OLS slope variance to detect statistically principled plateaus, preventing infinite training cycles.

### 4. Dynamic Neural Brain
**Location:** `brain/brain.py`, `brain/module.py`, `brain/router.py`, `brain/topology.py`
- **Encoder -> Router:** Latent features are dynamically weighted across available active modules.
- **Sparse Modules (`EvolveModule`):** Utilize RigL-inspired binary masks (`mask1`, `mask2`) to enforce sparse connectivity.
- **Topology Snapshot:** Recursively inspects the PyTorch computational graph to serialize the exact architecture.

### 5. Environment System
**Location:** `environment/base.py`, `environment/families.py`, `environment/generators.py`
- Generates procedurally scaled tasks (`HierarchicalNavigation`, `MemoryObservation`, `AdaptiveDynamics`).
- Ensures BFS solvability upon generation via the `GridGenerator`.

### 6. Evolution Controller
**Location:** `evolution/controller.py`, `evolution/policies.py`, `evolution/signals.py`
- Receives reward plateaus and gradient statistics.
- Derives a singular `NeedScore` driving the structural decision policy.

### 7. Mutation & Decision Flow (GROW, PRUNE, REWIRE)
**Location:** `evolution/mutation.py`
- **GROW:** Adds a new `EvolveModule` to the registry, expanding router dimensionality.
- **PRUNE:** Drops severely underutilized modules.
- **REWIRE:** Executes RigL updates—dropping low-magnitude weights and activating dormant weights with high gradients.
- **REUSE / ADAPT:** Retains existing topology, delegating strictly to parameter gradient updates.

### 8. Verification & Rollback
**Location:** `evolution/verify.py`, `evolution/rollback.py`
- **Checkpointing:** Before speculative topology shifts, the `RollbackManager` strictly captures PyTorch `state_dict`, optimizer momentums, and RNG states.
- **Verification:** The `Verifier` tests new architectures against the current task and past task retention.
- **Rollback:** If structural cost outweighs performance gain, or if catastrophic forgetting occurs, the exact prior state is strictly restored. Verified by invariants in `tests/test_invariants.py`.
