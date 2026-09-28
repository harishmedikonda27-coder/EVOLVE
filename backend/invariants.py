"""
Central invariant validation for EVOLVE.

These invariants must hold at all times during a training run.  They should be
called at critical junctures (after mutation, after rollback, before
checkpoint) to catch state corruption early.
"""

from typing import List, Optional, Dict, Any
import torch

class InvariantError(RuntimeError):
    """Raised when a structural invariant is violated."""
    pass

def validate_brain_state(
    brain,
    optimizer: Optional[torch.optim.Optimizer] = None,
    label: str = "",
) -> None:
    prefix = f"[INVARIANT:{label}] " if label else "[INVARIANT] "

    # 1. At least one active module
    active_ids = brain.registry.list_active_ids()
    if len(active_ids) == 0:
        raise InvariantError(f"{prefix}No active modules in registry.")

    # 2. Router output dimension consistency
    if brain.router.num_modules != len(active_ids):
        raise InvariantError(
            f"{prefix}Router num_modules ({brain.router.num_modules}) != active module count ({len(active_ids)})."
        )
    gate_out_features = brain.router.gate.out_features
    if gate_out_features != len(active_ids):
        raise InvariantError(
            f"{prefix}Router gate.out_features ({gate_out_features}) != active module count ({len(active_ids)})."
        )

    # 3. Every active module is valid
    for mod_id in active_ids:
        mod = brain.registry.get_module(mod_id)
        if mod.status != "active":
            raise InvariantError(f"{prefix}Module '{mod_id}' listed as active but status is '{mod.status}'.")
        if mod.input_dim != brain.latent_dim:
            raise InvariantError(f"{prefix}Module '{mod_id}' input_dim != latent_dim.")
        if mod.output_dim != brain.module_dim:
            raise InvariantError(f"{prefix}Module '{mod_id}' output_dim != module_dim.")

    # 4. Topology snapshot consistency
    topo = brain.topology_snapshot()
    topo_active = set(topo.get("active_ids", []))
    if topo_active != set(active_ids):
        raise InvariantError(f"{prefix}Topology active_ids != registry active_ids.")
    if topo["active_module_count"] != len(active_ids):
        raise InvariantError(f"{prefix}Topology active_module_count != registry count.")

    # 5. Optimizer <-> model parameter alignment
    device = next(brain.parameters()).device
    for p in brain.parameters():
        if p.device != device:
            raise InvariantError(f"{prefix}Device mismatch in model parameters: {p.device} vs {device}")

    if optimizer is not None:
        model_params = set(id(p) for p in brain.parameters())
        opt_params = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) in opt_params:
                    raise InvariantError(f"{prefix}Duplicate parameter in optimizer param_groups.")
                opt_params.add(id(p))
                
                # Check optimizer state device
                if p in optimizer.state:
                    for k, v in optimizer.state[p].items():
                        if isinstance(v, torch.Tensor) and v.device != device:
                            raise InvariantError(f"{prefix}Optimizer state device mismatch: {v.device} vs {device}")

        orphaned = model_params - opt_params
        leaked = opt_params - model_params
        if orphaned:
            raise InvariantError(f"{prefix}{len(orphaned)} model parameter(s) missing from optimizer.")
        if leaked:
            raise InvariantError(f"{prefix}{len(leaked)} optimizer parameter(s) not in model.")


def validate_system_state(agent, label: str = "") -> None:
    """Validate full system state, including target network and device consistency."""
    prefix = f"[INVARIANT:{label}] " if label else "[INVARIANT] "
    
    # 1. Validate online brain
    validate_brain_state(agent.brain, agent.optimizer, label)
    
    # 2. Validate Target Network
    if agent.target_brain is None:
        raise InvariantError(f"{prefix}Target brain is None.")
        
    validate_brain_state(agent.target_brain, None, label + "_target")
    
    # Target architecture matching
    online_active = agent.brain.registry.list_active_ids()
    target_active = agent.target_brain.registry.list_active_ids()
    if set(online_active) != set(target_active):
        raise InvariantError(f"{prefix}Target network modules {target_active} mismatch online network {online_active}.")
        
    # Parameter counts
    online_params = sum(p.numel() for p in agent.brain.parameters())
    target_params = sum(p.numel() for p in agent.target_brain.parameters())
    if online_params != target_params:
        raise InvariantError(f"{prefix}Target parameter count {target_params} mismatch online {online_params}.")
        
    # Device consistency across online/target
    online_device = next(agent.brain.parameters()).device
    target_device = next(agent.target_brain.parameters()).device
    if online_device != target_device:
        raise InvariantError(f"{prefix}Target device {target_device} mismatch online device {online_device}.")

