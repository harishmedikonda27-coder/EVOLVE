import copy
import torch
import numpy as np
import random
from typing import Dict, Any, Tuple


class RollbackManager:
    """Manages full state checkpointing and restoration for reversible structural mutations.
    
    On rollback, this manager must restore:
    1. Module registry topology (active/pruned module set)
    2. Router output dimensions
    3. Model parameter weights
    4. Optimizer state and parameter groups
    5. Exact RNG state (Torch, Numpy, Python)
    """

    def capture_checkpoint(self, brain: Any, optimizer: Any, controller_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model_state_dict": copy.deepcopy(brain.state_dict()),
            "registry_metadata": copy.deepcopy(brain.registry.get_metadata()),
            "active_module_ids": copy.deepcopy(brain.registry.list_active_ids()),
            "all_module_ids": list(brain.registry.modules_dict.keys()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "controller_state": copy.deepcopy(controller_state),
            "router_num_modules": brain.router.num_modules,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "topology_revision": brain.topology_revision
        }

    def restore_checkpoint(self, brain: Any, optimizer: Any, checkpoint: Dict[str, Any]) -> Tuple[Any, Any]:
        target_ids = set(checkpoint["all_module_ids"])
        current_ids = set(brain.registry.modules_dict.keys())

        # 1. Remove modules that were added after checkpoint
        for mod_id in current_ids - target_ids:
            if mod_id in brain.registry.modules_dict:
                del brain.registry.modules_dict[mod_id]
                if mod_id in brain.registry.metadata_store:
                    del brain.registry.metadata_store[mod_id]
            if mod_id in brain.registry.all_historical_ids:
                brain.registry.all_historical_ids.remove(mod_id)

        # 2. Restore status of modules that existed at checkpoint time
        for mod_id in target_ids:
            if mod_id in brain.registry.modules_dict:
                ckpt_meta = checkpoint["registry_metadata"].get(mod_id, {})
                brain.registry.modules_dict[mod_id].status = ckpt_meta.get("status", "active")
                if mod_id in brain.registry.metadata_store:
                    brain.registry.metadata_store[mod_id]["status"] = ckpt_meta.get("status", "active")

        # 3. Restore router dimension
        target_router_count = checkpoint.get("router_num_modules", len(checkpoint["active_module_ids"]))
        brain.router.update_module_count(target_router_count)

        # 4. Restore model state dict (now topology matches)
        brain.load_state_dict(checkpoint["model_state_dict"])

        # 5. Restore optimizer strictly
        new_optimizer = torch.optim.Adam(brain.parameters(), lr=optimizer.defaults.get("lr", 0.001))
        new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Ensure optimizer states match model device
        device = next(brain.parameters()).device
        for state in new_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # 6. Restore RNG strictly
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        if "python_rng" in checkpoint:
            random.setstate(checkpoint["python_rng"])

        if "topology_revision" in checkpoint:
            brain.topology_revision = checkpoint["topology_revision"]

        return brain, new_optimizer
