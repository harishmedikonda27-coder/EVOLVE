import torch.nn as nn
from typing import Dict, List, Optional, Any
from brain.module import EvolveModule


class ModuleRegistry(nn.Module):
    def __init__(self):
        super().__init__()
        self.modules_dict = nn.ModuleDict()
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.all_historical_ids: set = set()

    def add_module(self, module: EvolveModule, associated_tasks: Optional[List[str]] = None) -> str:
        mod_id = module.module_id
        if mod_id in self.all_historical_ids:
            raise ValueError(f"Module ID '{mod_id}' has already been used. Historical IDs cannot be reused.")

        self.modules_dict[mod_id] = module
        self.all_historical_ids.add(mod_id)

        meta = module.metadata()
        meta["associated_tasks"] = associated_tasks or ["A"]
        meta["utilization"] = 0.0
        self.metadata_store[mod_id] = meta
        return mod_id

    def deactivate_module(self, module_id: str) -> None:
        if module_id not in self.modules_dict:
            raise KeyError(f"Module ID '{module_id}' not found in registry.")
        self.modules_dict[module_id].status = "inactive"
        self.metadata_store[module_id]["status"] = "inactive"

    def remove_module(self, module_id: str) -> None:
        if module_id not in self.modules_dict:
            raise KeyError(f"Module ID '{module_id}' not found in registry.")
        self.modules_dict[module_id].status = "pruned"
        self.metadata_store[module_id]["status"] = "pruned"

    def get_module(self, module_id: str) -> EvolveModule:
        return self.modules_dict[module_id]

    def list_active_ids(self) -> List[str]:
        return [
            mod_id
            for mod_id, mod in self.modules_dict.items()
            if mod.status == "active"
        ]

    def get_metadata(self) -> Dict[str, Dict[str, Any]]:
        for mod_id, mod in self.modules_dict.items():
            meta = mod.metadata()
            meta.update({
                "associated_tasks": self.metadata_store[mod_id].get("associated_tasks", []),
                "utilization": self.metadata_store[mod_id].get("utilization", 0.0),
            })
            self.metadata_store[mod_id] = meta
        return self.metadata_store
