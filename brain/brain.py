import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional

from brain.encoder import Encoder
from brain.module import EvolveModule
from brain.router import Router
from brain.head import OutputHead
from brain.registry import ModuleRegistry
from brain.topology import TopologySnapshot


class ModularBrain(nn.Module):
    def __init__(
        self,
        obs_dim: int = 16,
        latent_dim: int = 32,
        module_dim: int = 64,
        action_dim: int = 4,
        initial_module_id: str = "module_A",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.module_dim = module_dim
        self.action_dim = action_dim

        self.encoder = Encoder(obs_dim, latent_dim)
        self.registry = ModuleRegistry()
        
        # Create initial module
        initial_module = EvolveModule(
            module_id=initial_module_id,
            input_dim=latent_dim,
            output_dim=module_dim,
            hidden_dim=64,
            created_step=0,
            origin_event="initial",
            status="active",
        )
        self.registry.add_module(initial_module)

        active_count = len(self.registry.list_active_ids())
        self.router = Router(latent_dim, num_modules=active_count)
        self.head = OutputHead(module_dim, action_dim)

        self.last_routing_weights: Optional[torch.Tensor] = None
        self.topology_revision = 0

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(obs)                      # [batch, latent_dim]
        routing_weights = self.router(latent)           # [batch, num_active_modules]
        self.last_routing_weights = routing_weights.detach()

        active_ids = self.registry.list_active_ids()
        module_outputs = []
        for mod_id in active_ids:
            mod = self.registry.get_module(mod_id)
            mod.apply_masks() # Enforce RigL sparsity
            out = mod(latent)                           # [batch, module_dim]
            module_outputs.append(out)

        stacked = torch.stack(module_outputs, dim=1)   # [batch, N_active, module_dim]
        combined = (routing_weights.unsqueeze(-1) * stacked).sum(dim=1) # [batch, module_dim]

        action_logits = self.head(combined)             # [batch, action_dim]
        return action_logits, routing_weights

    def add_module(
        self,
        module_id: str,
        created_step: int = 0,
        origin_event: str = "GROW",
        associated_tasks: Optional[List[str]] = None,
    ) -> str:
        device = next(self.parameters()).device
        new_module = EvolveModule(
            module_id=module_id,
            input_dim=self.latent_dim,
            output_dim=self.module_dim,
            hidden_dim=64,
            created_step=created_step,
            origin_event=origin_event,
            status="active",
        ).to(device)
        added_id = self.registry.add_module(new_module, associated_tasks)
        active_count = len(self.registry.list_active_ids())
        self.router.update_module_count(active_count)
        self.topology_revision += 1
        return added_id

    def remove_module(self, module_id: str) -> None:
        active_ids = self.registry.list_active_ids()
        if len(active_ids) <= 1:
            raise RuntimeError("Cannot prune the last active module in ModularBrain.")
        self.registry.remove_module(module_id)
        new_active_count = len(self.registry.list_active_ids())
        self.router.update_module_count(new_active_count)
        self.topology_revision += 1

    def parameter_count(self) -> int:
        count = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        count += sum(p.numel() for p in self.router.parameters() if p.requires_grad)
        count += sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        # RigL-aware active parameter count
        for mod_id in self.registry.list_active_ids():
            mod = self.registry.get_module(mod_id)
            count += mod.metadata()["parameter_count"]
            
        return count

    def topology_snapshot(self) -> Dict[str, Any]:
        return TopologySnapshot.generate(self)
