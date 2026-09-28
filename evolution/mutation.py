import base64
from dataclasses import dataclass
from typing import Dict, Any, Optional
import uuid
import torch
import numpy as np

from brain.brain import ModularBrain


@dataclass
class RewireDiff:
    mutation_id: str
    revision_before: int
    revision_after: int
    before_edge_count: int
    after_edge_count: int
    added_edge_count: int
    removed_edge_count: int
    unchanged_edge_count: int
    parameter_count_before: int
    parameter_count_after: int
    module_count_before: int
    module_count_after: int
    added_edges: Optional[list] = None
    removed_edges: Optional[list] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "revision_before": self.revision_before,
            "revision_after": self.revision_after,
            "before_edge_count": self.before_edge_count,
            "after_edge_count": self.after_edge_count,
            "added_edge_count": self.added_edge_count,
            "removed_edge_count": self.removed_edge_count,
            "unchanged_edge_count": self.unchanged_edge_count,
            "parameter_count_before": self.parameter_count_before,
            "parameter_count_after": self.parameter_count_after,
            "module_count_before": self.module_count_before,
            "module_count_after": self.module_count_after,
            "added_edges": self.added_edges or [],
            "removed_edges": self.removed_edges or []
        }


@dataclass
class MutationRecord:
    mutation_id: str
    operation: str
    level_id: int
    task_profile: Dict[str, Any]
    need_score: float
    component_scores: Dict[str, float]
    reason: str
    topology_before: Dict[str, Any]
    topology_after: Optional[Dict[str, Any]] = None
    parameter_count_before: int = 0
    parameter_count_after: Optional[int] = None
    new_task_metric_before: Optional[float] = None
    new_task_metric_after: Optional[float] = None
    retention_metric_before: Optional[float] = None
    retention_metric_after: Optional[float] = None
    verification_steps: Optional[int] = None
    budget_before: int = 0
    budget_after: Optional[int] = None
    status: str = "PENDING"  # ACCEPT, REJECT, ROLLBACK
    rollback_reason: Optional[str] = None
    rewire_diff: Optional[RewireDiff] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "operation": self.operation,
            "level_id": self.level_id,
            "task_profile": self.task_profile,
            "NeedScore": self.need_score,
            "component scores contributing to the NeedScore": self.component_scores,
            "reason/evidence": self.reason,
            "topology_before": self.topology_before,
            "topology_after": self.topology_after,
            "parameter_count_before": self.parameter_count_before,
            "parameter_count_after": self.parameter_count_after,
            "new_task_metric_before": self.new_task_metric_before,
            "new_task_metric_after": self.new_task_metric_after,
            "retention_metric_before": self.retention_metric_before,
            "retention_metric_after": self.retention_metric_after,
            "verification_steps": self.verification_steps,
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "status": self.status,
            "rollback_reason": self.rollback_reason,
            "rewire_diff": self.rewire_diff.to_dict() if self.rewire_diff else None,
        }



def decode_mask(mask_b64: str) -> np.ndarray:
    if not mask_b64:
        return np.array([], dtype=np.uint8)
    return np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8)

def compute_rewire_diff(topo_before: Dict[str, Any], topo_after: Dict[str, Any], mutation_id: str) -> RewireDiff:
    added = 0
    removed = 0
    unchanged = 0
    before_count = 0
    after_count = 0
    added_edges = []
    removed_edges = []
    
    nodes_before = {n["id"]: n for n in topo_before.get("nodes", [])}
    nodes_after = {n["id"]: n for n in topo_after.get("nodes", [])}
    
    for nid, node_after in nodes_after.items():
        node_before = nodes_before.get(nid)
        
        # Dense vs Sparse counting logic
        if node_before and node_before.get("sparse", {}).get("enabled"):
            m_before = decode_mask(node_before["sparse"].get("mask_b64", ""))
            before_c = int(np.sum(m_before))
            before_count += before_c
            
            if node_after.get("sparse", {}).get("enabled"):
                m_after = decode_mask(node_after["sparse"].get("mask_b64", ""))
                after_c = int(np.sum(m_after))
                after_count += after_c
                
                # Compute diffs
                if len(m_before) == len(m_after) and len(m_before) > 0:
                    added_idx = np.where((m_after == 1) & (m_before == 0))[0]
                    removed_idx = np.where((m_after == 0) & (m_before == 1))[0]
                    
                    added += len(added_idx)
                    removed += len(removed_idx)
                    unchanged += int(np.sum((m_after == 1) & (m_before == 1)))
                    
                    in_dim = node_after.get("input_dim", 1)
                    for idx in added_idx:
                        added_edges.append({
                            "layer_id": nid,
                            "source_index": int(idx) % in_dim,
                            "target_index": int(idx) // in_dim,
                            "flat_index": int(idx)
                        })
                    for idx in removed_idx:
                        removed_edges.append({
                            "layer_id": nid,
                            "source_index": int(idx) % in_dim,
                            "target_index": int(idx) // in_dim,
                            "flat_index": int(idx)
                        })
            else:
                # Became dense
                params = node_after.get("parameter_count", 0)
                after_count += params
                added += (params - before_c)
                unchanged += before_c
        elif node_before and not node_before.get("sparse", {}).get("enabled"):
            params_b = node_before.get("parameter_count", 0)
            before_count += params_b
            
            if node_after.get("sparse", {}).get("enabled"):
                m_after = decode_mask(node_after["sparse"].get("mask_b64", ""))
                after_c = int(np.sum(m_after))
                after_count += after_c
                removed += (params_b - after_c)
                unchanged += after_c
            else:
                params_a = node_after.get("parameter_count", 0)
                after_count += params_a
                unchanged += params_a
                
    return RewireDiff(
        mutation_id=mutation_id,
        revision_before=topo_before.get("topology_revision", 0),
        revision_after=topo_after.get("topology_revision", 0),
        before_edge_count=before_count,
        after_edge_count=after_count,
        added_edge_count=added,
        removed_edge_count=removed,
        unchanged_edge_count=unchanged,
        parameter_count_before=topo_before.get("total_parameter_count", 0),
        parameter_count_after=topo_after.get("total_parameter_count", 0),
        module_count_before=topo_before.get("active_module_count", 0),
        module_count_after=topo_after.get("active_module_count", 0),
        added_edges=added_edges,
        removed_edges=removed_edges
    )

class MutationManager:
    def create_mutation_record(
        self,
        operation: str,
        level_id: int,
        task_profile: Dict[str, Any],
        need_score: float,
        component_scores: Dict[str, float],
        reason: str,
        topology_before: Dict[str, Any],
        parameter_count_before: int,
        budget_before: int,
    ) -> MutationRecord:
        mut_id = f"mut_{uuid.uuid4().hex[:8]}"
        return MutationRecord(
            mutation_id=mut_id,
            operation=operation,
            level_id=level_id,
            task_profile=task_profile,
            need_score=need_score,
            component_scores=component_scores,
            reason=reason,
            topology_before=topology_before,
            parameter_count_before=parameter_count_before,
            budget_before=budget_before,
        )

    def execute_grow(self, brain: ModularBrain, step: int) -> str:
        new_id = f"module_{uuid.uuid4().hex[:6]}"
        brain.add_module(module_id=new_id, created_step=step, origin_event="GROW")
        return new_id

    def execute_prune(self, brain: ModularBrain, module_id: str) -> None:
        """OBD-inspired pruning proxy already determines target."""
        brain.remove_module(module_id)
    def can_rewire(self, brain: ModularBrain, sparsity_budget: float = 0.2) -> bool:
        for mod_id in brain.registry.list_active_ids():
            mod = brain.registry.get_module(mod_id)
            for layer, mask in [(mod.fc1, mod.mask1), (mod.fc2, mod.mask2)]:
                active_idx = torch.nonzero(mask.flatten()).squeeze()
                if active_idx.dim() == 0 or active_idx.size(0) == 0:
                    continue
                n_drop = max(1, int(active_idx.size(0) * sparsity_budget))
                if active_idx.size(0) <= n_drop:
                    n_drop = max(1, active_idx.size(0) - 1)
                if n_drop <= 0:
                    continue
                
                inactive_mask = (mask.flatten() == 0)
                eligible_idx = torch.nonzero(inactive_mask).squeeze(-1)
                max_growable = eligible_idx.numel()
                n_drop = min(n_drop, max_growable)
                if n_drop > 0:
                    return True
        return False

    def execute_rewire(self, brain: ModularBrain, sparsity_budget: float = 0.2) -> bool:
        """RigL-inspired rewiring: drop lowest magnitude weights, grow highest gradients."""
        structural_change = False
        for mod_id in brain.registry.list_active_ids():
            mod = brain.registry.get_module(mod_id)
            
            for layer, mask in [(mod.fc1, mod.mask1), (mod.fc2, mod.mask2)]:
                active_idx = torch.nonzero(mask.flatten()).squeeze()
                
                if active_idx.dim() == 0 or active_idx.size(0) == 0:
                    continue
                    
                # Drop a fraction of currently active connections
                n_drop = max(1, int(active_idx.size(0) * sparsity_budget))
                
                if active_idx.size(0) <= n_drop:
                    # Can't drop everything, leave at least 1
                    n_drop = max(1, active_idx.size(0) - 1)
                    
                if n_drop <= 0:
                    continue
                
                # Drop score: magnitude
                weights = layer.weight.data.flatten()[active_idx]
                drop_scores = torch.abs(weights)
                
                # Grow score: gradient magnitude (if available, else random for SET baseline)
                # EXCLUDE edges that were just selected for dropping to guarantee structural delta
                inactive_mask = (mask.flatten() == 0)
                eligible_idx = torch.nonzero(inactive_mask).squeeze(-1)
                
                # To maintain strict parameter count, we cannot drop more than we can grow from the strictly inactive pool
                max_growable = eligible_idx.numel()
                n_drop = min(n_drop, max_growable)
                
                if n_drop <= 0:
                    continue
                    
                # Sort and drop lowest
                _, drop_idx_rel = torch.sort(drop_scores)
                to_drop = active_idx[drop_idx_rel[:n_drop]]
                mask.view(-1)[to_drop] = 0.0
                
                if eligible_idx.numel() > 0:
                    grow_count = n_drop
                    
                    if layer.weight.grad is not None:
                        grads = layer.weight.grad.flatten()[eligible_idx]
                        grow_scores = torch.abs(grads)
                        _, grow_idx_rel = torch.sort(grow_scores, descending=True)
                        to_grow = eligible_idx[grow_idx_rel[:grow_count]]
                    else:
                        # SET random fallback
                        perm = torch.randperm(eligible_idx.numel())
                        to_grow = eligible_idx[perm[:grow_count]]
                        
                    mask.view(-1)[to_grow] = 1.0
                    structural_change = True
                    
            mod.apply_masks()
        
        if structural_change:
            brain.topology_revision += 1
            
        return structural_change

