import os
import json
import torch
import numpy as np
import random
from typing import Dict, Any, Optional, Tuple


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "storage/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        checkpoint_id: str,
        brain: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: Dict[str, Any],
        meta_trainer = None,
        consequence_tracker = None,
        intervention_memory = None,
        ood_monitor = None
    ) -> str:
        ckpt_folder = os.path.join(self.checkpoint_dir, checkpoint_id)
        os.makedirs(ckpt_folder, exist_ok=True)

        model_path = os.path.join(ckpt_folder, "model.pt")
        opt_path = os.path.join(ckpt_folder, "optimizer.pt")
        rng_path = os.path.join(ckpt_folder, "rng.pt")
        topo_path = os.path.join(ckpt_folder, "topology.json")
        manifest_path = os.path.join(ckpt_folder, "manifest.json")

        torch.save(brain.state_dict(), model_path)
        torch.save(optimizer.state_dict(), opt_path)
        torch.save({
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()
        }, rng_path)

        topology = brain.topology_snapshot()
        with open(topo_path, "w", encoding="utf-8") as f:
            json.dump(topology, f, indent=2)

        manifest = {
            "checkpoint_id": checkpoint_id,
            "metadata": metadata,
            "active_module_ids": brain.registry.list_active_ids(),
            "has_phase2": False
        }
        
        if meta_trainer is not None:
            manifest["has_phase2"] = True
            p2_path = os.path.join(ckpt_folder, "phase2.pt")
            p2_state = {
                "meta_trainer": meta_trainer.get_state()
            }
            if consequence_tracker is not None:
                p2_state["pending_interventions"] = consequence_tracker.pending
                p2_state["completed_tasks_since_decision"] = consequence_tracker.completed_tasks_since_decision
            if intervention_memory is not None:
                p2_state["intervention_buffer"] = list(intervention_memory.buffer)
            if ood_monitor is not None:
                p2_state["ood_ref"] = list(ood_monitor.reference_window)
                p2_state["ood_curr"] = list(ood_monitor.current_window)
                p2_state["ood_locked"] = ood_monitor.is_reference_locked
                
            torch.save(p2_state, p2_path)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return ckpt_folder

    def load_checkpoint(
        self,
        checkpoint_id: str,
        brain: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        meta_trainer = None,
        consequence_tracker = None,
        intervention_memory = None,
        ood_monitor = None
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
        ckpt_folder = os.path.join(self.checkpoint_dir, checkpoint_id)
        manifest_path = os.path.join(ckpt_folder, "manifest.json")
        model_path = os.path.join(ckpt_folder, "model.pt")
        opt_path = os.path.join(ckpt_folder, "optimizer.pt")
        rng_path = os.path.join(ckpt_folder, "rng.pt")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        brain.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(opt_path))

        if os.path.exists(rng_path):
            rng_state = torch.load(rng_path)
            torch.set_rng_state(rng_state["torch_rng"])
            np.random.set_state(rng_state["numpy_rng"])
            random.setstate(rng_state["python_rng"])
            
        p2_path = os.path.join(ckpt_folder, "phase2.pt")
        if manifest.get("has_phase2", False) and os.path.exists(p2_path):
            p2_state = torch.load(p2_path)
            if meta_trainer is not None and "meta_trainer" in p2_state:
                meta_trainer.load_state(p2_state["meta_trainer"])
                
            if consequence_tracker is not None and "pending_interventions" in p2_state:
                consequence_tracker.pending = p2_state["pending_interventions"]
                consequence_tracker.completed_tasks_since_decision = p2_state["completed_tasks_since_decision"]
                
            if intervention_memory is not None and "intervention_buffer" in p2_state:
                intervention_memory.buffer.clear()
                intervention_memory.buffer.extend(p2_state["intervention_buffer"])
                
            if ood_monitor is not None and "ood_ref" in p2_state:
                ood_monitor.reference_window.clear()
                ood_monitor.current_window.clear()
                ood_monitor.reference_window.extend(p2_state["ood_ref"])
                ood_monitor.current_window.extend(p2_state["ood_curr"])
                ood_monitor.is_reference_locked = p2_state.get("ood_locked", False)

        return brain, optimizer, manifest.get("metadata", {})
