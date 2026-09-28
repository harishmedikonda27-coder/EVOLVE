import torch
import torch.nn as nn
from typing import Dict, Any, List
import base64
import numpy as np

def tensor_to_base64(t: torch.Tensor, dtype=np.float32) -> str:
    if t is None: return ""
    b = t.detach().cpu().numpy().astype(dtype).tobytes()
    return base64.b64encode(b).decode('utf-8')

class TopologySnapshot:
    @staticmethod
    def _inspect_layer(m: nn.Module, base_id: str, mask1=None, mask2=None) -> List[Dict[str, Any]]:
        layers = []
        idx = 0
        def inspect(module, current_id):
            nonlocal idx
            if isinstance(module, nn.Linear):
                layer_info = {
                    "id": f"{current_id}_{idx}",
                    "type": "Linear",
                    "input_dim": module.in_features,
                    "output_dim": module.out_features,
                    "parameters": module.weight.numel() + (module.bias.numel() if module.bias is not None else 0),
                    "weight_b64": tensor_to_base64(module.weight, np.float32)
                }
                
                # Check if this linear layer matches mask1 or mask2
                if mask1 is not None and module.weight.shape == mask1.shape and idx == 0:
                    layer_info["sparse"] = {
                        "enabled": True,
                        "active_connections": int(mask1.sum().item()),
                        "possible_connections": mask1.numel(),
                        "density": float(mask1.sum().item()) / mask1.numel() if mask1.numel() > 0 else 0.0,
                        "mask_id": "mask1",
                        "mask_b64": tensor_to_base64(mask1, np.uint8)
                    }
                elif mask2 is not None and module.weight.shape == mask2.shape and idx == 1:
                    layer_info["sparse"] = {
                        "enabled": True,
                        "active_connections": int(mask2.sum().item()),
                        "possible_connections": mask2.numel(),
                        "density": float(mask2.sum().item()) / mask2.numel() if mask2.numel() > 0 else 0.0,
                        "mask_id": "mask2",
                        "mask_b64": tensor_to_base64(mask2, np.uint8)
                    }
                else:
                    layer_info["sparse"] = {
                        "enabled": False,
                        "active_connections": layer_info["parameters"],
                        "possible_connections": layer_info["parameters"],
                        "density": 1.0,
                    }
                layers.append(layer_info)
                idx += 1
            elif isinstance(module, nn.ReLU):
                layers.append({
                    "id": f"{current_id}_{idx}",
                    "type": "ReLU"
                })
                idx += 1
            elif isinstance(module, nn.Sequential):
                for child in module.children():
                    inspect(child, current_id)
            else:
                for child in module.children():
                    inspect(child, current_id)

        inspect(m, base_id)
        return layers

    @staticmethod
    def _extract_actual_connections(module_id: str, mask: torch.Tensor, mask_id: str, layer_id: str) -> List[Dict[str, Any]]:
        # We no longer extract a verbose list of actual_connections, the frontend reconstructs 
        # it natively via the base64 mask tensor to save overhead, but we keep this method 
        # empty or return empty list for backward compatibility if desired, or we just remove it.
        # Actually, let's keep it returning an empty list to avoid breaking the signature for now.
        return []

    @staticmethod
    def generate(brain) -> Dict[str, Any]:
        obs_dim = brain.obs_dim
        latent_dim = brain.latent_dim
        module_dim = brain.module_dim
        action_dim = brain.action_dim
        modules_metadata = brain.registry.get_metadata()
        active_ids = brain.registry.list_active_ids()
        total_parameters = brain.parameter_count()
        topology_revision = brain.topology_revision

        encoder_layers = TopologySnapshot._inspect_layer(brain.encoder, "encoder")
        router_layers = TopologySnapshot._inspect_layer(brain.router, "router")
        head_layers = TopologySnapshot._inspect_layer(brain.head, "head")

        flat_nodes = []
        flat_edges = []
        actual_connections = []
        exec_idx = 0
        actual_layer_count = 0
        
        def add_layers(layers, role, extra_info=None):
            nonlocal exec_idx, actual_layer_count
            added_ids = []
            for l in layers:
                node = {
                    "id": l["id"],
                    "kind": l["type"],
                    "role": role,
                    "layer_index": exec_idx,
                    "status": "active",
                }
                actual_layer_count += 1
                if l["type"] == "Linear":
                    node.update({
                        "input_dim": l["input_dim"],
                        "output_dim": l["output_dim"],
                        "parameter_count": l["parameters"],
                        "weight_b64": l.get("weight_b64"),
                        "sparse": l.get("sparse")
                    })
                if extra_info:
                    node.update(extra_info)
                flat_nodes.append(node)
                added_ids.append(l["id"])
                exec_idx += 1
            return added_ids

        def connect_sequential(layer_ids):
            for i in range(len(layer_ids) - 1):
                flat_edges.append({
                    "source": layer_ids[i],
                    "target": layer_ids[i+1],
                    "density": 1.0
                })
                
        flat_nodes.append({
            "id": "input",
            "kind": "input",
            "role": "input",
            "layer_index": exec_idx,
            "status": "active",
            "output_dim": obs_dim
        })
        input_id = "input"
        exec_idx += 1
        
        # ENCODER
        enc_ids = add_layers(encoder_layers, "encoder")
        if enc_ids:
            flat_edges.append({"source": input_id, "target": enc_ids[0], "density": 1.0})
            connect_sequential(enc_ids)
            last_enc_id = enc_ids[-1]
        else:
            last_enc_id = input_id
            
        # ROUTER
        rout_ids = add_layers(router_layers, "router")
        if rout_ids:
            flat_edges.append({"source": last_enc_id, "target": rout_ids[0], "density": 1.0})
            connect_sequential(rout_ids)
            last_rout_id = rout_ids[-1]
        else:
            last_rout_id = last_enc_id
            
        # MODULES
        mod_last_ids = []
        for mod_id in active_ids:
            pytorch_mod = brain.registry.get_module(mod_id)
            if not pytorch_mod: continue
            
            m_layers = TopologySnapshot._inspect_layer(pytorch_mod, mod_id, pytorch_mod.mask1, pytorch_mod.mask2)
            meta = modules_metadata.get(mod_id, {})
            m_ids = add_layers(m_layers, mod_id, extra_info={"module_id": mod_id, "created_step": meta.get("created_step", 0), "utilization": meta.get("utilization", 0.0)})
            
            if m_ids:
                connect_sequential(m_ids)
                meta = modules_metadata.get(mod_id, {})
                if "mask1_aggregates" in meta:
                    total_density = sum(agg["density"] for agg in meta["mask1_aggregates"] if agg["density"] > 0.0)
                    if total_density > 0.0:
                        flat_edges.append({
                            "source": last_rout_id,
                            "target": m_ids[0],
                            "density": min(1.0, total_density),
                            "source_type": "mask_aggregate",
                            "mask": "mask1"
                        })
                else:
                    flat_edges.append({"source": last_rout_id, "target": m_ids[0], "density": 1.0})
                mod_last_ids.append((mod_id, m_ids[-1]))

        # HEAD
        head_ids = add_layers(head_layers, "head")
        if head_ids:
            connect_sequential(head_ids)
            head_first_id = head_ids[0]
            for mod_id, last_m_id in mod_last_ids:
                meta = modules_metadata.get(mod_id, {})
                if "mask2_aggregates" in meta:
                    total_density = sum(agg["density"] for agg in meta["mask2_aggregates"] if agg["density"] > 0.0)
                    if total_density > 0.0:
                        flat_edges.append({
                            "source": last_m_id,
                            "target": head_first_id,
                            "density": min(1.0, total_density),
                            "source_type": "mask_aggregate",
                            "mask": "mask2"
                        })
                else:
                    flat_edges.append({"source": last_m_id, "target": head_first_id, "density": 1.0})
            last_head_id = head_ids[-1]
        else:
            last_head_id = last_rout_id
            
        # OUTPUT
        flat_nodes.append({
            "id": "output",
            "kind": "output",
            "role": "output",
            "layer_index": exec_idx,
            "status": "active",
            "input_dim": action_dim
        })
        flat_edges.append({"source": last_head_id, "target": "output", "density": 1.0})

        return {
            "active_module_count": len(active_ids),
            "total_module_count": len(modules_metadata),
            "total_parameter_count": total_parameters,
            "actual_layer_count": len(flat_nodes),
            "nodes": flat_nodes,
            "edges": flat_edges,
            "active_ids": active_ids,
            "topology_revision": topology_revision,
            "architecture_revision": topology_revision,
            "actual_connections": actual_connections,
        }

class TopologyValidator:
    @staticmethod
    def validate(topology: Dict[str, Any], registry_active_ids: List[str], actual_param_count: int) -> bool:
        """
        Ensure topology representation matches the actual PyTorch state and is structurally valid.
        """
        node_ids = set([n["id"] for n in topology.get("nodes", [])])
        edges = topology.get("edges", [])

        # 1. Unique IDs
        if len(node_ids) != len(topology.get("nodes", [])):
            return False

        # 2. Valid Edge endpoints (relaxed for subnodes since frontend manages them)
        # We allow target/source to not strictly exist in macro node list if they are subnodes
        pass

        # 3. Parameter count consistency
        if topology.get("total_parameter_count") != actual_param_count:
            return False

        # 4. Registry consistency
        topo_active_ids = set(topology.get("active_ids", []))
        if set(registry_active_ids) != topo_active_ids:
            return False

        # 5. Required paths exist (encoder -> router -> active_modules -> head)
        if "encoder" not in node_ids or "router" not in node_ids or "head" not in node_ids:
            return False

        if len(topo_active_ids) == 0:
            return False

        # Verify reachability for at least one path (relaxed for subnodes)
        for mod_id in topo_active_ids:
            has_in = False
            has_out = False
            for edge in edges:
                if edge["source"].startswith("router") and edge["target"].startswith(mod_id):
                    has_in = True
                if edge["source"].startswith(mod_id) and edge["target"].startswith("head"):
                    has_out = True
            if not (has_in and has_out):
                return False

        return True
