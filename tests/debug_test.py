import torch
import base64
import numpy as np
from brain.brain import ModularBrain
from evolution.mutation import compute_rewire_diff, MutationManager
from brain.topology import tensor_to_base64

def debug():
    brain = ModularBrain(10, 5)
    mod_id = "test_mod"
    brain.add_module(mod_id, created_step=0, origin_event="TEST")
    
    topo_before = brain.topology_snapshot()
    
    mod = brain.registry.get_module(mod_id)
    mod.fc1.weight.grad = torch.randn_like(mod.fc1.weight)
    mod.fc2.weight.grad = torch.randn_like(mod.fc2.weight)
    
    print("Before rewire sum:", mod.mask1.sum().item())
    
    mut_manager = MutationManager()
    mut_manager.execute_rewire(brain, sparsity_budget=0.2)
    
    print("After rewire sum:", mod.mask1.sum().item())
    
    topo_after = brain.topology_snapshot()
    
    b64_before = [n for n in topo_before["nodes"] if n["id"] == "test_mod_0"][0]["sparse"]["mask_b64"]
    b64_after = [n for n in topo_after["nodes"] if n["id"] == "test_mod_0"][0]["sparse"]["mask_b64"]
    
    print("Base64 equal:", b64_before == b64_after)
    
    diff = compute_rewire_diff(topo_before, topo_after, "mut_1")
    print(diff)

debug()
