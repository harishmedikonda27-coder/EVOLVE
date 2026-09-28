import sys
import torch
import random
import base64
import numpy as np
from brain.brain import ModularBrain
from evolution.mutation import MutationManager

def decode_mask(b64: str, shape: tuple):
    b = base64.b64decode(b64)
    arr = np.frombuffer(b, dtype=np.uint8)
    return arr.reshape(shape)

def get_mask_from_topo(topo, mask_id="mask1"):
    for node in topo["nodes"]:
        if node["type"] == "module":
            for layer in node["actual_layers"]:
                if layer.get("sparse", {}).get("enabled"):
                    if layer["sparse"]["mask_id"] == mask_id:
                        if not layer["sparse"].get("mask_b64"):
                            print(f"DEBUG: Found {mask_id} but no mask_b64 exists!")
                            return None
                        return decode_mask(layer["sparse"]["mask_b64"], (layer["output_dim"], layer["input_dim"]))
    print(f"DEBUG: Could not find {mask_id} in topology")
    return None

def test_rewire():
    torch.manual_seed(100)
    random.seed(100)
    
    brain = ModularBrain()
    mod = brain.registry.get_module("module_A")
    mut = MutationManager()
    
    with torch.no_grad():
        mod.mask1.view(-1)[:500] = 0.0
        
    topo_before = brain.topology_snapshot()
    mask_before = get_mask_from_topo(topo_before, "mask1")
    
    mut.execute_rewire(brain, sparsity_budget=0.2)
    
    topo_after = brain.topology_snapshot()
    mask_after = get_mask_from_topo(topo_after, "mask1")
    
    diff = mask_after.astype(int) - mask_before.astype(int)
    added = np.sum(diff > 0)
    removed = np.sum(diff < 0)
    
    print(f"REWIRE TEST:")
    print(f"  Added edges:   {added}")
    print(f"  Removed edges: {removed}")
    
    assert added > 0 and removed > 0, "Rewire did not change edges"

def test_grow():
    brain = ModularBrain()
    topo_before = brain.topology_snapshot()
    mut = MutationManager()
    
    mut.execute_grow(brain, step=0)
    
    topo_after = brain.topology_snapshot()
    print(f"GROW TEST:")
    print(f"  Module count before: {topo_before['active_module_count']}")
    print(f"  Module count after:  {topo_after['active_module_count']}")
    assert topo_after['active_module_count'] > topo_before['active_module_count'], "Module count did not increase"

if __name__ == "__main__":
    test_rewire()
    test_grow()
    print("RUNTIME TESTS PASSED")
