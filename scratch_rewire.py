import sys
import torch
import random
from brain.brain import ModularBrain
from evolution.mutation import MutationManager

def main():
    torch.manual_seed(100)
    random.seed(100)
    
    brain = ModularBrain()
    mod = brain.registry.get_module("module_A")
    mut = MutationManager()
    
    # Introduce artificial sparsity so REWIRE actually swaps different elements
    with torch.no_grad():
        mod.mask1.view(-1)[:500] = 0.0
        mod.mask2.view(-1)[:500] = 0.0
    
    print("=== BEFORE REWIRE ===")
    topo1 = brain.topology_snapshot()
    rev1 = topo1["topology_revision"]
    edges1 = [e for e in topo1["edges"] if e.get("source_type") == "mask_aggregate"]
    mask1_before = mod.mask1.clone()
    print(f"Topology Revision: {rev1}")
    print(f"Module Count: {topo1['active_module_count']}")
    print(f"Active Mask1 Stats: {mask1_before.sum().item()} active")
    print(f"Aggregate edges: {len(edges1)}")
    
    # Force rewire
    print("\n=== MUTATING ===")
    mut.execute_rewire(brain, sparsity_budget=0.2)
    
    print("\n=== AFTER REWIRE ===")
    topo2 = brain.topology_snapshot()
    rev2 = topo2["topology_revision"]
    edges2 = [e for e in topo2["edges"] if e.get("source_type") == "mask_aggregate"]
    mask1_after = mod.mask1.clone()
    print(f"Topology Revision: {rev2}")
    print(f"Module Count: {topo2['active_module_count']}")
    print(f"Active Mask1 Stats: {mask1_after.sum().item()} active")
    print(f"Aggregate edges: {len(edges2)}")
    
    # Calculate mask changes
    added_mask = (mask1_after > mask1_before).sum().item()
    removed_mask = (mask1_before > mask1_after).sum().item()
    
    # Calculate aggregate changes
    def edge_key(e):
        return f"{e['source']}->{e['target']}"
        
    e1_map = {edge_key(e): e["density"] for e in edges1}
    e2_map = {edge_key(e): e["density"] for e in edges2}
    
    changed_edges = 0
    for k in e1_map.keys():
        if e1_map[k] != e2_map.get(k, 0):
            changed_edges += 1
            
    print("\n=== REPORT ===")
    print(f"Mask entries added: {added_mask}")
    print(f"Mask entries removed: {removed_mask}")
    print(f"Aggregate visual edges changed: {changed_edges}")
    
if __name__ == "__main__":
    main()
