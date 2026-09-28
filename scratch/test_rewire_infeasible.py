import os
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.brain import ModularBrain
from evolution.mutation import MutationManager

def test_rewire_infeasible():
    brain = ModularBrain(10, 2)
    brain.add_module("m1", 0, "INIT")
    
    for mod_id in brain.registry.list_active_ids():
        mod = brain.registry.get_module(mod_id)
        mod.mask1.data.zero_()
        mod.mask2.data.zero_()
    
    mut_manager = MutationManager()
    structural_change = mut_manager.execute_rewire(brain)
    
    print(f"Structural Change: {structural_change}")
    assert structural_change is False, "Expected REWIRE to be infeasible and return False"

if __name__ == '__main__':
    test_rewire_infeasible()
