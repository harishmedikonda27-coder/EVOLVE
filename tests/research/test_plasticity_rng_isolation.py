import pytest
import torch
import numpy as np
import random
from utils.rng_isolation import IsolatedRNGContext

def test_rng_isolation_preserves_main_trajectory():
    """
    Prove that executing code inside the IsolatedRNGContext completely isolates
    stochastic operations and leaves the main RNG trajectory identical.
    """
    # Seed globally
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate some main trajectory numbers
    main_py_1 = random.random()
    main_np_1 = np.random.rand()
    main_pt_1 = torch.rand(1).item()
    
    # Enter probe context
    with IsolatedRNGContext():
        # Do stochastic stuff inside
        _ = random.random()
        _ = np.random.rand(100)
        _ = torch.rand(50)
        
    # Generate next main trajectory numbers
    main_py_2 = random.random()
    main_np_2 = np.random.rand()
    main_pt_2 = torch.rand(1).item()
    
    # Now reset and do it without the probe
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    alt_py_1 = random.random()
    alt_np_1 = np.random.rand()
    alt_pt_1 = torch.rand(1).item()
    
    alt_py_2 = random.random()
    alt_np_2 = np.random.rand()
    alt_pt_2 = torch.rand(1).item()
    
    # Assert perfectly identical
    assert main_py_1 == alt_py_1
    assert main_np_1 == alt_np_1
    assert main_pt_1 == alt_pt_1
    
    assert main_py_2 == alt_py_2, "Python RNG leaked!"
    assert main_np_2 == alt_np_2, "NumPy RNG leaked!"
    assert main_pt_2 == alt_pt_2, "PyTorch RNG leaked!"
