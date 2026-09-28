import random
import numpy as np
import torch
import copy

class IsolatedRNGContext:
    """
    A strict context manager that captures all relevant global random number 
    generator states upon entry, and restores them perfectly upon exit.
    This guarantees that stochastic operations executed within the context 
    do not contaminate or shift the RNG stream of the main trajectory.
    """
    def __init__(self):
        self.py_rng_state = None
        self.np_rng_state = None
        self.pt_rng_state = None
        self.cuda_rng_state = None
        
    def __enter__(self):
        self.py_rng_state = random.getstate()
        self.np_rng_state = np.random.get_state()
        self.pt_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_rng_state = torch.cuda.get_rng_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.py_rng_state)
        np.random.set_state(self.np_rng_state)
        torch.set_rng_state(self.pt_rng_state)
        if torch.cuda.is_available() and self.cuda_rng_state is not None:
            torch.cuda.set_rng_state(self.cuda_rng_state)
