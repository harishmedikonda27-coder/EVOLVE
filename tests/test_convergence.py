import pytest
import numpy as np
from collections import deque

def test_convergence_variance_math():
    # If all values are identical, variance should be 0.0
    history = deque([0.0, 0.0, 0.0, 0.0, 0.0], maxlen=5)
    var = np.var(history)
    assert var == 0.0

    # Test realistic values
    history2 = deque([0.93, 0.92, 0.93, 0.93, 0.90], maxlen=5)
    var2 = np.var(history2)
    # Mean is 0.922, var is ((0.008)^2*3 + (-0.002)^2 + (-0.022)^2)/5 = (0.000064*3 + 0.000004 + 0.000484)/5 = 0.000136
    assert np.isclose(var2, 0.000136)
