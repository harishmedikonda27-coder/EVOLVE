import pytest
from collections import deque
import numpy as np

def test_convergence_saturation_rejection():
    # Synthetic history: raw values fluctuate heavily, but normalized clamps to exactly 0.0
    # Because target is 0.8, even if variance is 0.0, performance_target_met must be False
    
    # Simulate the logic in run.py
    eval_history = deque([0.0, 0.0, 0.0, 0.0, 0.0], maxlen=5)
    target_score = 0.8
    
    performance_target_met = all(x >= target_score for x in eval_history) if len(eval_history) == 5 else False
    stable_windows = len(eval_history)
    variance = np.var(eval_history) if len(eval_history) > 1 else 0.0
    
    # 0.000 variance, 5 windows, but target NOT met!
    assert variance == 0.0
    assert stable_windows == 5
    assert performance_target_met is False
    
    # Thus, convergence fails.
    level_converged = False
    if performance_target_met and stable_windows >= 4 and variance < 0.005:
        level_converged = True
        
    assert level_converged is False
