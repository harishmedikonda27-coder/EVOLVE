import numpy as np
from typing import List, Optional, Dict
from collections import deque

class OODMonitor:
    """
    Out-Of-Distribution (OOD) Shift Monitor.
    
    This component explicitly tracks the running distribution of observations over a reference window
    and compares it to a current window via empirical statistical distance (Wasserstein-1 proxy on marginals).
    
    Feature Representation:
    The monitor assumes observations are 1D continuous vectors (e.g., shape: [obs_dim]).
    It tracks the marginal mean and variance for each dimension.
    
    Reference Window:
    A fixed-size deque of earlier observations acting as the baseline distribution.
    
    Current Window:
    A fixed-size deque of recent observations.
    
    Distance Calculation:
    Normalized Euclidean distance between the empirical means of the reference and current windows.
    If the distance exceeds the threshold, an OOD shift is declared.
    """
    
    def __init__(self, obs_dim: int, window_size: int = 1000, threshold: float = 2.0, min_samples: int = 500):
        self.obs_dim = obs_dim
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples
        
        self.reference_window: deque = deque(maxlen=window_size)
        self.current_window: deque = deque(maxlen=window_size)
        
        self.is_reference_locked = False
        
    def lock_reference(self):
        """Locks the reference window. Subsequent observations only go into the current window."""
        if len(self.reference_window) >= self.min_samples:
            self.is_reference_locked = True
        
    def add_observation(self, obs: np.ndarray):
        """
        Add an observation to the monitor.
        If reference is not locked, it goes into the reference window.
        Otherwise, it goes into the current window.
        """
        # Ensure it is a 1D array
        if obs.ndim > 1:
            obs = obs.flatten()
            
        if not self.is_reference_locked:
            self.reference_window.append(obs.copy())
            if len(self.reference_window) >= self.window_size:
                self.lock_reference()
        else:
            self.current_window.append(obs.copy())
            
    def compute_distance(self) -> Dict[str, float]:
        """
        Computes the distance between the reference and current windows.
        Insufficient-data behavior: Returns 0.0 if there aren't enough samples.
        """
        if len(self.reference_window) < self.min_samples or len(self.current_window) < self.min_samples:
            return {
                "mean_distance": 0.0,
                "raw_distance": 0.0,
                "ref_std_mean": 0.0
            }
            
        ref_data = np.stack(self.reference_window)
        curr_data = np.stack(self.current_window)
        
        ref_mean = np.mean(ref_data, axis=0)
        curr_mean = np.mean(curr_data, axis=0)
        
        # Bound the standard deviation to a noise floor to prevent exploding values
        ref_std = np.maximum(np.std(ref_data, axis=0), 1e-2)
        
        # Normalized distance (Z-score like distance on means)
        # Sum of absolute differences normalized by reference standard deviation
        raw_diff = np.abs(ref_mean - curr_mean)
        distances = raw_diff / ref_std
        mean_distance = float(np.mean(distances))
        
        return {
            "mean_distance": mean_distance,
            "raw_distance": float(np.mean(raw_diff)),
            "ref_std_mean": float(np.mean(ref_std))
        }
        
    def is_shift_detected(self) -> bool:
        """Returns True if the computed distance exceeds the threshold."""
        res = self.compute_distance()
        return res["mean_distance"] > self.threshold
        
    def get_metrics(self) -> Dict[str, float]:
        res = self.compute_distance()
        dist = res["mean_distance"]
        return {
            "ood_distance": dist,
            "raw_distance": res["raw_distance"],
            "normalization_factor": res["ref_std_mean"],
            "ood_threshold": self.threshold,
            "shift_detected": float(dist > self.threshold),
            "ref_samples": float(len(self.reference_window)),
            "curr_samples": float(len(self.current_window))
        }

    def reset_current_window(self):
        """Clears the current window. Useful after adapting to a shift."""
        self.current_window.clear()
        
    def reset_all(self):
        """Completely resets both reference and current windows."""
        self.reference_window.clear()
        self.current_window.clear()
        self.is_reference_locked = False
