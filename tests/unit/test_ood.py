import numpy as np
import pytest
from evolution.ood import OODMonitor

def test_ood_monitor_no_shift():
    monitor = OODMonitor(obs_dim=2, window_size=100, threshold=2.0, min_samples=50)
    
    # Reference window
    for _ in range(60):
        obs = np.random.normal(loc=0.0, scale=1.0, size=(2,))
        monitor.add_observation(obs)
        
    monitor.lock_reference()
    
    assert monitor.is_reference_locked is True
    
    # Current window (same distribution)
    for _ in range(60):
        obs = np.random.normal(loc=0.0, scale=1.0, size=(2,))
        monitor.add_observation(obs)
        
    dist = monitor.compute_distance()["mean_distance"]
    assert dist < 2.0
    assert not monitor.is_shift_detected()

def test_ood_monitor_with_shift():
    monitor = OODMonitor(obs_dim=2, window_size=100, threshold=2.0, min_samples=50)
    
    # Reference window
    for _ in range(60):
        obs = np.random.normal(loc=0.0, scale=1.0, size=(2,))
        monitor.add_observation(obs)
        
    monitor.lock_reference()
        
    # Current window (shifted distribution)
    for _ in range(60):
        obs = np.random.normal(loc=5.0, scale=1.0, size=(2,))
        monitor.add_observation(obs)
        
    dist = monitor.compute_distance()["mean_distance"]
    assert dist > 2.0
    assert monitor.is_shift_detected()
    
def test_ood_monitor_insufficient_data():
    monitor = OODMonitor(obs_dim=2, window_size=100, threshold=2.0, min_samples=50)
    
    # Only 10 samples
    for _ in range(10):
        obs = np.random.normal(loc=0.0, scale=1.0, size=(2,))
        monitor.add_observation(obs)
        
    assert not monitor.is_reference_locked
    assert monitor.compute_distance()["mean_distance"] == 0.0
