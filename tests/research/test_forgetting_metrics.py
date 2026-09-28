import pytest
from backend.tasks import TaskRegistry

def test_forgetting_explicit_metric():
    registry = TaskRegistry()
    registry.register_task("task1", "Navigation", {}, 42, 1)
    
    # A[i] = [0.50, 0.70, 0.65, 0.60]
    registry.update_performance("task1", 0.50, global_step=100)
    registry.update_performance("task1", 0.70, global_step=200)
    registry.update_performance("task1", 0.65, global_step=300)
    registry.update_performance("task1", 0.60, global_step=400)
    
    # max = 0.70, final = 0.60
    forgetting = registry.calculate_forgetting("task1")
    assert pytest.approx(forgetting, 0.001) == 0.10

def test_retention():
    registry = TaskRegistry()
    registry.register_task("task1", "Navigation", {}, 42, 1)
    registry.update_performance("task1", 0.8, global_step=100)
    
    registry.register_task("task2", "Navigation", {}, 43, 2)
    registry.update_performance("task2", 0.6, global_step=200)
    
    # Retention excludes the current task (task2), only looks at past tasks (task1)
    retention = registry.calculate_retention(current_step=300)
    assert pytest.approx(retention, 0.001) == 0.8
    
def test_forward_learning():
    registry = TaskRegistry()
    # FL = Continual - Fresh
    fl = registry.calculate_forward_learning(0.80, 0.65)
    assert pytest.approx(fl, 0.001) == 0.15
