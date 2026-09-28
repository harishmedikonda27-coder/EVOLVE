import pytest
import numpy as np

def test_one_task_per_level():
    from backend.experiments.run import create_environment
    from brain.brain import ModularBrain
    from evolution.mutation import MutationManager
    
    # 1. Level starts
    level = 1
    level_base_seed = 42 + level * 10000
    env = create_environment("AdaptiveFrontier", level, level_base_seed)
    
    task_id = f"AdaptiveFrontier_L{level:02d}"
    requested_seed = level_base_seed
    resolved_seed = env.spec.resolved_seed
    
    brain = ModularBrain(16, 5)
    brain.add_module("mod1", 0, "TEST")
    mut_manager = MutationManager()
    
    # Capture fingerprint
    def get_fingerprint(env):
        return hash(env.grid.tobytes())
        
    initial_fingerprint = get_fingerprint(env)
    
    # 2. Run many training steps (simulate term/trunc)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(np.random.randint(0, 5))
        if term or trunc:
            env.reset()
            # ASSERT identity remains constant after reset
            assert env.spec.resolved_seed == resolved_seed
            assert get_fingerprint(env) == initial_fingerprint
            
    # 3. Simulate Evaluation on same task
    eval_env = create_environment("AdaptiveFrontier", level, level_base_seed)
    assert eval_env.spec.resolved_seed == resolved_seed
    assert get_fingerprint(eval_env) == initial_fingerprint
    
    # 4. Simulate Mutation
    from evolution.rollback import RollbackManager
    import torch.optim as optim
    optimizer = optim.Adam(brain.parameters(), lr=0.001)
    
    rollback = RollbackManager()
    ckpt = rollback.capture_checkpoint(brain, optimizer, {})
    
    mut_manager.execute_grow(brain, 100)
    
    # Brain mutated, but environment remains exactly the same
    env.reset()
    assert env.spec.resolved_seed == resolved_seed
    assert get_fingerprint(env) == initial_fingerprint
    
    # 5. Simulate Rollback
    brain, optimizer = rollback.restore_checkpoint(brain, optimizer, ckpt)
    env.reset()
    assert env.spec.resolved_seed == resolved_seed
    assert get_fingerprint(env) == initial_fingerprint
