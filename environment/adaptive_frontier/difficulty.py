from .types import TaskSpec
import random
import hashlib

def get_level_spec(level: int, seed: int, distribution: str = "train") -> TaskSpec:
    # Deterministic generation for seed + level
    task_hash = hashlib.md5(f"{seed}_{level}_{distribution}".encode()).hexdigest()
    rng = random.Random(int(task_hash[:8], 16))
    
    # Base defaults
    grid_size = 8
    obstacle_density = 0.0
    target_distance = 4
    resource_required = False
    gate_required = False
    hazard_density = 0.0
    dynamic_terrain_density = 0.0
    transition_uncertainty = 0.0
    cue_duration = 0
    dependency_delay = 0
    objective_type = "DIRECT_REACH"
    
    if distribution == "ood":
        # OOD distribution
        grid_size = rng.randint(14, 16)
        obstacle_density = rng.uniform(0.30, 0.40)
        target_distance = rng.randint(8, 12)
        resource_required = True
        gate_required = True
        hazard_density = rng.uniform(0.15, 0.20)
        dynamic_terrain_density = rng.uniform(0.20, 0.30)
        transition_uncertainty = rng.uniform(0.20, 0.30)
        cue_duration = rng.randint(1, 3)
        objective_type = "OOD_COMBINED"
    elif distribution == "plasticity_probe":
        # Held-out structural distribution explicitly for plasticity probe testing
        grid_size = rng.randint(9, 11)
        obstacle_density = rng.uniform(0.15, 0.25)
        target_distance = rng.randint(5, 7)
        resource_required = False
        gate_required = False
        hazard_density = 0.0
        dynamic_terrain_density = rng.uniform(0.10, 0.20)
        transition_uncertainty = 0.0
        cue_duration = 0
        objective_type = "PLASTICITY_PROBE"
    else:
        # PROGRESSION
        if level <= 1:
            # LEVEL 1: hard but learnable
            grid_size = rng.randint(8, 9)
            obstacle_density = rng.uniform(0.15, 0.20)
            target_distance = rng.randint(4, 6)
            objective_type = "HARD_REACH"
        elif level == 2:
            # LEVEL 2: increased spatial complexity
            grid_size = rng.randint(9, 10)
            obstacle_density = rng.uniform(0.20, 0.25)
            target_distance = rng.randint(5, 8)
            objective_type = "SPATIAL_COMPLEXITY"
        elif level == 3:
            # LEVEL 3: spatial + temporal pressure
            grid_size = rng.randint(9, 10)
            obstacle_density = rng.uniform(0.20, 0.25)
            target_distance = rng.randint(5, 8)
            resource_required = True
            gate_required = True
            cue_duration = rng.randint(3, 5)
            objective_type = "SPATIAL_TEMPORAL"
        elif level == 4:
            # LEVEL 4: spatial + temporal + dynamic uncertainty
            grid_size = rng.randint(10, 11)
            obstacle_density = rng.uniform(0.20, 0.25)
            target_distance = rng.randint(5, 8)
            resource_required = True
            gate_required = True
            cue_duration = rng.randint(3, 5)
            hazard_density = rng.uniform(0.05, 0.10)
            dynamic_terrain_density = rng.uniform(0.10, 0.15)
            transition_uncertainty = rng.uniform(0.10, 0.15)
            objective_type = "DYNAMIC_UNCERTAINTY"
        elif level == 5:
            # LEVEL 5: compound objective + stronger distribution shift
            grid_size = rng.randint(11, 12)
            obstacle_density = rng.uniform(0.25, 0.30)
            target_distance = rng.randint(6, 9)
            resource_required = True
            gate_required = True
            cue_duration = rng.randint(5, 8)
            hazard_density = rng.uniform(0.10, 0.15)
            dynamic_terrain_density = rng.uniform(0.15, 0.20)
            transition_uncertainty = rng.uniform(0.15, 0.20)
            objective_type = "COMPOUND_SHIFT"
        else:
            # LEVEL 6+: progressively compound difficulty
            grid_size = min(16, 12 + (level - 5) // 2)
            obstacle_density = min(0.40, rng.uniform(0.25, 0.30) + (level - 5) * 0.01)
            target_distance = min(12, rng.randint(6, 9) + (level - 5) // 2)
            resource_required = True
            gate_required = True
            cue_duration = min(10, rng.randint(5, 8) + (level - 5) // 3)
            hazard_density = min(0.25, rng.uniform(0.10, 0.15) + (level - 5) * 0.01)
            dynamic_terrain_density = min(0.35, rng.uniform(0.15, 0.20) + (level - 5) * 0.01)
            transition_uncertainty = min(0.30, rng.uniform(0.15, 0.20) + (level - 5) * 0.01)
            objective_type = "MAX_PRESSURE"
            
    return TaskSpec(
        task_id=f"lvl_{level}_{distribution}",
        requested_seed=seed,
        resolved_seed=seed,
        generation_attempt=0,
        level=level,
        grid_size=grid_size,
        obstacle_density=obstacle_density,
        target_distance=target_distance,
        resource_required=resource_required,
        gate_required=gate_required,
        hazard_density=hazard_density,
        dynamic_terrain_density=dynamic_terrain_density,
        transition_uncertainty=transition_uncertainty,
        cue_duration=cue_duration,
        dependency_delay=dependency_delay,
        distribution_id=distribution,
        objective_type=objective_type
    )
