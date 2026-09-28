from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class TaskSpec:
    task_id: str
    requested_seed: int
    resolved_seed: int
    generation_attempt: int
    level: int
    grid_size: int
    obstacle_density: float
    target_distance: int
    resource_required: bool
    gate_required: bool
    hazard_density: float
    dynamic_terrain_density: float
    transition_uncertainty: float
    cue_duration: int
    dependency_delay: int
    distribution_id: str
    objective_type: str

@dataclass
class OODConfig:
    distribution_id: str
    grid_size_range: tuple
    obstacle_density_range: tuple
    cue_duration_range: tuple
    uncertainty_range: tuple
    hazard_density_range: tuple
    
@dataclass
class LevelConfig:
    level_id: str
    task_variant: str
    difficulty_level: int
    seed: int
    ood_shift: bool = False
    shift_magnitude: float = 0.0
