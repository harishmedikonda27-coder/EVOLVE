import numpy as np
import random
from typing import Tuple, Dict, Any, List
from .objects import ObjectType
from .types import TaskSpec

class LevelGenerationError(Exception):
    pass

class WorldGenerator:
    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.rng = np.random.RandomState(spec.resolved_seed)
        self.grid_size = spec.grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        self.agent_pos = (0, 0)
        self.target_pos = (0, 0)
        self.resource_pos = None
        self.gate_pos = None
        
        self._generate()

    def _generate(self):
        # Place agent and target
        self.agent_pos = (
            self.rng.randint(0, self.grid_size),
            self.rng.randint(0, self.grid_size)
        )
        
        # Place target at approximate distance if possible, otherwise random valid
        self.target_pos = self._place_at_distance(self.agent_pos, self.spec.target_distance)
        self.grid[self.target_pos] = ObjectType.TARGET.value

        if self.spec.resource_required:
            self.resource_pos = self._place_empty()
            self.grid[self.resource_pos] = ObjectType.RESOURCE.value
            
        if self.spec.gate_required:
            self.gate_pos = self._place_empty()
            self.grid[self.gate_pos] = ObjectType.GATE.value

        # Place obstacles
        num_obstacles = int(self.grid_size * self.grid_size * self.spec.obstacle_density)
        for _ in range(num_obstacles):
            pos = self._place_empty()
            if pos:
                self.grid[pos] = ObjectType.OBSTACLE.value

        # Place hazards
        num_hazards = int(self.grid_size * self.grid_size * self.spec.hazard_density)
        for _ in range(num_hazards):
            pos = self._place_empty()
            if pos:
                self.grid[pos] = ObjectType.HAZARD.value
                
        # Place dynamic terrain
        num_dyn = int(self.grid_size * self.grid_size * self.spec.dynamic_terrain_density)
        for _ in range(num_dyn):
            pos = self._place_empty()
            if pos:
                self.grid[pos] = ObjectType.DYNAMIC_TERRAIN.value

        if not self._verify_solvability():
            raise LevelGenerationError("Generated level is not solvable")

    def _place_empty(self) -> Tuple[int, int]:
        empty_cells = np.argwhere(self.grid == ObjectType.EMPTY.value)
        # Exclude agent pos
        empty_cells = [tuple(c) for c in empty_cells if tuple(c) != self.agent_pos]
        if not empty_cells:
            return None
        return empty_cells[self.rng.choice(len(empty_cells))]

    def _place_at_distance(self, start: Tuple[int, int], dist: int) -> Tuple[int, int]:
        valid = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if abs(r - start[0]) + abs(c - start[1]) >= dist and self.grid[r, c] == ObjectType.EMPTY.value:
                    valid.append((r, c))
        if valid:
            return valid[self.rng.choice(len(valid))]
        # Fallback to random empty
        return self._place_empty()

    def _verify_solvability(self) -> bool:
        # Simple BFS
        def bfs(start, goals, allowed):
            q = [start]
            visited = {start}
            while q:
                curr = q.pop(0)
                if curr in goals:
                    return True
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = curr[0]+dr, curr[1]+dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if (nr, nc) not in visited and self.grid[nr, nc] in allowed:
                            visited.add((nr, nc))
                            q.append((nr, nc))
            return False

        allowed_movement = [
            ObjectType.EMPTY.value,
            ObjectType.TARGET.value,
            ObjectType.RESOURCE.value,
            ObjectType.GATE.value,
            ObjectType.HAZARD.value,
            ObjectType.DYNAMIC_TERRAIN.value
        ]

        if self.spec.resource_required and self.spec.gate_required:
            # path to resource -> path to gate -> path to target
            if not bfs(self.agent_pos, [self.resource_pos], allowed_movement):
                return False
            if not bfs(self.resource_pos, [self.gate_pos], allowed_movement):
                return False
            if not bfs(self.gate_pos, [self.target_pos], allowed_movement):
                return False
        elif self.spec.resource_required:
            if not bfs(self.agent_pos, [self.resource_pos], allowed_movement):
                return False
            if not bfs(self.resource_pos, [self.target_pos], allowed_movement):
                return False
        else:
            if not bfs(self.agent_pos, [self.target_pos], allowed_movement):
                return False
        
        return True
