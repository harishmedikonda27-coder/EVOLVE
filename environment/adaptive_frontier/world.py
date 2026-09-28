import numpy as np
import uuid
import copy
from typing import Dict, Any, Tuple, Optional
from .types import TaskSpec
from .objects import ObjectType
from .generator import WorldGenerator, LevelGenerationError
from environment.base import BaseEnvironment, TaskGenerationFailure

class AdaptiveFrontierWorld(BaseEnvironment):
    def __init__(self, spec: TaskSpec):
        super().__init__(obs_dim=16, action_dim=5, max_steps=200)
        self.spec = spec
        self.rng = np.random.RandomState(spec.resolved_seed)
        
        # Retry generation a few times if invalid
        for i in range(10):
            try:
                gen_spec = copy.deepcopy(spec)
                gen_spec.resolved_seed = spec.requested_seed + i
                gen_spec.generation_attempt = i
                self.generator = WorldGenerator(gen_spec)
                self.spec = gen_spec  # store the definitively resolved spec
                break
            except LevelGenerationError:
                if i == 9:
                    print(f"[TASK] L{spec.level:02d} instance=? requested_seed={spec.requested_seed} UNSOLVABLE after 10 attempts")
                    raise TaskGenerationFailure(spec.level, spec.requested_seed, 10, spec.__dict__)
                else:
                    print(f"[TASK] L{spec.level:02d} instance=? requested_seed={spec.requested_seed} generation_attempt={i} seed={spec.requested_seed+i} UNSOLVABLE")
        print(f"[TASK] L{gen_spec.level:02d} instance=? requested_seed={gen_spec.requested_seed} generation_attempt={i} seed={gen_spec.resolved_seed} SOLVABLE")

        self.grid = self.generator.grid.copy()
        self.agent_pos = self.generator.agent_pos
        self.grid_size = spec.grid_size
        
        self.has_resource = False
        self.gate_unlocked = False
        self.step_count = 0
        
        self.cue_active = True if spec.cue_duration > 0 else False
        
        self.target_reached = False
        self.is_done = False
        self.last_reward = 0.0
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        self.grid = self.generator.grid.copy()
        self.agent_pos = self.generator.agent_pos
        self.has_resource = False
        self.gate_unlocked = False
        self.step_count = 0
        self.cue_active = True if self.spec.cue_duration > 0 else False
        self.target_reached = False
        self.is_done = False
        self.last_reward = 0.0
        
        return self._get_obs(), self.get_state()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = -0.01  # step cost
        
        if self.step_count > self.spec.cue_duration:
            self.cue_active = False

        # ACTIONS: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
        dr, dc = 0, 0
        if action == 0: dr = -1
        elif action == 1: dr = 1
        elif action == 2: dc = -1
        elif action == 3: dc = 1
        
        new_r, new_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        
        # Dynamic terrain transition uncertainty
        if action != 4 and 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
            if self.grid[new_r, new_c] == ObjectType.DYNAMIC_TERRAIN.value:
                if self.rng.random() < self.spec.transition_uncertainty:
                    # Random movement instead
                    dirs = [(-1,0), (1,0), (0,-1), (0,1), (0,0)]
                    dr, dc = dirs[self.rng.choice(len(dirs))]
                    new_r, new_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        moved = False
        if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
            cell_type = self.grid[new_r, new_c]
            
            if cell_type == ObjectType.OBSTACLE.value:
                reward -= 0.05
            elif cell_type == ObjectType.GATE.value and self.spec.gate_required and not self.has_resource:
                reward -= 0.05 # locked gate
            else:
                self.agent_pos = (new_r, new_c)
                moved = True
                
                if cell_type == ObjectType.HAZARD.value:
                    reward -= 0.1
                elif cell_type == ObjectType.RESOURCE.value:
                    if not self.has_resource:
                        self.has_resource = True
                        reward += 0.5
                        # Remove resource
                        self.grid[new_r, new_c] = ObjectType.EMPTY.value
                elif cell_type == ObjectType.TARGET.value:
                    if (not self.spec.gate_required) or (self.spec.gate_required and self.has_resource):
                        self.target_reached = True
                        reward += 1.0
                        self.is_done = True
        else:
            if action != 4:
                reward -= 0.05 # boundary bump

        if self.step_count >= 200: # Max steps
            self.is_done = True

        self.last_reward = reward
        return self._get_obs(), reward, self.is_done, False, {"success": self.target_reached}
        
    def _get_obs(self) -> np.ndarray:
        # Construct 16-dim observation to maintain compatibility
        obs = np.zeros(16, dtype=np.float32)
        
        # 0,1: Normalized agent pos
        obs[0] = self.agent_pos[0] / self.grid_size
        obs[1] = self.agent_pos[1] / self.grid_size
        
        # 2,3: Normalized target pos
        target = self.generator.target_pos
        obs[2] = (target[0] - self.agent_pos[0]) / self.grid_size
        obs[3] = (target[1] - self.agent_pos[1]) / self.grid_size
        
        # 4,5: Normalized resource pos (if cue is active or no cue req)
        if self.spec.resource_required:
            if self.cue_active or self.spec.cue_duration == 0:
                res = self.generator.resource_pos
                obs[4] = (res[0] - self.agent_pos[0]) / self.grid_size
                obs[5] = (res[1] - self.agent_pos[1]) / self.grid_size
                
        # 6,7: Normalized gate pos
        if self.spec.gate_required:
            gate = self.generator.gate_pos
            obs[6] = (gate[0] - self.agent_pos[0]) / self.grid_size
            obs[7] = (gate[1] - self.agent_pos[1]) / self.grid_size
            
        # 8: Has resource
        obs[8] = 1.0 if self.has_resource else 0.0
        
        # 9: Cue active
        obs[9] = 1.0 if self.cue_active else 0.0
        
        # 10: Uncertainty
        obs[10] = self.spec.transition_uncertainty
        
        # 11: Hazard nearby
        hazard_near = 0.0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = self.agent_pos[0]+dr, self.agent_pos[1]+dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if self.grid[nr, nc] == ObjectType.HAZARD.value:
                    hazard_near = 1.0
        obs[11] = hazard_near
        
        # 12-15: Obstacles/Walls around (up, down, left, right)
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]
        for i, (dr, dc) in enumerate(dirs):
            nr, nc = self.agent_pos[0]+dr, self.agent_pos[1]+dc
            if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
                obs[12+i] = 1.0
            elif self.grid[nr, nc] == ObjectType.OBSTACLE.value:
                obs[12+i] = 1.0
                
        return obs

    def get_state(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.tolist(),
            "agent_pos": list(self.agent_pos),
            "target_pos": list(self.generator.target_pos),
            "grid_size": self.grid_size,
            "has_resource": self.has_resource,
            "cue_active": self.cue_active,
            "uncertainty": self.spec.transition_uncertainty,
            "step_count": self.step_count,
            "spec": {
                "level": self.spec.level,
                "objective": self.spec.objective_type
            }
        }

    def get_task_profile(self) -> Dict[str, Any]:
        return {
            "name": f"AdaptiveFrontier_L{self.spec.level}",
            "grid_size": (self.grid_size, self.grid_size),
            "target_count": 1,
            "max_steps": 200,
            "reward_spec": {
                "R_min": -2.0,
                "R_max": 1.0,
            },
            "observation_shape": (16,),
            "action_space": 5, # Up, Down, Left, Right, Stay
            "dynamics": "deterministic" if self.spec.transition_uncertainty == 0 else "stochastic",
            "memory_required": self.spec.cue_duration > 0
        }

    def get_fingerprint(self) -> str:
        import hashlib
        grid_bytes = self.generator.grid.tobytes()
        features = f"{self.spec.grid_size}_{self.spec.obstacle_density}_{self.spec.objective_type}_{self.spec.level}_{self.spec.requested_seed}"
        return hashlib.md5(grid_bytes + features.encode()).hexdigest()
