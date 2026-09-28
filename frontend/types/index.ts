export type RunState = 'CREATED' | 'INITIALIZING' | 'RUNNING' | 'PAUSED' | 'STOPPING' | 'STOPPED' | 'COMPLETED' | 'FAILED';

export interface Signals {
  performance_gap: number;
  stagnation: number;
  novelty: number;
  interference: number;
  utilization: number;
  cost: number;
}

export interface AdaptationState {
  current_performance: number;
  stagnation_slope: number;
  novelty: number;
  ood_shift: number;
  interference: number;
  plasticity_auc: number;
  utilization_mean: number;
  utilization_variance: number;
  module_count: number;
  parameter_count: number;
  budget_remaining: number;
  youngest_module_age: number;
}

export interface Decision {
  action: 'REUSE' | 'ADAPT' | 'GROW' | 'PRUNE' | 'REWIRE';
  need_score: number;
  signals: Signals;
  reason: string;
  budget_remaining: number;
  cooldown_active: boolean;
  q_values?: Record<string, number>;
  adaptation_state?: AdaptationState;
}

export interface ConsequenceResolution {
  intervention_id: string;
  action: string;
  decision_step: number;
  task_id_at_decision: string;
  cumulative_meta_reward: number;
}

export interface ActualLayer {
  id: string;
  type: string;
  input_dim?: number;
  output_dim?: number;
  parameters?: number;
  sparse?: {
    enabled: boolean;
    active_connections: number;
    possible_connections: number;
    density: number;
    mask_id?: string;
  };
}

export interface ActualConnection {
  id: string;
  layer_id: string;
  source_index: number;
  target_index: number;
  active: boolean;
  mask_id: string;
}

export interface NodeInfo {
  id: string;
  kind: string; // 'Linear', 'ReLU', 'input', 'output'
  role: string; // 'encoder', 'router', 'module_A', 'head', 'input', 'output'
  layer_index: number;
  status: string;
  input_dim?: number;
  output_dim?: number;
  parameter_count?: number;
  weight_b64?: string;
  created_step?: number;
  utilization?: number;
  sparse?: {
    enabled: boolean;
    active_connections: number;
    possible_connections: number;
    density: number;
    mask_id?: string;
    mask_b64?: string;
  };
  module_id?: string;
}

export interface EdgeInfo {
  source: string;
  target: string;
  density?: number;
  active_connections?: number;
  possible_connections?: number;
  source_type?: string;
  mask?: string;
}

export interface TopologySnapshot {
  active_module_count: number;
  total_module_count: number;
  total_parameter_count: number;
  actual_layer_count?: number;
  nodes: NodeInfo[];
  edges: EdgeInfo[];
  active_ids: string[];
  topology_revision?: number;
  architecture_revision?: number;
  actual_connections?: ActualConnection[];
}

export interface EventData {
  event_id: string;
  event_type: string;
  run_id: string;
  step: number;
  phase: string;
  timestamp: string;
  payload: Record<string, any>;
}

export interface EnvironmentDefinition {
  id: string;
  name: string;
  description: string;
  grid_size: [number, number];
  num_obstacles: number | null;
  hazards_enabled: boolean | null;
  moving_obstacles: boolean | null;
  target_count: number;
}

export interface MovementTransition {
  step: number;
  previous_position: [number, number];
  position: [number, number];
  action: string;
  blocked: boolean;
}

export interface BaseEnvState {
  grid_size: [number, number];
  agent_pos: [number, number];
  obstacles: [number, number][];
  hazards: [number, number][];
  movements?: MovementTransition[];
}

export type EnvironmentState = Record<string, any>;

export interface LevelHistory {
  level_index: number;
  level_id: string;
  event_count: number;
  initial_time: string;
  final_time: string;
  status: 'CONVERGED' | 'FAILED' | 'CURRENT' | 'PENDING';
}

export interface HistoryEntry {
  history_id: string;
  run_id: string;
  level_id: string;
  level_index: number;
  event_id: string;
  event_type: string;
  mutation_type: string;
  mutation_status: string;
  label: string;
  timestamp: string;
  topology_revision: number;
  performance: number;
  parameter_count: number;
  active_module_count: number;
  edge_count: number;
  topology_snapshot?: TopologySnapshot;
}
