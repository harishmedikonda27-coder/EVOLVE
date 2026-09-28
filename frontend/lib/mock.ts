import { TopologySnapshot, Signals, Decision, EventData } from '../types';

export const mockTopology: TopologySnapshot = {
  active_module_count: 2,
  total_module_count: 2,
  total_parameter_count: 8312,
  active_ids: ['module_A', 'module_B'],
  nodes: [
    { id: 'encoder_0', kind: 'Linear', role: 'encoder', layer_index: 1, status: 'active', input_dim: 16, output_dim: 32 },
    { id: 'router_0', kind: 'Linear', role: 'router', layer_index: 2, status: 'active', input_dim: 32, output_dim: 2 },
    { id: 'module_A_0', kind: 'Linear', role: 'module_A', layer_index: 3, status: 'active', parameter_count: 2112, utilization: 0.68, created_step: 0 },
    { id: 'module_B_0', kind: 'Linear', role: 'module_B', layer_index: 4, status: 'active', parameter_count: 2112, utilization: 0.32, created_step: 1200 },
    { id: 'head_0', kind: 'Linear', role: 'head', layer_index: 5, status: 'active', input_dim: 64, output_dim: 4 },
  ],
  edges: [
    { source: 'router', target: 'module_A' },
    { source: 'module_A', target: 'head' },
    { source: 'router', target: 'module_B' },
    { source: 'module_B', target: 'head' },
  ],
};

export const mockSignals: Signals = {
  performance_gap: 0.78,
  stagnation: 0.65,
  novelty: 0.72,
  interference: 0.45,
  utilization: 0.32,
  cost: 0.18,
};

export const mockDecision: Decision = {
  action: 'GROW',
  need_score: 0.71,
  signals: mockSignals,
  reason: 'Need score (0.71) exceeds GROW threshold (0.65). Evidence supports structural growth.',
  budget_remaining: 2,
  cooldown_active: false,
};

export const mockEvents: EventData[] = [
  {
    event_id: 'evt_001',
    event_type: 'TRAINING_STARTED',
    run_id: 'run_mock_001',
    step: 0,
    phase: 'A',
    timestamp: '2026-09-16T19:55:00Z',
    payload: { task: 'Environment A', seed: 42 },
  },
  {
    event_id: 'evt_002',
    event_type: 'PERFORMANCE_DROP',
    run_id: 'run_mock_001',
    step: 2000,
    phase: 'B',
    timestamp: '2026-09-16T19:56:30Z',
    payload: { task: 'Environment B', drop: 0.75 },
  },
  {
    event_id: 'evt_003',
    event_type: 'DECISION_MADE',
    run_id: 'run_mock_001',
    step: 2100,
    phase: 'B',
    timestamp: '2026-09-16T19:57:00Z',
    payload: { decision: 'GROW', need_score: 0.71 },
  },
  {
    event_id: 'evt_004',
    event_type: 'MUTATION_ACCEPTED',
    run_id: 'run_mock_001',
    step: 2200,
    phase: 'B',
    timestamp: '2026-09-16T19:58:00Z',
    payload: { mutation_id: 'mut_01', added_module: 'module_B', new_params: 8312 },
  },
];
