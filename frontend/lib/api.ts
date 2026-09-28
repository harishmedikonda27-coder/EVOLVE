const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface RunCreateRequest {
  mode: string;
  seed: number;
  experiment_type: string;
  task_sequence: string[];
}

export interface RunResponse {
  run_id: string;
  state: string;
  mode: string;
  seed: number;
  experiment_type: string;
  created_at: string;
  results?: ExperimentResults;
}

export interface ExperimentResults {
  experiment: string;
  a_before: number;
  b_after: number;
  a_after: number;
  a_success_before: number;
  b_success_after: number;
  a_success_after: number;
  retention_ratio: number;
  parameter_count: number;
  active_modules?: number;
  mutations?: MutationEntry[];
}

export interface MutationEntry {
  module: string;
  step: number;
  need_score: number;
  accepted: boolean;
  reason: string;
  b_score_after: number;
  a_retention: number;
}

export async function healthCheck(): Promise<{ status: string; service: string }> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function createRun(req: RunCreateRequest): Promise<RunResponse> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function getRun(runId: string): Promise<RunResponse> {
  const res = await fetch(`${API_BASE}/runs/${runId}`);
  return res.json();
}

export async function startRun(runId: string): Promise<{ run_id: string; state: string }> {
  const res = await fetch(`${API_BASE}/runs/${runId}/start`, { method: 'POST' });
  return res.json();
}

export async function getRunResults(runId: string): Promise<ExperimentResults> {
  const res = await fetch(`${API_BASE}/runs/${runId}/results`);
  return res.json();
}

export function createWebSocket(runId: string): WebSocket {
  const wsBase = API_BASE.replace('http', 'ws');
  return new WebSocket(`${wsBase}/runs/${runId}/stream`);
}
