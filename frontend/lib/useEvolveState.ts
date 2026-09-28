import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  TopologySnapshot, 
  EnvironmentState, 
  Decision,
  EventData,
  LevelHistory,
  HistoryEntry
} from '../types';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';
export type AppState = 'INITIAL' | 'SELECTING_ENV' | 'WORLD_READY' | 'INITIALIZING' | 'CONNECTING' | 'ACTIVE' | 'SNAPSHOT';

export interface EvolveMetrics {
  globalStep: number;
  levelStep: number;
  reward: number;
  rewardRaw?: number;
  performance: number;
  family: string;
  level: number;
  taskId: string;
  retention?: number | null;
  forgetting?: number | null;
  ood_shift?: number | null;
  ood_recovery?: number | null;
  plasticity_status?: string | null;
}

export interface MutationState {
  status: 'IDLE' | 'PROPOSED' | 'ACCEPTED' | 'REJECTED' | 'ROLLBACK';
  action: string | null;
  reason?: string;
  gain?: number;
  timestamp: number;
  record?: any;
}

export interface CompletedLevel {
  taskId: string;
  family: string;
  level: number;
  converged: boolean;
  steps: number;
  snapshotId: string;
  finalPerformance: number;
  topology: TopologySnapshot;
}

export function useEvolveState(apiBase: string) {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [runId, setRunId] = useState<string | null>(null);
  const [backendRunState, setBackendRunState] = useState<string>('IDLE');
  
  // Live State
  const [topology, setTopology] = useState<TopologySnapshot | null>(null);
  const [envState, setEnvState] = useState<EnvironmentState | null>(null);
  const [metrics, setMetrics] = useState<EvolveMetrics>({
    globalStep: 0, levelStep: 0, reward: 0, rewardRaw: 0, performance: 0, family: '', level: 0, taskId: ''
  });
  const [decision, setDecision] = useState<Decision | null>(null);
  const [mutationState, setMutationState] = useState<MutationState>({ status: 'IDLE', action: null, timestamp: 0 });
  
  // Phase 2 State
  const [oodShift, setOodShift] = useState<number>(0);
  const [plasticity, setPlasticity] = useState<any>(null);
  const [metaLoss, setMetaLoss] = useState<number | null>(null);
  const [consequences, setConsequences] = useState<any[]>([]);

  // History
  const [completedLevels, setCompletedLevels] = useState<CompletedLevel[]>([]);
  const [events, setEvents] = useState<EventData[]>([]);

  // Phase 3: Structural History Explorer
  const [levelsHistory, setLevelsHistory] = useState<LevelHistory[]>([]);
  const [selectedLevelId, setSelectedLevelId] = useState<string | null>(null);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [selectedHistoricalTopology, setSelectedHistoricalTopology] = useState<TopologySnapshot | null>(null);
  const [historyEntries, setHistoryEntries] = useState<HistoryEntry[]>([]);
  const [isHistoricalView, setIsHistoricalView] = useState<boolean>(false);
  const [historyLoading, setHistoryLoading] = useState<boolean>(false);
  
  const wsRef = useRef<WebSocket | null>(null);

  const fetchLevelsHistory = useCallback(async (currentRunId: string) => {
    try {
      const res = await fetch(`${apiBase}/runs/${currentRunId}/history`);
      if (res.ok) {
        const data = await res.json();
        setLevelsHistory(data.levels || []);
      }
    } catch (e) {
      console.error('Failed to fetch levels history', e);
    }
  }, [apiBase]);

  const fetchLevelHistory = useCallback(async (levelId: string) => {
    if (!runId) return;
    setHistoryLoading(true);
    try {
      const res = await fetch(`${apiBase}/runs/${runId}/history/${levelId}`);
      if (res.ok) {
        const data = await res.json();
        setHistoryEntries(data.entries || []);
      }
    } catch (e) {
      console.error('Failed to fetch level history', e);
    } finally {
      setHistoryLoading(false);
    }
  }, [runId, apiBase]);

  const selectHistoryEntry = useCallback(async (historyId: string) => {
    if (!runId) return;
    setHistoryLoading(true);
    try {
      const res = await fetch(`${apiBase}/runs/${runId}/history/entry/${historyId}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedHistoricalTopology(data.topology_snapshot);
        setSelectedHistoryId(historyId);
        setIsHistoricalView(true);
      }
    } catch (e) {
      console.error('Failed to fetch history entry snapshot', e);
    } finally {
      setHistoryLoading(false);
    }
  }, [runId, apiBase]);

  const returnToLive = useCallback(() => {
    setSelectedHistoricalTopology(null);
    setSelectedHistoryId(null);
    setIsHistoricalView(false);
  }, []);

  const selectLevel = useCallback((levelId: string) => {
    setSelectedLevelId(levelId);
    setSelectedHistoryId(null);
    setSelectedHistoricalTopology(null);
    setIsHistoricalView(false);
    fetchLevelHistory(levelId);
  }, [fetchLevelHistory]);

  const connectToRun = useCallback((newRunId: string) => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    setRunId(newRunId);
    setBackendRunState('CREATED');
    setConnectionStatus('connecting');
    
    const wsUrl = apiBase.replace('http', 'ws');
    const ws = new WebSocket(`${wsUrl}/runs/${newRunId}/stream`);
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      fetchLevelsHistory(newRunId);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const type = data.event_type;
        const payload = data.payload;
        const phase = data.phase; // usually family name or 'INITIAL'/'COMPLETE'
        
        setEvents(prev => [...prev.slice(-49), data]); 
        
        switch (type) {
          case 'TOPOLOGY_SNAPSHOT':
            setTopology(payload);
            break;
            
          case 'LEVEL_HISTORY_UPDATED':
            // The backend saved a new history entry. Refresh levels history.
            fetchLevelsHistory(newRunId);
            // If the user is currently viewing the structural history of the level that just updated, refresh it too.
            setSelectedLevelId(prevLevelId => {
               if (prevLevelId === payload.level_id) {
                   fetchLevelHistory(payload.level_id);
               }
               return prevLevelId;
            });
            break;
            
          case 'ENVIRONMENT_STATE_UPDATE':
            setEnvState(payload);
            if (payload.training_step !== undefined) {
              setMetrics(m => ({
                ...m,
                globalStep: payload.global_step ?? m.globalStep,
                levelStep: payload.training_step,
              }));
            }
            break;
            
          case 'METRICS_UPDATE':
            setMetrics(m => ({
              ...m,
              globalStep: payload.step,
              levelStep: payload.training_step || 0,
              reward: payload.reward_raw || payload.reward || 0,
              rewardRaw: payload.reward_raw !== undefined ? payload.reward_raw : payload.reward,
              performance: payload.performance || 0,
              retention: payload.retention,
              forgetting: payload.forgetting,
              ood_shift: payload.ood_shift,
              ood_recovery: payload.ood_recovery,
              plasticity_status: payload.plasticity_status
            }));
            break;
            
          case 'LEVEL_STARTED':
            setMetrics(m => ({
              ...m,
              family: payload.family,
              level: payload.level,
              taskId: payload.task_id,
              levelStep: 0
            }));
            setEnvState(null); 
            break;
            
          case 'DECISION_MADE':
            setDecision(payload);
            break;
            
          case 'MUTATION_PROPOSED':
          case 'MUTATION_VERIFYING':
            setMutationState({ status: 'PROPOSED', action: payload.action, timestamp: Date.now(), record: payload.record });
            break;
            
          case 'MUTATION_ACCEPTED':
            setMutationState({ status: 'ACCEPTED', action: payload.action, gain: payload.gain, timestamp: Date.now(), record: payload.record });
            break;
            
          case 'MUTATION_REJECTED':
            setMutationState({ status: 'REJECTED', action: payload.action, reason: payload.reason, timestamp: Date.now(), record: payload.record });
            break;
            
          case 'ROLLBACK':
            setMutationState({ status: 'ROLLBACK', action: payload.action, timestamp: Date.now(), record: payload.record });
            break;
            
          case 'OOD_SHIFT_DETECTED':
            setOodShift(payload.shift_distance);
            break;
            
          case 'PLASTICITY_PROBE':
            setPlasticity(payload);
            break;
            
          case 'CONSEQUENCE_RESOLVED':
            setConsequences(prev => [payload, ...prev].slice(0, 50));
            break;
            
          case 'META_TRAIN_UPDATE':
            setMetaLoss(payload.loss);
            break;

          case 'LEVEL_COMPLETED':
            setCompletedLevels(prev => {
              if (prev.find(l => l.taskId === payload.task_id)) return prev;
              
              // Extract level from task_id if possible (e.g. HierarchicalNavigation_L0)
              let parsedLevel = 0;
              const match = payload.task_id.match(/_L(\d+)$/);
              if (match) parsedLevel = parseInt(match[1]);

              return [...prev, {
                taskId: payload.task_id,
                family: phase,
                level: parsedLevel, 
                converged: payload.converged,
                steps: payload.steps,
                snapshotId: payload.snapshot_id,
                finalPerformance: payload.final_performance,
                topology: payload.topology
              }];
            });
            break;
            
          case 'RUN_STARTED':
            setBackendRunState('RUNNING');
            break;

          case 'RUN_COMPLETED':
            setBackendRunState('COMPLETED');
            break;
            
          case 'RUN_STOPPING':
            setBackendRunState('STOPPING');
            break;
            
          case 'RUN_STOPPED':
            setBackendRunState('STOPPED');
            break;
            
          case 'EXPERIMENT_FAILED':
            console.error("Experiment failed:", payload);
            setBackendRunState('FAILED');
            break;
            
          case 'ERROR':
            console.error("Backend Error Event:", payload);
            setBackendRunState('FAILED');
            break;
        }
      } catch (err) {
        console.error("WebSocket parsing error:", err);
      }
    };
    
    ws.onclose = () => {
      setConnectionStatus('disconnected');
    };
    
    ws.onerror = () => {
      setConnectionStatus('error');
    };
    
    wsRef.current = ws;
  }, [apiBase]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setRunId(null);
    setBackendRunState('IDLE');
    setConnectionStatus('disconnected');
  }, []);

  const startRun = useCallback(async (id?: string) => {
    const targetId = id || runId;
    if (!targetId) return;
    try {
      setBackendRunState('STARTING');
      await fetch(`${apiBase}/runs/${targetId}/start`, { method: 'POST' });
    } catch (e) {
      console.error("Failed to start run", e);
      setBackendRunState('FAILED');
    }
  }, [apiBase, runId]);

  const stopRun = useCallback(async () => {
    if (!runId) return;
    try {
      setBackendRunState('STOPPING');
      await fetch(`${apiBase}/runs/${runId}/stop`, { method: 'POST' });
    } catch (e) {
      console.error("Failed to stop run", e);
    }
  }, [apiBase, runId]);

  return {
    connectionStatus,
    runId,
    backendRunState,
    topology,
    envState,
    metrics,
    decision,
    mutationState,
    completedLevels,
    events,
    oodShift,
    plasticity,
    metaLoss,
    consequences,
    connectToRun,
    disconnect,
    startRun,
    stopRun,
    setEnvState,
    setMetrics,
    levelsHistory,
    selectedLevelId,
    selectedHistoryId,
    selectedHistoricalTopology,
    historyEntries,
    isHistoricalView,
    historyLoading,
    selectLevel,
    selectHistoryEntry,
    returnToLive
  };
}
