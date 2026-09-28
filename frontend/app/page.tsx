'use client';
import React, { useState, useEffect } from 'react';
import { EnvironmentView } from '../components/EnvironmentView';
import { BrainTopology } from '../components/BrainTopology';
import { ActualArchitectureView } from '../components/ActualArchitectureView';
import { NeuralChangeIndicator } from '../components/NeuralChangeIndicator';
import { TelemetryPanel } from '../components/TelemetryPanel';
import { HistoryExplorer } from '../components/HistoryExplorer';
import { EventTimeline } from '../components/EventTimeline';
import { useEvolveState, CompletedLevel, AppState } from '../lib/useEvolveState';

export default function ResearchDashboard() {
  const [appState, setAppState] = useState<AppState>('INITIAL');
  const [snapshotMode, setSnapshotMode] = useState<CompletedLevel | null>(null);
  const [topologyMode, setTopologyMode] = useState<'ABSTRACT' | 'ACTUAL'>('ABSTRACT');
  const [dragStart, setDragStart] = useState<{x: number, y: number} | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  const {
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
  } = useEvolveState(API_BASE);

  // Fetch initial topology for the hero display
  const [initialTopology, setInitialTopology] = useState(null);
  useEffect(() => {
    fetch(`${API_BASE}/brain/topology`)
      .then(res => res.json())
      .then(data => setInitialTopology(data))
      .catch(() => {});
  }, [API_BASE]);

  type UIState = 'INITIAL' | 'SELECTING_ENV' | 'WORLD_READY' | 'INITIALIZING' | 'CONNECTING' | 'READY' | 'STARTING' | 'TRAINING' | 'STOPPING' | 'STOPPED' | 'COMPLETED' | 'ERROR';

  // Derive UI state from backend run state + app state
  let uiState: UIState = 'INITIAL';
  if (appState === 'SELECTING_ENV') uiState = 'SELECTING_ENV';
  else if (appState === 'WORLD_READY') uiState = 'WORLD_READY';
  else if (appState === 'INITIALIZING') uiState = 'INITIALIZING';
  else if (appState === 'CONNECTING') uiState = 'CONNECTING';
  else if (backendRunState === 'CREATED') uiState = 'READY';
  else if (backendRunState === 'STARTING') uiState = 'STARTING';
  else if (backendRunState === 'RUNNING') uiState = 'TRAINING';
  else if (backendRunState === 'STOPPING') uiState = 'STOPPING';
  else if (backendRunState === 'STOPPED') uiState = 'STOPPED';
  else if (backendRunState === 'COMPLETED') uiState = 'COMPLETED';
  else if (backendRunState === 'FAILED') uiState = 'ERROR';

  const isActiveRun = uiState !== 'INITIAL' && uiState !== 'SELECTING_ENV';

  const handleEnterWorld = async () => {
    try {
      setAppState('WORLD_READY');
      const res = await fetch(`${API_BASE}/world/preview`);
      if (res.ok) {
        const previewState = await res.json();
        setEnvState(previewState);
        setMetrics(m => ({ ...m, level: previewState.spec.level || 1, family: 'AdaptiveFrontier' }));
      }
    } catch (e) {
      console.error("Failed to load world preview", e);
    }
  };

  const handleStartLearning = async () => {
    try {
      setAppState('INITIALIZING');
      const createRes = await fetch(`${API_BASE}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'research',
          seed: 42,
          experiment_type: 'evolve',
          levels: 30,
        }),
      });
      const runData = await createRes.json();
      setAppState('CONNECTING');
      connectToRun(runData.run_id);
    } catch (e) {
      console.error("Failed to start learning", e);
      setAppState('INITIAL');
      disconnect();
    }
  };

  // Wait for WS connected then start
  useEffect(() => {
    if (appState === 'CONNECTING' && connectionStatus === 'connected' && runId) {
      setAppState('ACTIVE');
      startRun(runId);
    }
  }, [appState, connectionStatus, runId, startRun]);

  const handleSelectSnapshot = (level: CompletedLevel) => setSnapshotMode(level);
  const handleReturnToLive = () => {
    setSnapshotMode(null);
    returnToLive();
  };

  const isLive = connectionStatus === 'connected';

  const displayEvents = uiState === 'WORLD_READY' ? [
    { event_id: 'world-ready', run_id: 'preview', event_type: 'WORLD_READY', timestamp: new Date().toISOString(), payload: {}, step: 0, phase: 'INITIAL' }
  ] : events;

  const displayMetrics = snapshotMode ? {
    globalStep: 0,
    levelStep: snapshotMode.steps,
    reward: 0,
    performance: snapshotMode.finalPerformance,
    family: snapshotMode.family,
    level: snapshotMode.level,
    taskId: snapshotMode.taskId
  } : metrics;

  const displayTopology = selectedHistoricalTopology || (snapshotMode ? snapshotMode.topology : (topology || initialTopology));


  // Status label
  const statusLabel = (() => {
    if (snapshotMode) return 'SNAPSHOT';
    switch (uiState) {
      case 'INITIALIZING': return 'INITIALIZING...';
      case 'CONNECTING': return 'CONNECTING...';
      case 'READY': return 'READY';
      case 'WORLD_READY': return 'WORLD READY';
      case 'STARTING': return 'STARTING...';
      case 'TRAINING': return 'TRAINING';
      case 'STOPPING': return 'STOPPING';
      case 'STOPPED': return 'STOPPED';
      case 'COMPLETED': return 'COMPLETED';
      case 'ERROR': return 'ERROR';
      default: return 'IDLE';
    }
  })();

  const statusColor = (() => {
    switch (statusLabel) {
      case 'TRAINING': return 'text-emerald-400';
      case 'READY': return 'text-blue-400';
      case 'WORLD READY': return 'text-blue-400';
      case 'STOPPING': return 'text-amber-400';
      case 'STOPPED': return 'text-slate-400';
      case 'COMPLETED': return 'text-emerald-400';
      case 'ERROR': return 'text-red-400';
      case 'SNAPSHOT': return 'text-violet-400';
      default: return 'text-slate-500';
    }
  })();

  return (
    <>
    <main className="min-h-[100dvh] flex flex-col pt-4 pb-3 px-4 max-w-[1920px] mx-auto overflow-hidden h-[100dvh] text-slate-100 bg-[#020812]">
      {/* ─── Header ─────────────────────────────────────────────── */}
      <header className="flex items-center justify-between mb-4 mt-2 px-6 flex-none z-30 relative">
        <div>
          <div className="flex items-center gap-6">
            <h1 className="flex items-center">
              <span className="font-sans font-light tracking-[0.5em] text-slate-100 text-2xl uppercase">
                E V <span className="text-blue-400 drop-shadow-[0_0_12px_rgba(96,165,250,0.8)] relative"><span className="absolute inset-0 rounded-full bg-blue-400/20 blur-sm"></span>O</span> L V E
              </span>
            </h1>
            <span className="text-[10px] text-slate-500 tracking-[0.3em] uppercase font-mono mt-1">
              {snapshotMode ? `Snapshot · Level ${snapshotMode.level + 1}` : 'Self-Adaptive Neural Architecture'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4 font-mono text-sm">
          {uiState === 'WORLD_READY' && !snapshotMode && (
            <button
              onClick={() => { setAppState('INITIAL'); setEnvState(null); }}
              className="text-[10px] text-slate-500 hover:text-slate-300 px-3 py-1 transition-all uppercase tracking-widest"
            >
              ← BACK
            </button>
          )}

          {snapshotMode && !isHistoricalView && (
            <button
              onClick={handleReturnToLive}
              className="text-[10px] bg-violet-600/80 hover:bg-violet-500 text-white px-3 py-1 rounded-full transition-all uppercase tracking-widest"
            >
              Return to Live
            </button>
          )}

          {/* Status indicator */}
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${
              isLive && uiState === 'TRAINING' ? 'bg-emerald-500 shadow-[0_0_8px_#10b981] animate-pulse' :
              isLive ? 'bg-blue-500 shadow-[0_0_6px_#3b82f6]' :
              connectionStatus === 'connecting' ? 'bg-amber-500' :
              connectionStatus === 'error' ? 'bg-red-500' :
              'bg-slate-600'
            }`} />
            <span className={`text-xs ${statusColor}`}>{statusLabel}</span>
          </div>

          {/* Stop button */}
          {uiState === 'TRAINING' && !snapshotMode && (
            <button
              onClick={stopRun}
              className="text-[10px] bg-red-900/50 hover:bg-red-800 text-red-200 border border-red-800/50 px-3 py-1 rounded-full transition-all uppercase tracking-widest"
            >
              Stop Learning
            </button>
          )}
          {uiState === 'STOPPING' && !snapshotMode && (
            <button disabled className="text-[10px] bg-slate-800 text-slate-500 border border-slate-700 px-3 py-1 rounded-full cursor-not-allowed uppercase tracking-widest">
              Stopping...
            </button>
          )}

          {/* Family & Run labels */}
          {isActiveRun && (
            <span className="text-slate-500 border-l border-[#1a2744] pl-4 text-[10px] uppercase tracking-widest">
              EVOLVE WORLD
            </span>
          )}
          {runId && (
            <span className="text-slate-600 text-[10px] font-mono">
              {runId.substring(0, 12)}
            </span>
          )}
        </div>
      </header>

      {/* ─── Main Content ───────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-h-0 relative">

        {/* HERO: Neural Brain Topology */}
        <div 
          className={`transition-all duration-700 ease-in-out relative overflow-hidden flex flex-col justify-center cursor-pointer ${
            !isActiveRun 
              ? 'flex-1 min-h-[calc(100vh-100px)] bg-transparent border border-[rgba(80,110,160,0.15)] rounded-2xl shadow-[inset_0_0_100px_rgba(0,0,0,0.5)]' 
              : 'flex-1 min-h-0 bg-[#0a1121] rounded-t-xl border border-[#1a2744] border-b-0'
          }`}
          onPointerDown={(e) => {
            if (e.button !== 0) return;
            setDragStart({ x: e.clientX, y: e.clientY });
          }}
          onPointerUp={(e) => {
            if (e.button !== 0 || !dragStart) return;
            const dx = e.clientX - dragStart.x;
            const dy = e.clientY - dragStart.y;
            if (Math.sqrt(dx * dx + dy * dy) < 5) {
              setTopologyMode(m => m === 'ABSTRACT' ? 'ACTUAL' : 'ABSTRACT');
            }
            setDragStart(null);
          }}
        >
          {/* Ambient background layers for idle page */}
          {!isActiveRun && (
            <div className="absolute inset-0 pointer-events-none overflow-hidden flex items-center justify-center">
              {/* Layer 1: Subtle radial gradient */}
              <div className="absolute w-[1400px] h-[1400px] bg-[radial-gradient(circle,rgba(20,40,90,0.08)_0%,rgba(0,0,0,0)_60%)] rounded-full mix-blend-screen"></div>
              
              {/* Layer 2: Faint technical grid */}
              <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: 'radial-gradient(rgba(255, 255, 255, 0.5) 1px, transparent 1px)', backgroundSize: '50px 50px', backgroundPosition: 'center center' }}></div>
              
              {/* Layer 4: Orbital / Concentric Arcs */}
              <svg className="absolute inset-0 w-full h-full opacity-[0.25]" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid slice">
                {/* Center concentric circles */}
                <circle cx="960" cy="540" r="380" fill="none" stroke="rgba(96,165,250,0.15)" strokeWidth="1" strokeDasharray="2 12" />
                <circle cx="960" cy="540" r="550" fill="none" stroke="rgba(96,165,250,0.1)" strokeWidth="0.5" />
                <circle cx="960" cy="540" r="750" fill="none" stroke="rgba(96,165,250,0.05)" strokeWidth="1" />
                
                {/* Corner arcs simulating lenses/instruments */}
                <path d="M -50 200 Q 250 540 -50 880" fill="none" stroke="rgba(96,165,250,0.3)" strokeWidth="1.5" />
                <path d="M 1970 200 Q 1670 540 1970 880" fill="none" stroke="rgba(96,165,250,0.3)" strokeWidth="1.5" />
                
                {/* Subtle outer glow on lenses */}
                <path d="M -50 200 Q 250 540 -50 880" fill="none" stroke="rgba(96,165,250,0.1)" strokeWidth="8" filter="blur(4px)" />
                <path d="M 1970 200 Q 1670 540 1970 880" fill="none" stroke="rgba(96,165,250,0.1)" strokeWidth="8" filter="blur(4px)" />
              </svg>

              {/* Layer 5: Sparse ambient particles */}
              <div className="absolute w-1 h-1 bg-blue-300/40 rounded-full top-[30%] left-[22%] shadow-[0_0_8px_rgba(96,165,250,0.8)]"></div>
              <div className="absolute w-1 h-1 bg-blue-300/40 rounded-full top-[70%] left-[75%] shadow-[0_0_8px_rgba(96,165,250,0.8)]"></div>
              <div className="absolute w-0.5 h-0.5 bg-blue-200/50 rounded-full top-[25%] left-[65%]"></div>
              <div className="absolute w-0.5 h-0.5 bg-blue-200/50 rounded-full top-[65%] left-[25%]"></div>
              
              {/* Layer 6: Technical Atmosphere Text (Corner blocks) */}
              <div className="absolute top-24 left-16 text-[9px] font-mono tracking-[0.2em] text-slate-500/70 uppercase leading-loose text-left flex flex-col gap-8">
                <div>
                  Adaptive<br/>System
                </div>
                <div className="border-t border-slate-700/50 pt-3 text-slate-600/50">
                  Learn<br/>Evolve<br/>Generalize<br/>Repeat
                </div>
              </div>
              
              <div className="absolute top-24 right-16 text-[9px] font-mono tracking-[0.2em] text-slate-500/70 uppercase leading-loose text-right flex flex-col gap-8 items-end">
                <div>
                  Real-time<br/>Neural Topology
                </div>
                <div className="border-t border-slate-700/50 pt-3 text-slate-600/50 text-right">
                  Dynamic Modules<br/>Self-Improving<br/>Open-Ended<br/>Intelligence
                </div>
              </div>
              
              <div className="absolute top-32 left-1/2 -translate-x-1/2 text-[9px] font-mono tracking-[0.25em] text-slate-500/50 uppercase text-center flex flex-col items-center gap-3">
                <span className="w-8 h-[1px] bg-slate-700/80"></span>
                <span>A living neural architecture<br/>for open-ended learning</span>
              </div>

              <div className="absolute bottom-16 left-16 text-[9px] font-mono tracking-[0.2em] text-slate-500/60 uppercase leading-loose text-left flex flex-col gap-3">
                <span className="w-4 h-[1px] bg-slate-700/60 mb-1"></span>
                Exploration<br/>Drives<br/>Intelligence
                <span className="w-4 h-[1px] bg-slate-700/60 mt-1"></span>
              </div>
              
              <div className="absolute bottom-16 right-16 text-[9px] font-mono tracking-[0.2em] text-slate-500/60 uppercase leading-loose text-right flex flex-col gap-3 items-end">
                <span className="w-4 h-[1px] bg-slate-700/60 mb-1"></span>
                Scalable<br/>Adaptive<br/>Open-Ended
                <span className="w-4 h-[1px] bg-slate-700/60 mt-1"></span>
              </div>
            </div>
          )}

          {/* Crossfading Views */}
          <div className="absolute inset-0 z-10 flex flex-col pointer-events-none">
            <div className={`absolute inset-0 transition-opacity duration-300 ease-in-out ${topologyMode === 'ABSTRACT' ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}>
              <BrainTopology topology={displayTopology} isInitial={!isActiveRun} uiState={uiState} />
            </div>
            <div className={`absolute inset-0 transition-opacity duration-300 ease-in-out ${topologyMode === 'ACTUAL' ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}>
              <ActualArchitectureView topology={displayTopology} isInitial={!isActiveRun} uiState={uiState} />
            </div>
          </div>

          {/* Neural State Overlay */}
          <NeuralChangeIndicator 
            mutationState={mutationState} 
            topology={displayTopology} 
            uiState={uiState} 
            topologyMode={topologyMode} 
            setTopologyMode={setTopologyMode} 
          />

          {/* START LEARNING Button — ONLY shown when no run is active */}
          {!isActiveRun && (
            <div className="absolute bottom-16 left-1/2 -translate-x-1/2 flex flex-col items-center z-20">
              <div className="text-slate-400 text-xs mb-8 tracking-[0.3em] uppercase font-mono text-center leading-relaxed">
                A LIVING NEURAL ARCHITECTURE<br/>FOR CONTINUAL LEARNING
              </div>
              <button
                onPointerDown={e => e.stopPropagation()}
                onClick={uiState === 'INITIAL' ? handleEnterWorld : undefined}
                disabled={uiState !== 'INITIAL'}
                className={`group flex items-center gap-3 font-mono tracking-[0.2em] text-sm px-12 py-4 rounded-full border shadow-[0_0_20px_rgba(59,130,246,0.3)] transition-all uppercase pointer-events-auto backdrop-blur-sm ${
                  uiState === 'INITIAL' 
                  ? 'bg-blue-600/20 hover:bg-blue-600/40 text-blue-100 border-blue-400/50 hover:border-blue-300 cursor-pointer' 
                  : 'bg-slate-800 text-slate-500 border-slate-700 cursor-not-allowed'
                }`}
              >
                <span>
                  {uiState === 'INITIAL' ? 'ENTER WORLD' : uiState}
                </span>
              </button>
              <div className="text-slate-500/70 text-[10px] mt-6 font-mono tracking-[0.4em] uppercase">
                ONE WORLD · MANY TASKS
              </div>
            </div>
          )}
        </div>

        {/* LOWER AREA: Telemetry | Environment | Levels */}
        <div className={`transition-all duration-700 ease-in-out flex flex-1 min-h-0 overflow-hidden bg-[#0a1121] border border-[#1a2744] rounded-b-xl ${
          isActiveRun ? 'opacity-100' : 'opacity-0 h-0 border-t-0 overflow-hidden'
        }`}>

          {/* LEFT: Telemetry */}
          <div className="w-[22%] min-w-[260px] flex-none">
            <TelemetryPanel
              metrics={displayMetrics}
              decision={snapshotMode ? null : decision}
              mutationState={mutationState}
              isInitial={['WORLD_READY', 'INITIALIZING', 'CONNECTING', 'READY', 'STARTING'].includes(uiState)}
              topologyStats={{
                modules: displayTopology?.active_module_count || 1,
                parameters: displayTopology?.total_parameter_count || 0
              }}
              oodShift={oodShift}
              plasticity={plasticity}
            />
          </div>

          {/* CENTER: Environment */}
          <div className="flex-1 min-w-[300px] relative border-x border-[#1a2744]/40">
            {snapshotMode ? (
              <div className="w-full h-full flex items-center justify-center font-mono text-slate-500 bg-[#060a14]">
                <div className="text-center">
                  <div className="text-lg text-slate-300 mb-1">{snapshotMode.family}</div>
                  <div className="text-[10px] uppercase tracking-widest text-violet-400 mb-6">
                    Level {snapshotMode.level + 1} · Historical Snapshot
                  </div>
                  <div className="text-sm">
                    <span className="text-slate-400">Final Performance: </span>
                    <span className="text-emerald-400">{(snapshotMode.finalPerformance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {snapshotMode.steps.toLocaleString()} steps · {snapshotMode.converged ? 'Converged' : 'Did not converge'}
                  </div>
                </div>
              </div>
            ) : ['WORLD_READY', 'INITIALIZING', 'CONNECTING', 'READY', 'STARTING'].includes(uiState) ? (
              <div className="w-full h-full relative overflow-hidden bg-[#02050a]">
                {/* World Content - Blurred during WORLD_READY */}
                <div className={`w-full h-full transition-all duration-500 ease-in-out ${uiState === 'WORLD_READY' ? 'blur-[4px] opacity-80 brightness-75 pointer-events-none' : 'blur-0 opacity-100 brightness-100'}`}>
                  {envState ? (
                    <EnvironmentView phase={metrics.family} step={metrics.levelStep} envState={envState} uiState={uiState} level={metrics.level} />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center font-mono text-slate-500 bg-[#060a14]">
                      <div className="flex flex-col items-center gap-2">
                        <div className="w-5 h-5 border-2 border-slate-600 border-t-slate-300 rounded-full animate-spin" />
                        <span className="text-[10px] uppercase tracking-widest">Loading World Preview...</span>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Centered Launch Overlay */}
                <div className={`absolute inset-0 z-30 flex flex-col items-center justify-center transition-all duration-400 ease-in-out ${uiState === 'WORLD_READY' ? 'opacity-100 pointer-events-auto scale-100' : 'opacity-0 pointer-events-none scale-[0.98]'}`}>
                  <div className="bg-[#060a14]/85 backdrop-blur-sm border border-[#1a2744]/80 p-8 rounded flex flex-col items-center shadow-[0_10px_40px_rgba(0,0,0,0.8)] shadow-blue-900/10">
                    <div className="text-[11px] uppercase tracking-[0.25em] mb-2 text-blue-400/90 font-mono">World Ready</div>
                    <div className="text-xs text-slate-400 mb-8 uppercase tracking-widest font-mono text-center leading-relaxed">
                      Ready to begin continual<br/>learning
                    </div>
                    <button
                      onClick={handleStartLearning}
                      disabled={uiState !== 'WORLD_READY'}
                      className="bg-blue-600/10 border border-blue-500/30 hover:bg-blue-600/30 hover:border-blue-400/80 disabled:bg-slate-800/50 disabled:border-slate-700/50 disabled:text-slate-500 text-blue-100 font-mono tracking-[0.2em] text-xs px-12 py-3 rounded-sm shadow-[0_0_15px_rgba(59,130,246,0.1)] transition-all uppercase whitespace-nowrap"
                    >
                      {uiState === 'STARTING' ? 'STARTING...' : uiState === 'INITIALIZING' ? 'INITIALIZING...' : uiState === 'CONNECTING' || uiState === 'READY' ? 'CONNECTING...' : '▶ START LEARNING'}
                    </button>
                    {envState && (
                      <div className="text-[9px] text-slate-600 uppercase tracking-widest mt-6 font-mono">
                        LEVEL {String(envState.spec?.level || 1).padStart(2, '0')} · ADAPTIVE FRONTIER
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : envState ? (
              <EnvironmentView phase={metrics.family} step={metrics.levelStep} envState={envState} uiState={uiState} level={metrics.level} />
            ) : uiState === 'STOPPED' || uiState === 'STOPPING' ? (
              <div className="w-full h-full flex items-center justify-center font-mono text-slate-500 bg-[#060a14]">
                <div className="text-center flex flex-col items-center">
                  <div className="text-[10px] uppercase tracking-widest mb-3 text-amber-500/80">
                    {uiState === 'STOPPING' ? 'Stopping...' : 'Training Stopped'}
                  </div>
                  <div className="text-xl text-slate-200 mb-1">EVOLVE WORLD</div>
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">Run was manually halted</div>
                </div>
              </div>
            ) : uiState === 'COMPLETED' ? (
              <div className="w-full h-full flex items-center justify-center font-mono text-slate-500 bg-[#060a14]">
                <div className="text-center flex flex-col items-center">
                  <div className="text-[10px] uppercase tracking-widest mb-3 text-emerald-400/80">Experiment Complete</div>
                  <div className="text-xl text-slate-200 mb-1">EVOLVE WORLD</div>
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">All levels finished</div>
                </div>
              </div>
            ) : uiState === 'ERROR' ? (
              <div className="w-full h-full flex items-center justify-center font-mono text-slate-500 bg-[#060a14]">
                <div className="text-center flex flex-col items-center">
                  <div className="text-[10px] uppercase tracking-widest mb-3 text-red-500/80">Experiment Failed</div>
                  <div className="text-xl text-slate-200 mb-1">EVOLVE WORLD</div>
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">A fatal error occurred</div>
                </div>
              </div>
            ) : (
              <div className="w-full h-full flex items-center justify-center font-mono text-slate-500 bg-[#060a14]">
                <div className="flex flex-col items-center gap-2">
                  <div className="w-5 h-5 border-2 border-slate-600 border-t-slate-300 rounded-full animate-spin" />
                  <span className="text-[10px] uppercase tracking-widest">Initializing Environment</span>
                </div>
              </div>
            )}
          </div>

          {/* RIGHT: Evolution Trace */}
          <div className="w-[20%] min-w-[220px] flex-none flex flex-col border-l border-[#1a2744]/60 bg-[#060a14]">
            {isActiveRun ? (
              <>
                <div className="h-1/2 flex flex-col border-b border-[#1a2744]/60">
                  <HistoryExplorer
                    levelsHistory={levelsHistory}
                    selectedLevelId={selectedLevelId}
                    selectedHistoryId={selectedHistoryId}
                    historyEntries={historyEntries}
                    historyLoading={historyLoading}
                    selectLevel={selectLevel}
                    selectHistoryEntry={selectHistoryEntry}
                    isHistoricalView={isHistoricalView}
                    returnToLive={handleReturnToLive}
                  />
                </div>
                <div className="h-1/2 flex flex-col">
                  <EventTimeline events={displayEvents} />
                </div>
              </>
            ) : (
              <div className="w-full h-full p-5 flex flex-col font-mono relative opacity-50 justify-center items-center">
                <div className="text-slate-500 uppercase tracking-widest text-xs">Waiting for trace</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ─── Modals ─────────────────────────────────────────────── */}

    </main>
    <button
      onClick={() => {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(err => console.log(err));
        } else {
          document.exitFullscreen();
        }
      }}
      className="absolute bottom-4 right-4 z-50 text-slate-500 hover:text-slate-300 font-mono text-[9px] tracking-widest uppercase transition-colors"
    >
      [ F11 ] FULLSCREEN
    </button>
    </>
  );
}
