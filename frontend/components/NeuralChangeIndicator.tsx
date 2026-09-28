'use client';
import React, { useEffect, useState } from 'react';
import { TopologySnapshot } from '../types';
import { MutationState } from '../lib/useEvolveState';

interface Props {
  mutationState: MutationState;
  topology: TopologySnapshot | null;
  uiState: string;
  topologyMode: 'ABSTRACT' | 'ACTUAL';
  setTopologyMode: (mode: 'ABSTRACT' | 'ACTUAL') => void;
}

export const NeuralChangeIndicator: React.FC<Props> = ({ mutationState, topology, uiState, topologyMode, setTopologyMode }) => {
  const [displayText, setDisplayText] = useState('STABLE — NO STRUCTURAL CHANGE');
  const [color, setColor] = useState('text-slate-400');
  const [isPulsing, setIsPulsing] = useState(false);

  useEffect(() => {
    if (!mutationState) return;
    const { status, action, record } = mutationState;

    let text = 'STABLE — NO STRUCTURAL CHANGE';
    let c = 'text-slate-400';
    let pulsing = false;

    if (action === 'ADAPT') {
      text = 'ADAPT — LEARNING';
      c = 'text-blue-400';
      pulsing = true;
    } else if (action === 'REUSE') {
      text = 'REUSE — ACTIVE';
      c = 'text-blue-400';
      pulsing = true;
    } else if (status === 'PROPOSED') {
      pulsing = true;
      if (action === 'REWIRE') { text = 'REWIRE — VERIFYING'; c = 'text-blue-400'; }
      else if (action === 'GROW') { text = 'GROW — VERIFYING'; c = 'text-emerald-400'; }
      else if (action === 'PRUNE') { text = 'PRUNE — VERIFYING'; c = 'text-amber-500'; }
      else { text = `${action} — VERIFYING`; }
    } else if (status === 'ACCEPTED') {
      if (action === 'REWIRE') { text = `REWIRE — ACCEPTED`; c = 'text-blue-400'; }
      else if (action === 'GROW') { text = `GROW — ACCEPTED`; c = 'text-emerald-400'; }
      else if (action === 'PRUNE') { text = `PRUNE — ACCEPTED`; c = 'text-amber-500'; }
      else { text = `${action} — ACCEPTED`; }
    } else if (status === 'ROLLBACK' || status === 'REJECTED') {
      text = `${action || 'MUTATION'} — ROLLED BACK`;
      c = 'text-amber-500';
    }

    // Check if it's a no-op rewire
    if (action === 'REWIRE' && record && record.rewire_diff) {
        if (record.rewire_diff.added_edge_count === 0 && record.rewire_diff.removed_edge_count === 0 && (status === 'ACCEPTED' || status === 'ROLLBACK')) {
            text = 'REWIRE — NO STRUCTURAL CHANGE';
            c = 'text-slate-400';
        }
    }

    setDisplayText(text);
    setColor(c);
    setIsPulsing(pulsing);
  }, [mutationState]);

  if (uiState === 'INITIAL' || uiState === 'SELECTING_ENV') {
    return null;
  }

  const { status, action, record } = mutationState;

  return (
    <div className="absolute top-6 right-8 z-40 flex flex-col items-end pointer-events-none">
      <div className="font-mono text-[9px] tracking-[0.2em] text-slate-500 uppercase mb-1 drop-shadow-[0_0_8px_rgba(0,0,0,0.8)]">
        STRUCTURAL INTERVENTION
      </div>
      <div className={`font-mono text-[10px] font-bold tracking-[0.1em] uppercase flex items-center gap-2 ${color} drop-shadow-[0_0_8px_rgba(255,255,255,0.1)]`}>
        {displayText}
        {isPulsing && (
          <span className="relative flex h-1.5 w-1.5 ml-2">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-current`}></span>
            <span className={`relative inline-flex rounded-full h-1.5 w-1.5 bg-current`}></span>
          </span>
        )}
      </div>

      {status === 'PROPOSED' && record && (
        <div className="mt-3 bg-slate-900/80 border border-[#1a2744] p-3 flex gap-6 text-[10px] text-slate-400 font-mono tracking-widest text-left backdrop-blur-sm">
          <div>
            <div className="text-slate-500 mb-1 border-b border-slate-700/50 pb-1">BEFORE</div>
            <div>Modules: {record.topology_before?.active_module_count ?? '?'}</div>
            <div>Params: {record.parameter_count_before?.toLocaleString() ?? '?'}</div>
            <div>Edges: {record.rewire_diff?.before_edge_count?.toLocaleString() ?? '?'}</div>
          </div>
          <div>
            <div className={`${action === 'REWIRE' ? 'text-blue-400' : action === 'GROW' ? 'text-emerald-400' : 'text-amber-400'} mb-1 border-b border-slate-700/50 pb-1`}>CANDIDATE</div>
            <div className="text-slate-300">Modules: {record.topology_after?.active_module_count ?? '?'}</div>
            <div className="text-slate-300">Params: {record.parameter_count_after?.toLocaleString() ?? '?'}</div>
            <div className="text-slate-300">Edges: {record.rewire_diff?.after_edge_count?.toLocaleString() ?? '?'}</div>
          </div>
        </div>
      )}

      {status === 'ACCEPTED' && record && (
        <div className="mt-3 bg-[#0a1121]/90 border border-blue-900/40 p-3 flex flex-col gap-3 text-[9px] text-slate-400 font-mono tracking-widest text-left backdrop-blur-sm min-w-[220px]">
          <div>
             <div className="text-blue-400 mb-1 border-b border-blue-900/50 pb-1 uppercase">Performance</div>
             <div className="flex justify-between">
                <span>BEFORE</span><span>{record.new_task_metric_before?.toFixed(4) ?? '—'}</span>
             </div>
             <div className="flex justify-between">
                <span>AFTER</span><span className="text-slate-200">{record.new_task_metric_after?.toFixed(4) ?? '—'}</span>
             </div>
             <div className="flex justify-between mt-1 text-blue-300">
                <span>GAIN</span><span>{((record.new_task_metric_after || 0) - (record.new_task_metric_before || 0)).toFixed(4)}</span>
             </div>
          </div>
          
          {record.rewire_diff && (
             <div>
                <div className="text-blue-400 mb-1 border-b border-blue-900/50 pb-1 uppercase">Connectivity</div>
                <div className="flex justify-between text-emerald-400">
                   <span>ADDED</span><span>+{record.rewire_diff.added_edge_count}</span>
                </div>
                <div className="flex justify-between text-rose-400">
                   <span>REMOVED</span><span>-{record.rewire_diff.removed_edge_count}</span>
                </div>
             </div>
          )}
          
          <div>
             <div className="text-blue-400 mb-1 border-b border-blue-900/50 pb-1 uppercase">Capacity</div>
             <div className="flex justify-between">
                <span>MODULES</span><span>{record.topology_before?.active_module_count ?? '?'} &rarr; {record.topology_after?.active_module_count ?? '?'}</span>
             </div>
             <div className="flex justify-between">
                <span>PARAMS</span><span>{record.parameter_count_before?.toLocaleString() ?? '?'} &rarr; {record.parameter_count_after?.toLocaleString() ?? '?'}</span>
             </div>
          </div>
        </div>
      )}

      {status === 'ROLLBACK' && record && (
        <div className="mt-3 bg-amber-900/20 border border-amber-500/20 p-3 flex flex-col gap-2 text-[10px] text-amber-500/80 font-mono tracking-widest min-w-[200px]">
          <div className="text-center font-bold mb-1 border-b border-amber-500/20 pb-1">TOPOLOGY RESTORED</div>
          {record.rewire_diff && (
             <div className="text-[9px]">
                <div className="mb-1 text-amber-400/80">STRUCTURAL CHANGE REVERSED</div>
                <div className="flex justify-between text-emerald-500/60"><span>ADDED</span><span>+{record.rewire_diff.added_edge_count}</span></div>
                <div className="flex justify-between text-rose-500/60"><span>REMOVED</span><span>-{record.rewire_diff.removed_edge_count}</span></div>
             </div>
          )}
        </div>
      )}

      <div className="mt-3 flex items-center pointer-events-auto bg-[#050a14] border border-[#1a2744] rounded overflow-hidden font-mono text-[9px] tracking-widest">
        <button
          onClick={() => setTopologyMode('ABSTRACT')}
          className={`px-3 py-1.5 transition-colors ${topologyMode === 'ABSTRACT' ? 'bg-[#1a2744] text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-[#0a1121]'}`}
        >
          ABSTRACT
        </button>
        <button
          onClick={() => setTopologyMode('ACTUAL')}
          className={`px-3 py-1.5 transition-colors ${topologyMode === 'ACTUAL' ? 'bg-[#1a2744] text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-[#0a1121]'}`}
        >
          ACTUAL ARCHITECTURE
        </button>
      </div>
    </div>
  );
};
