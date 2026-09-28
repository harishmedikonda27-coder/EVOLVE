import React from 'react';

interface AdaptiveFrontierState {
  grid: number[][];
  agent_pos: [number, number];
  target_pos: [number, number];
  grid_size: number;
  has_resource: boolean;
  cue_active: boolean;
  uncertainty: number;
  step_count: number;
  spec: {
    level: number;
    objective: string;
  };
  movements?: any[];
}

interface Props {
  envState: AdaptiveFrontierState;
  step: number;
}

export const AdaptiveFrontierRenderer: React.FC<Props> = ({ envState, step }) => {
  const { grid, grid_size, spec } = envState;

  // Object types:
  // EMPTY = 0, OBSTACLE = 1, TARGET = 2, RESOURCE = 3, GATE = 4, HAZARD = 5, DYNAMIC = 6, AGENT = 7

  const renderCellContent = (cellType: number, r: number, c: number) => {
    const isAgent = envState.agent_pos && r === envState.agent_pos[0] && c === envState.agent_pos[1];
    const isTarget = envState.target_pos && r === envState.target_pos[0] && c === envState.target_pos[1];
    
    return (
      <>
        {/* Empty traversable floor styling for empty cells */}
        {(cellType === 0 && !isTarget) && (
          <div className="absolute inset-[1px] bg-[#080d1a] border border-[#121b2e] rounded-[1px]"></div>
        )}

        {/* Cell Background / Structure */}
        {cellType === 1 && (
          <div className="absolute inset-[1px] bg-slate-800 border border-slate-700/80 rounded-[1px] shadow-[inset_0_0_8px_rgba(0,0,0,0.5)]">
            <div className="w-full h-full opacity-30" style={{ backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(255,255,255,0.1) 4px, rgba(255,255,255,0.1) 8px)' }}></div>
          </div>
        )}
        {(cellType === 2 || isTarget) && (
          <div className="absolute inset-[1px] bg-purple-900/20 border border-purple-500/30 flex items-center justify-center z-10">
             <div className="w-2/3 h-2/3 rounded-full border border-purple-400/80 flex items-center justify-center animate-pulse shadow-[0_0_12px_rgba(168,85,247,0.4)]">
                <div className="w-1/3 h-1/3 rounded-full bg-purple-400"></div>
             </div>
          </div>
        )}
        {cellType === 3 && (
          <div className="absolute inset-[1px] bg-emerald-900/10 flex items-center justify-center">
             <div className="text-emerald-400 text-xl drop-shadow-[0_0_8px_rgba(52,211,153,0.8)] animate-pulse" style={{ animationDuration: '2s' }}>◆</div>
          </div>
        )}
        {cellType === 4 && (
          <div className="absolute inset-[1px] bg-amber-900/20 border border-amber-500/40 flex items-center justify-center">
             <div className="text-amber-500 text-lg font-bold drop-shadow-[0_0_6px_rgba(245,158,11,0.6)]">║</div>
          </div>
        )}
        {cellType === 5 && (
          <div className="absolute inset-[1px] bg-red-900/20 border border-red-500/30 flex items-center justify-center">
             <div className="text-red-500 text-base drop-shadow-[0_0_8px_rgba(239,68,68,0.8)] animate-pulse">▲</div>
          </div>
        )}
        {cellType === 6 && (
          <div className="absolute inset-[1px] border border-blue-500/40 border-dashed bg-blue-900/10 animate-[pulse_2s_ease-in-out_infinite]"></div>
        )}

        {/* Empty traversable floor styling for empty cells */}
        {/* Moved to top to avoid overriding others */}

        {/* Agent overlay */}
        {isAgent && (
          <div className="absolute inset-[15%] bg-cyan-400 rounded-full shadow-[0_0_15px_rgba(34,211,238,0.9)] z-20 flex items-center justify-center">
             <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
             <div className="absolute -top-1 w-1 h-1 border-l-2 border-t-2 border-cyan-200 rotate-45 transform"></div>
          </div>
        )}
      </>
    );
  };

  const formatObjective = (obj: string | undefined) => {
    if (!obj) return 'NAVIGATE ENVIRONMENT';
    const mapping: Record<string, string> = {
      'DIRECT_REACH': 'REACH TARGET',
      'RESOURCE_THEN_REACH': 'COLLECT RESOURCE → REACH TARGET',
      'RESOURCE_GATE_REACH': 'RESOURCE → OPEN GATE → REACH TARGET',
      'AVOID_AND_REACH': 'AVOID HAZARDS → REACH TARGET',
      'CUE_THEN_REACH': 'REMEMBER CUE → REACH TARGET'
    };
    return mapping[obj] || obj.replace(/_/g, ' ');
  };

  const agentState = envState.movements && envState.movements.length > 0 
    ? `LAST ACTION: ${envState.movements[envState.movements.length - 1].action}`
    : 'NAVIGATING';

  return (
    <div className="w-full h-full flex flex-col p-6 bg-[#040812] relative overflow-hidden text-slate-200">
      {/* Header */}
      <div className="flex justify-between items-start mb-6">
        <div>
          <div className="font-mono text-sm tracking-[0.3em] text-slate-100 uppercase drop-shadow-[0_0_5px_rgba(255,255,255,0.3)]">Evolve World</div>
          <div className="font-mono text-[10px] tracking-widest text-cyan-400 mt-1 uppercase">Adaptive Frontier</div>
        </div>
        <div className="font-mono text-xs tracking-widest text-slate-400 border border-slate-700/50 px-3 py-1 rounded-full shadow-[inset_0_0_10px_rgba(0,0,0,0.5)] bg-[#0a0f1c]">
          LEVEL {spec?.level || 1} / ∞
        </div>
      </div>

      <div className="flex flex-1 min-h-0 gap-8 items-center justify-center">
        {/* Grid Container */}
        <div className="flex-1 flex items-center justify-center min-w-0 h-full">
          <div 
            className="relative bg-[#020408] border-2 border-[#1a2744] p-1.5 rounded shadow-[0_0_40px_rgba(0,0,0,0.8)]"
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${grid_size}, 1fr)`,
              gridTemplateRows: `repeat(${grid_size}, 1fr)`,
              aspectRatio: '1/1',
              maxHeight: '100%',
              width: 'auto',
              minWidth: '50%'
            }}
          >
            {grid && grid.map((row, r) => 
              row.map((cell, c) => (
                <div key={`${r}-${c}`} className="relative w-full h-full">
                  {renderCellContent(cell, r, c)}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Side Panel: Objective & Status */}
        <div className="w-64 flex-none flex flex-col gap-6 font-mono">
          <div className="bg-[#0a0f1c] border border-[#1a2744] p-4 rounded-lg shadow-lg relative overflow-hidden">
            <div className="absolute top-0 left-0 w-1 h-full bg-emerald-500"></div>
            <div className="text-[9px] text-slate-500 tracking-widest mb-2 uppercase">Current Objective</div>
            <div className="text-sm tracking-wider text-emerald-400 leading-relaxed uppercase">
              {formatObjective(spec?.objective)}
            </div>
          </div>

          <div className="bg-[#0a0f1c] border border-[#1a2744] p-4 rounded-lg shadow-lg">
            <div className="text-[9px] text-slate-500 tracking-widest mb-4 uppercase">World Status</div>
            <div className="text-[10px] tracking-wider text-slate-300 flex flex-col gap-3">
              <div className="flex justify-between items-center">
                <span className="text-slate-500">AGENT STATE</span>
                <span className={agentState !== 'NAVIGATING' ? "text-cyan-400" : "text-slate-300"}>
                  {agentState}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-500">UNCERTAINTY</span>
                <span className={envState?.uncertainty > 0 ? "text-amber-400" : "text-slate-400"}>
                  {(envState?.uncertainty * 100 || 0).toFixed(0)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-500">MEMORY CUE</span>
                <span className={envState?.cue_active ? "text-amber-400 animate-pulse" : "text-slate-500"}>
                  {envState?.cue_active ? 'ACTIVE' : 'INACTIVE'}
                </span>
              </div>
              <div className="flex justify-between items-center border-t border-[#1a2744] pt-3 mt-1">
                <span className="text-slate-500">TRAINING STEP</span>
                <span className="text-emerald-400 font-bold text-xs">{step}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 flex-none border-t border-[#1a2744] pt-4 flex flex-wrap gap-x-8 gap-y-3 justify-center text-[10px] font-mono tracking-widest text-slate-400">
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]"></div> AGENT</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full border border-purple-400 flex items-center justify-center"><div className="w-1.5 h-1.5 bg-purple-400 rounded-full"></div></div> TARGET</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm bg-slate-800 border border-slate-700" style={{ backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,255,255,0.2) 2px, rgba(255,255,255,0.2) 4px)' }}></div> OBSTACLE</div>
        <div className="flex items-center gap-2"><div className="text-emerald-400 text-xs">◆</div> RESOURCE</div>
        <div className="flex items-center gap-2"><div className="text-amber-500 font-bold text-xs">║</div> GATE</div>
        <div className="flex items-center gap-2"><div className="text-red-500 text-xs">▲</div> HAZARD</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm border border-blue-500/50 border-dashed"></div> DYNAMIC</div>
      </div>
    </div>
  );
};
