'use client';
import React from 'react';
import { AdaptiveFrontierRenderer } from './environment/AdaptiveFrontierRenderer';

interface EnvironmentViewProps {
  phase: string;
  step: number;
  envState: any;
  uiState?: string;
  level?: number;
}

export const EnvironmentView: React.FC<EnvironmentViewProps> = ({ phase, step, envState, uiState, level }) => {
  if (!envState) {
    return (
      <div className="w-full h-full flex items-center justify-center font-mono text-slate-500 bg-[#060a14]">
        <div className="flex flex-col items-center gap-2">
          <div className="w-6 h-6 border-2 border-slate-600 border-t-slate-300 rounded-full animate-spin" />
          <span className="text-xs uppercase tracking-widest">Awaiting Environment State</span>
        </div>
      </div>
    );
  }

  // Validate minimum required fields
  if (!envState.grid_size || !envState.agent_pos) {
    return (
      <div className="w-full h-full flex items-center justify-center font-mono text-red-500/60 bg-[#060a14]">
        <span className="text-xs uppercase tracking-widest">Environment State Malformed</span>
      </div>
    );
  }

  return <AdaptiveFrontierRenderer envState={envState} step={step} />;
};
