'use client';
import React from 'react';

interface EvolutionStatusProps {
  currentDecision: 'REUSE' | 'ADAPT' | 'GROW' | 'PRUNE' | null;
}

export const EvolutionStatus: React.FC<EvolutionStatusProps> = ({ currentDecision }) => {
  const steps = ['REUSE', 'ADAPT', 'GROW', 'PRUNE', 'REWIRE'];
  
  return (
    <div className="flex items-center justify-center gap-4 py-6 text-sm font-mono tracking-widest text-slate-500 uppercase">
      {steps.map((step, idx) => {
        const isActive = currentDecision === step;
        return (
          <React.Fragment key={step}>
            <div className={`px-6 py-2 rounded-full border transition-all duration-300 ${
              isActive 
                ? 'border-amber-400 text-amber-400 text-glow-amber bg-amber-950/20 shadow-[0_0_15px_rgba(245,158,11,0.2)]' 
                : 'border-slate-800 text-slate-500'
            }`}>
              {step} {isActive && <span className="ml-1 text-[10px] bg-amber-900/50 px-1.5 py-0.5 rounded text-amber-200">ACTIVE</span>}
            </div>
            {idx < steps.length - 1 && (
              <span className="text-slate-700">→</span>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
