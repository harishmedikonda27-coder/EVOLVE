'use client';
import React from 'react';
import { Decision } from '../types';

interface DecisionCardProps {
  decision: Decision;
}

export const DecisionCard: React.FC<DecisionCardProps> = ({ decision }) => {
  let badgeColor = 'bg-blue-500/20 text-blue-400 border-blue-500/40';
  if (decision.action === 'GROW') badgeColor = 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40';
  if (decision.action === 'PRUNE') badgeColor = 'bg-amber-500/20 text-amber-400 border-amber-500/40';

  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-lg text-slate-100 flex items-center gap-2">
          Controller Decision
        </h3>
        <span className={`text-xs font-mono font-bold px-3 py-1 rounded-full border ${badgeColor}`}>
          {decision.action}
        </span>
      </div>

      <p className="text-sm text-slate-300 bg-slate-950 border border-slate-800 p-3 rounded-lg leading-relaxed">
        {decision.reason}
      </p>

      <div className="flex items-center justify-between text-xs text-slate-400 border-t border-slate-800 pt-2 font-mono">
        <span>Growth Budget Remaining: {decision.budget_remaining}</span>
        <span>Cooldown Active: {decision.cooldown_active ? 'Yes' : 'No'}</span>
      </div>
    </div>
  );
};
