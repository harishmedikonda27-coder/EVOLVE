'use client';
import React from 'react';
import { Signals } from '../types';

interface ControllerPanelProps {
  signals: Signals;
  needScore: number;
}

export const ControllerPanel: React.FC<ControllerPanelProps> = ({ signals, needScore }) => {
  const signalItems = [
    { label: 'Performance Gap (P)', value: signals.performance_gap, color: 'bg-red-500' },
    { label: 'Stagnation (S)', value: signals.stagnation, color: 'bg-amber-500' },
    { label: 'Task Novelty (N)', value: signals.novelty, color: 'bg-blue-500' },
    { label: 'Interference (I)', value: signals.interference, color: 'bg-purple-500' },
    { label: 'Utilization (U)', value: signals.utilization, color: 'bg-emerald-500' },
    { label: 'Structural Cost (C)', value: signals.cost, color: 'bg-slate-500' },
  ];

  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="font-semibold text-lg text-slate-100 flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-purple-500 animate-pulse" />
          Evolution Controller Telemetry
        </h3>
        <div className="flex items-center gap-2 bg-purple-950/80 border border-purple-700/80 px-3 py-1 rounded-lg">
          <span className="text-xs text-purple-300 font-medium">Need Score:</span>
          <span className="text-base font-mono font-bold text-purple-200">{needScore.toFixed(3)}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {signalItems.map((item) => (
          <div key={item.label} className="bg-slate-950 border border-slate-800 p-3 rounded-lg flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs text-slate-300 font-medium">
              <span>{item.label}</span>
              <span className="font-mono font-bold">{item.value.toFixed(2)}</span>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden">
              <div
                className={`${item.color} h-full rounded-full transition-all duration-500`}
                style={{ width: `${Math.round(item.value * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
