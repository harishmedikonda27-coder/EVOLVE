'use client';
import React from 'react';
import { EvolveMetrics, MutationState } from '../lib/useEvolveState';
import { Decision, ConsequenceResolution } from '../types';

interface TelemetryPanelProps {
  metrics: EvolveMetrics;
  decision: Decision | null;
  mutationState: MutationState;
  isInitial?: boolean;
  topologyStats: {
    modules: number;
    parameters: number;
  };
  oodShift?: number;
  plasticity?: any;
  consequences?: ConsequenceResolution[];
}

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({
  metrics,
  decision,
  mutationState,
  isInitial,
  topologyStats,
  oodShift = 0,
  plasticity = null,
  consequences = []
}) => {
  const formatParams = (count: number) => {
    if (count > 1000000) return `${(count / 1000000).toFixed(2)}M`;
    if (count > 1000) return `${(count / 1000).toFixed(1)}K`;
    return count.toString();
  };

  const dash = '\u2014'; // em dash

  const adaptationState = decision?.adaptation_state;
  const signals = decision?.signals;

  const renderMetricRow = (label: string, value: number | undefined, valueStr: string | React.ReactNode, isWarning = false, isGood = false) => (
    <div key={label} className="grid grid-cols-[minmax(0,1fr)_auto] items-center group gap-2">
      <span className="text-[10px] uppercase tracking-widest text-slate-500 whitespace-nowrap overflow-hidden text-ellipsis">
        {label}
      </span>
      <div className="flex items-center justify-end gap-2">
        {value !== undefined && (
          <div className="w-12 h-1 bg-[#1a2744] rounded-full overflow-hidden hidden xl:block opacity-30 group-hover:opacity-100 transition-opacity">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${Math.min(Math.abs(value) * 100, 100)}%`,
                backgroundColor: isWarning ? '#f59e0b' : isGood ? '#10b981' : '#475569',
              }}
            />
          </div>
        )}
        <span className={`text-sm tabular-nums whitespace-nowrap text-right ${isWarning ? 'text-amber-400' : isGood ? 'text-emerald-400' : 'text-slate-300'}`}>
          {valueStr}
        </span>
      </div>
    </div>
  );

  return (
    <div className="w-full h-full border-r border-[#1a2744]/60 bg-[#060a14] p-4 pb-6 flex flex-col font-mono relative overflow-hidden">
      <h2 className="text-[10px] text-slate-500 tracking-[0.2em] uppercase mb-2 pb-2 border-b border-[#1a2744]/60 flex-none">
        Learning Health
      </h2>

      <div className="flex-1 flex flex-col gap-3">
        
        {/* LEARNING */}
        <div className="flex flex-col gap-1.5">
          <h3 className="text-[9px] text-blue-400/70 tracking-[0.3em] uppercase mb-1">Learning</h3>
          {renderMetricRow('CURRENT TASK', metrics.performance, isInitial ? dash : `${(metrics.performance * 100).toFixed(1)}%`, false, metrics.performance >= 0.8)}
          {renderMetricRow('RETENTION', metrics.retention ?? undefined, metrics.retention != null ? `${(metrics.retention * 100).toFixed(1)}%` : dash)}
          {renderMetricRow('FORGETTING', metrics.forgetting ?? undefined, metrics.forgetting != null ? `${(metrics.forgetting * 100).toFixed(1)}%` : dash)}
        </div>

        {/* ADAPTATION */}
        <div className="flex flex-col gap-1.5 pt-2 border-t border-[#1a2744]/30">
          <h3 className="text-[9px] text-blue-400/70 tracking-[0.3em] uppercase mb-1">Adaptation</h3>
          {(() => {
            if (!plasticity || !plasticity.state) return renderMetricRow('PLASTICITY', undefined, 'NOT TESTED');
            const init = (plasticity.initial_perf * 100).toFixed(1);
            const fin = (plasticity.final_perf * 100).toFixed(1);
            return (
              <div key="PLASTICITY" className="grid grid-cols-[minmax(0,1fr)_auto] items-center group gap-2">
                <span className="text-[10px] uppercase tracking-widest text-slate-500">PLASTICITY</span>
                <div className="flex flex-col items-end text-right">
                  <span className="text-sm tabular-nums text-slate-300">{init}% → {fin}%</span>
                  {plasticity.auc !== undefined && (
                    <span className="text-[10px] text-slate-500">AUC {plasticity.auc.toFixed(3)} <span className="text-slate-600 ml-1">{plasticity.state}</span></span>
                  )}
                </div>
              </div>
            );
          })()}
          {(() => {
            const shift = metrics.ood_shift ?? 0;
            if (shift > 0) {
              return (
                <div key="OOD SHIFT" className="grid grid-cols-[minmax(0,1fr)_auto] items-center group gap-2">
                  <span className="text-[10px] uppercase tracking-widest text-slate-500">OOD SHIFT</span>
                  <div className="flex flex-col items-end text-right">
                    <span className="text-sm tabular-nums text-amber-400">+{shift.toFixed(3)}</span>
                    <span className="text-[10px] text-slate-500">DETECTED</span>
                  </div>
                </div>
              );
            }
            return renderMetricRow('OOD SHIFT', undefined, 'NONE');
          })()}
          {renderMetricRow('OOD RECOVERY', metrics.ood_recovery ?? undefined, metrics.ood_recovery != null ? `${(metrics.ood_recovery * 100).toFixed(0)}%` : dash)}
        </div>

        {/* STRUCTURE */}
        <div className="flex flex-col gap-1.5 pt-2 border-t border-[#1a2744]/30">
          <h3 className="text-[9px] text-blue-400/70 tracking-[0.3em] uppercase mb-1">Structure</h3>
          {renderMetricRow('MODULES', undefined, topologyStats.modules)}
          {renderMetricRow('PARAMETERS', undefined, formatParams(topologyStats.parameters))}
          
          {/* Last Action logic */}
          {(() => {
            let lastActionStr = dash;
            if (mutationState.status !== 'IDLE' && mutationState.action) {
               if (mutationState.status === 'PROPOSED') lastActionStr = `${mutationState.action} · PENDING`;
               else if (mutationState.status === 'ACCEPTED') lastActionStr = `${mutationState.action}`;
               else if (mutationState.status === 'ROLLBACK') lastActionStr = `${mutationState.action} · ROLLED BACK`;
               else if (mutationState.status === 'REJECTED') lastActionStr = `${mutationState.action} · REJECTED`;
            } else if (decision?.action) {
               lastActionStr = decision.action;
            } else if (!isInitial) {
               lastActionStr = 'REUSE';
            }
            return renderMetricRow('LAST ACTION', undefined, lastActionStr);
          })()}
        </div>
      </div>
    </div>
  );
};
