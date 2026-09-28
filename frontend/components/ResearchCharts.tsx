'use client';
import React from 'react';

export const ResearchCharts: React.FC = () => {
  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-3">
      <h3 className="font-semibold text-lg text-slate-100 flex items-center gap-2">
        Adaptation & Retention Analytics
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Performance Curve Mock Chart */}
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex flex-col gap-2 h-44">
          <span className="text-xs font-medium text-slate-400">Task Performance vs Step</span>
          <div className="flex-1 border-b border-l border-slate-800 flex items-end p-2 gap-1 relative">
            <div className="absolute left-1/2 top-0 bottom-0 border-l border-dashed border-red-500/60" />
            <span className="absolute left-[51%] top-1 text-[9px] text-red-400 font-mono">Shift to B</span>
            {/* Mock curve bars */}
            {[0.2, 0.4, 0.6, 0.85, 0.9, 0.25, 0.45, 0.65, 0.82, 0.88].map((val, idx) => (
              <div
                key={idx}
                className="flex-1 bg-blue-500/60 hover:bg-blue-400 transition-all rounded-t-sm"
                style={{ height: `${val * 100}%` }}
              />
            ))}
          </div>
        </div>

        {/* Parameter Growth Mock Chart */}
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex flex-col gap-2 h-44">
          <span className="text-xs font-medium text-slate-400">Parameter Capacity Growth</span>
          <div className="flex-1 border-b border-l border-slate-800 flex items-end p-2 gap-1">
            {[6200, 6200, 6200, 6200, 6200, 8312, 8312, 8312, 8312, 8312].map((val, idx) => (
              <div
                key={idx}
                className="flex-1 bg-emerald-500/60 hover:bg-emerald-400 transition-all rounded-t-sm"
                style={{ height: `${(val / 10000) * 100}%` }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
