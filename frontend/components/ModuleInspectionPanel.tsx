'use client';
import React from 'react';
import { NodeInfo } from '../types';

interface ModuleInspectionPanelProps {
  node: NodeInfo;
  x: number;
  y: number;
  onClose: () => void;
}

export const ModuleInspectionPanel: React.FC<ModuleInspectionPanelProps> = ({ node, x, y, onClose }) => {
  // Try to position the panel so it doesn't fall off screen. 
  // We'll just put it to the right and slightly up from the node.
  const isNew = node.created_step && node.created_step > 0;
  
  return (
    <div 
      className="absolute z-50 bg-[#0d1627] border border-slate-700/80 rounded-lg shadow-2xl p-4 text-xs font-mono w-[240px] pointer-events-auto"
      style={{ left: x + 20, top: y - 80 }}
    >
      <button 
        onClick={onClose}
        className="absolute top-2 right-2 text-slate-500 hover:text-slate-300"
      >
        ✕
      </button>
      
      <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-800">
        <span className="font-bold text-slate-200">
          {node.id === 'encoder' ? 'Encoder' : 
           node.id === 'router' ? 'Router' : 
           node.id === 'head' ? 'Output Head' : 
           `Module ${node.id.replace('module_', '')}`}
        </span>
        {isNew && <span className="text-amber-400 text-[10px]">(New)</span>}
      </div>

      <div className="flex flex-col gap-1.5 text-slate-400">
        <div className="flex justify-between">
          <span>Type</span>
          <span className="text-slate-200">{node.kind}</span>
        </div>
        {node.input_dim && (
          <div className="flex justify-between">
            <span>Input Dim</span>
            <span className="text-slate-200">{node.input_dim}</span>
          </div>
        )}
        {node.output_dim && (
          <div className="flex justify-between">
            <span>Output Dim</span>
            <span className="text-slate-200">{node.output_dim}</span>
          </div>
        )}
        {node.parameter_count !== undefined && (
          <div className="flex justify-between">
            <span>Parameters</span>
            <span className="text-amber-400">{node.parameter_count >= 1000 ? `${(node.parameter_count / 1000).toFixed(1)} K` : node.parameter_count}</span>
          </div>
        )}
        {node.utilization !== undefined && (
          <div className="flex justify-between">
            <span>Utilization</span>
            <span className="text-emerald-400">{(node.utilization * 100).toFixed(0)}%</span>
          </div>
        )}
        {node.created_step !== undefined && node.created_step > 0 && (
          <div className="flex justify-between">
            <span>Created At</span>
            <span className="text-slate-200">Step {node.created_step}</span>
          </div>
        )}
        {node.status && (
          <div className="flex justify-between mt-1">
            <span>Status</span>
            <span className={node.status === 'active' ? 'text-emerald-400' : 'text-slate-500'}>
              {node.status === 'initializing' ? 'Initializing...' : node.status.charAt(0).toUpperCase() + node.status.slice(1)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
