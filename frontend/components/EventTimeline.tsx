'use client';
import React from 'react';
import { EventData } from '../types';

interface EventTimelineProps {
  events: EventData[];
}

export const EventTimeline: React.FC<EventTimelineProps> = ({ events }) => {
  return (
    <div className="w-full h-full bg-[#060a14] p-5 pt-3 flex flex-col font-mono relative overflow-y-auto overflow-x-hidden">
      <h2 className="text-[10px] text-slate-500 tracking-[0.2em] uppercase mb-4 pb-3 flex-none">
        Evolution Trace
      </h2>

      <div className="flex-1 flex flex-col gap-5 overflow-y-auto pr-1">
        {[...events].reverse().filter(e => !['LEVEL_HISTORY_UPDATED', 'TOPOLOGY_HISTORY_ENTRY', 'TOPOLOGY_SNAPSHOT', 'ENVIRONMENT_STATE_UPDATE'].includes(e.event_type)).reduce((acc, evt) => {
          if (evt.event_type === "ENVIRONMENT_STATE_UPDATE") {
            const last = acc[acc.length - 1];
            if (last && last.event_type === "ENVIRONMENT_STATE_UPDATE") {
              last.count = (last.count || 1) + 1;
              last.timestamp = evt.timestamp;
              return acc;
            }
            acc.push({ ...evt, count: 1 });
            return acc;
          }
          acc.push({ ...evt, count: 1 });
          return acc;
        }, [] as any[]).map((evt: any) => {
          let title = evt.event_type;
          let color = "text-slate-300";
          let desc = "";

          if (evt.event_type === "ENVIRONMENT_STATE_UPDATE") {
            title = evt.count > 1 ? `WORLD UPDATE × ${evt.count}` : `WORLD UPDATE`;
            color = "text-slate-500";
          } else if (evt.event_type === "RUN_STARTED") {
            title = "RUN STARTED";
          } else if (evt.event_type === "LEVEL_STARTED") {
            title = `LEVEL ${evt.payload?.level_id || evt.step} STARTED`;
            color = "text-blue-400";
          } else if (evt.event_type === "WORLD_READY") {
            title = "WORLD READY";
            color = "text-blue-400";
          } else if (evt.event_type === "METRICS_UPDATE") {
            title = "PERFORMANCE UPDATED";
            color = "text-emerald-400";
          } else if (evt.event_type === "DECISION_MADE") {
            title = `POLICY → ${evt.payload?.action}`;
            color = "text-amber-400";
          } else if (evt.event_type === "MUTATION_PROPOSED" || evt.event_type === "MUTATION_VERIFYING") {
            title = `${evt.payload?.action || 'MUTATION'} · VERIFYING`;
            color = "text-blue-400";
          } else if (evt.event_type === "OOD_SHIFT_DETECTED") {
            title = "OOD SHIFT DETECTED";
            color = "text-amber-500";
          } else if (evt.event_type === "ROLLBACK") {
            title = `${evt.payload?.action || 'MUTATION'} · ROLLED BACK`;
            desc = "TOPOLOGY RESTORED";
            color = "text-amber-500";
            if (evt.payload?.record?.rewire_diff) {
                const diff = evt.payload.record.rewire_diff;
                if (diff.added_edge_count > 0 || diff.removed_edge_count > 0) {
                    desc = `STRUCTURAL DIFF REVERSED\n+ ${diff.added_edge_count} EDGES\n- ${diff.removed_edge_count} EDGES`;
                }
            }
          } else if (evt.event_type === "MUTATION_ACCEPTED") {
            title = `${evt.payload?.action || 'MUTATION'} · ACCEPTED`;
            color = "text-emerald-400";
            if (evt.payload?.action === 'REWIRE' && evt.payload?.record?.rewire_diff) {
                const diff = evt.payload.record.rewire_diff;
                if (diff.added_edge_count === 0 && diff.removed_edge_count === 0) {
                    desc = "NO STRUCTURAL CHANGE";
                } else {
                    desc = `TOPOLOGY DIFF\n+ ${diff.added_edge_count} EDGES\n- ${diff.removed_edge_count} EDGES`;
                }
            } else {
                desc = "TOPOLOGY CHANGED";
            }
          } else if (evt.event_type === "PLASTICITY_PROBE") {
            title = "PLASTICITY PROBE";
            color = "text-purple-400";
          } else if (evt.event_type === "CONVERGENCE") {
            title = "TASK COMPLETED";
            color = "text-emerald-400";
          }

          return (
            <div key={`${evt.event_id}-${evt.count}`} className="flex flex-col gap-1">
              <span className="text-[10px] text-slate-500">{evt.timestamp.slice(11, 19) || "00:00:00"}</span>
              <span className={`text-xs ${color}`}>{title}</span>
              {desc && <span className="text-[10px] text-slate-400 whitespace-pre-line">{desc}</span>}
            </div>
          )
        })}
      </div>
    </div>
  );
};
