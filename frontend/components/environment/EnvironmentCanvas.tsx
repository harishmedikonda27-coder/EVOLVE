'use client';
import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';

// ─── Coordinate Transformation ──────────────────────────────────────
// Backend uses [row, col] where row=0 is top, col=0 is left.
// SVG viewBox is set to [0, 0, cols, rows] so that 1 unit = 1 cell.
// Cell center at grid position (r, c) maps to SVG point (c + 0.5, r + 0.5).

export function gridToSvg(row: number, col: number): [number, number] {
  return [col + 0.5, row + 0.5];
}

export function clampToGrid(row: number, col: number, rows: number, cols: number): [number, number] {
  return [
    Math.max(0, Math.min(row, rows - 1)),
    Math.max(0, Math.min(col, cols - 1)),
  ];
}

// ─── Agent Interpolation Hook ───────────────────────────────────────
export function useAgentInterpolation(
  targetPos: [number, number] | null,
  rows: number,
  cols: number,
  duration: number = 120
): [number, number] | null {
  const [renderPos, setRenderPos] = useState<[number, number] | null>(null);
  const animRef = useRef<number | null>(null);
  const currentRef = useRef<[number, number] | null>(null);

  useEffect(() => {
    if (!targetPos) return;

    const [tr, tc] = clampToGrid(targetPos[0], targetPos[1], rows, cols);

    if (!currentRef.current) {
      currentRef.current = [tr, tc];
      setRenderPos([tr, tc]);
      return;
    }

    const startR = currentRef.current[0];
    const startC = currentRef.current[1];

    if (startR === tr && startC === tc) return;

    // Cancel any previous animation
    if (animRef.current) cancelAnimationFrame(animRef.current);

    let startTime: number | null = null;

    const animate = (ts: number) => {
      if (!startTime) startTime = ts;
      const progress = Math.min((ts - startTime) / duration, 1.0);
      const r = startR + (tr - startR) * progress;
      const c = startC + (tc - startC) * progress;
      const clamped = clampToGrid(r, c, rows, cols);
      currentRef.current = clamped;
      setRenderPos(clamped);
      if (progress < 1.0) {
        animRef.current = requestAnimationFrame(animate);
      } else {
        animRef.current = null;
      }
    };

    animRef.current = requestAnimationFrame(animate);

    return () => {
      if (animRef.current) {
        cancelAnimationFrame(animRef.current);
        animRef.current = null;
      }
    };
  }, [targetPos, rows, cols, duration]);

  return renderPos;
}

// ─── SVG Filter Definitions ─────────────────────────────────────────
export const SvgDefs: React.FC = () => (
  <defs>
    {/* Agent glow */}
    <filter id="glow-agent" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="0.15" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
    {/* Key glow */}
    <filter id="glow-key" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="0.12" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
    {/* Target glow */}
    <filter id="glow-target" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="0.2" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
    {/* Cue glow */}
    <filter id="glow-cue" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="0.18" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
    {/* Hazard glow */}
    <filter id="glow-hazard" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="0.1" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>
);

// ─── Grid Background ────────────────────────────────────────────────
interface GridProps {
  rows: number;
  cols: number;
}

export const GridBackground: React.FC<GridProps> = ({ rows, cols }) => (
  <g className="grid-lines">
    {/* Vertical lines */}
    {Array.from({ length: cols + 1 }).map((_, i) => (
      <line
        key={`v-${i}`}
        x1={i} y1={0} x2={i} y2={rows}
        stroke="#1a2744"
        strokeWidth="0.02"
      />
    ))}
    {/* Horizontal lines */}
    {Array.from({ length: rows + 1 }).map((_, i) => (
      <line
        key={`h-${i}`}
        x1={0} y1={i} x2={cols} y2={i}
        stroke="#1a2744"
        strokeWidth="0.02"
      />
    ))}
  </g>
);

// ─── Obstacle Renderer ──────────────────────────────────────────────
interface ObstaclesProps {
  obstacles: [number, number][];
}

export const Obstacles: React.FC<ObstaclesProps> = ({ obstacles }) => (
  <g className="obstacles">
    {obstacles.map(([r, c], i) => (
      <rect
        key={`obs-${i}`}
        x={c + 0.08}
        y={r + 0.08}
        width={0.84}
        height={0.84}
        rx={0.06}
        fill="#374151"
        stroke="#4b5563"
        strokeWidth="0.03"
      />
    ))}
  </g>
);

// ─── Hazard Renderer ────────────────────────────────────────────────
interface HazardsProps {
  hazards: [number, number][];
}

export const Hazards: React.FC<HazardsProps> = ({ hazards }) => (
  <g className="hazards" filter="url(#glow-hazard)">
    {hazards.map(([r, c], i) => {
      const cx = c + 0.5;
      const cy = r + 0.5;
      return (
        <g key={`haz-${i}`}>
          <rect x={c + 0.1} y={r + 0.1} width={0.8} height={0.8} rx={0.05} fill="#7f1d1d" fillOpacity={0.4} />
          {/* X mark */}
          <line x1={cx - 0.2} y1={cy - 0.2} x2={cx + 0.2} y2={cy + 0.2} stroke="#ef4444" strokeWidth="0.06" strokeLinecap="round" />
          <line x1={cx + 0.2} y1={cy - 0.2} x2={cx - 0.2} y2={cy + 0.2} stroke="#ef4444" strokeWidth="0.06" strokeLinecap="round" />
        </g>
      );
    })}
  </g>
);

// ─── Agent Renderer ─────────────────────────────────────────────────
interface AgentProps {
  pos: [number, number];
  color?: string;
}

export const Agent: React.FC<AgentProps> = ({ pos, color = '#3b82f6' }) => {
  const [sx, sy] = gridToSvg(pos[0], pos[1]);
  const size = 0.28;
  // Upward-pointing triangle
  const points = `${sx},${sy - size} ${sx - size * 0.85},${sy + size * 0.6} ${sx + size * 0.85},${sy + size * 0.6}`;
  return (
    <g filter="url(#glow-agent)">
      <polygon
        points={points}
        fill={color}
        stroke="white"
        strokeWidth="0.04"
        strokeLinejoin="round"
      />
      {/* Inner highlight */}
      <circle cx={sx} cy={sy + 0.02} r={0.08} fill="white" opacity={0.6} />
    </g>
  );
};

// ─── Path Trail Renderer ────────────────────────────────────────────
interface PathTrailProps {
  positions: [number, number][];
  color?: string;
}

export const PathTrail: React.FC<PathTrailProps> = ({ positions, color = '#3b82f6' }) => {
  if (positions.length < 2) return null;
  return (
    <g className="path-trail">
      {positions.map((pos, i) => {
        if (i === positions.length - 1) return null;
        const opacity = 0.15 + (i / positions.length) * 0.4;
        const [sx, sy] = gridToSvg(pos[0], pos[1]);
        return (
          <circle
            key={`trail-${i}`}
            cx={sx}
            cy={sy}
            r={0.08 + (i / positions.length) * 0.06}
            fill={color}
            opacity={opacity}
          />
        );
      })}
    </g>
  );
};

// ─── Main Canvas Container ──────────────────────────────────────────
interface EnvironmentCanvasProps {
  rows: number;
  cols: number;
  title: string;
  subtitle?: React.ReactNode;
  step: number;
  hudContent?: React.ReactNode;
  children: React.ReactNode;
}

export const EnvironmentCanvas: React.FC<EnvironmentCanvasProps> = ({
  rows,
  cols,
  title,
  subtitle,
  step,
  hudContent,
  children,
}) => {
  return (
    <div className="w-full h-full flex flex-col bg-[#060a14] relative overflow-hidden">
      {/* HUD Header */}
      <div className="flex items-center justify-between px-5 py-2 flex-none border-b border-[#1a2744]/60">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono font-semibold text-slate-200 tracking-wide">{title}</span>
          {subtitle && (
            <span className="text-[10px] font-mono text-slate-500 tracking-wider">{subtitle}</span>
          )}
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500 uppercase tracking-widest">
          <span>Step {step}</span>
          {hudContent}
        </div>
      </div>

      {/* SVG Environment */}
      <div className="flex-1 flex items-center justify-center p-3 min-h-0">
        <svg
          viewBox={`-0.1 -0.1 ${cols + 0.2} ${rows + 0.2}`}
          className="w-full h-full max-w-full max-h-full"
          preserveAspectRatio="xMidYMid meet"
          style={{ overflow: 'visible' }}
        >
          <SvgDefs />

          {/* Deep background */}
          <rect x={-0.1} y={-0.1} width={cols + 0.2} height={rows + 0.2} fill="#080e1a" rx={0.15} />

          {/* Grid border */}
          <rect x={0} y={0} width={cols} height={rows} fill="none" stroke="#1e3a5f" strokeWidth="0.04" rx={0.08} />

          {/* Grid lines */}
          <GridBackground rows={rows} cols={cols} />

          {/* Environment-specific content */}
          {children}
        </svg>
      </div>
    </div>
  );
};
