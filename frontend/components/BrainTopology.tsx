'use client';
import React, { useState, useMemo, useEffect, useRef } from 'react';
import { TopologySnapshot, NodeInfo } from '../types';
import { ModuleInspectionPanel } from './ModuleInspectionPanel';

interface BrainTopologyProps {
  topology: TopologySnapshot | null;
  isInitial?: boolean;
  uiState?: string;
}

interface SubNode {
  id: string; // e.g. "encoder_0"
  parent: NodeInfo | null;
  layerIdx: number;
  subIdx: number;
  totalInLayer: number;
  x: number;
  y: number;
  label?: string; 
  groupName: string;
  isExiting?: boolean;
}

interface Edge {
  id: string;
  sourceId: string;
  targetId: string;
  x1: number; y1: number;
  x2: number; y2: number;
  isExiting?: boolean;
  density?: number;
}

export const BrainTopology: React.FC<BrainTopologyProps> = ({ topology, isInitial, uiState = 'IDLE' }) => {
  const [selectedNode, setSelectedNode] = useState<{ node: NodeInfo, x: number, y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    let animationFrameId: number;
    
    const animatePaths = (timestamp: number) => {
      if (!svgRef.current) return;
      
      const timeSec = timestamp * 0.001;
      const edgeGroups = svgRef.current.querySelectorAll('.edge-group');
      
      const getBezierPoint = (t: number, p0: number, p1: number, p2: number, p3: number) => {
        const mt = 1 - t;
        return mt*mt*mt*p0 + 3*mt*mt*t*p1 + 3*mt*t*t*p2 + t*t*t*p3;
      };

      const isTraining = uiState === 'TRAINING';
      
      edgeGroups.forEach((group) => {
        const path = group.querySelector('.neural-connection-path');
        const pulse = group.querySelector('.pulse-circle');
        if (!path || !pulse) return;

        const x1 = parseFloat(path.getAttribute('data-x1') || '0');
        const y1 = parseFloat(path.getAttribute('data-y1') || '0');
        const x2 = parseFloat(path.getAttribute('data-x2') || '0');
        const y2 = parseFloat(path.getAttribute('data-y2') || '0');
        const idStr = path.getAttribute('data-edge-id') || '0';
        
        // Deterministic hash from string ID for stable animation
        let numericHash = 0;
        for (let i = 0; i < idStr.length; i++) {
          numericHash = (numericHash << 5) - numericHash + idStr.charCodeAt(i);
          numericHash |= 0;
        }
        const absHash = Math.abs(numericHash);

        const dx = Math.abs(x2 - x1) * 0.5;
        const dyBase = (Math.abs(y1 - y2) < 2) ? ((x1 + y1) % 2 === 0 ? 12 : -12) : 0;
        
        // Phase varies deterministically by edge ID hash
        const phase = absHash * 0.73;
        // Speed varies slightly
        const speed = 0.8 + (absHash % 4) * 0.2;
        // Subtle amplitude ~1.5px to 2px
        const waveDy = Math.sin(timeSec * speed + phase) * 1.8;
        
        const currentDy = dyBase + waveDy;
        path.setAttribute('d', `M ${x1} ${y1} C ${x1 + dx} ${y1 + currentDy}, ${x2 - dx} ${y2 + currentDy}, ${x2} ${y2}`);

        // Calculate traveling pulse
        const pulsePeriod = isTraining ? (3 + (absHash % 4)) : (6 + (absHash % 8)); 
        const pulseDuration = isTraining ? 1.2 : 2.0;
        const cycleTime = (timeSec + (absHash % 10)) % pulsePeriod;
        
        if (cycleTime < pulseDuration) {
          const t = cycleTime / pulseDuration; // 0 to 1
          const pulseX = getBezierPoint(t, x1, x1 + dx, x2 - dx, x2);
          const pulseY = getBezierPoint(t, y1, y1 + currentDy, y2 + currentDy, y2);
          
          const fadeInOut = t < 0.2 ? (t / 0.2) : t > 0.8 ? (1 - t) / 0.2 : 1;
          const maxOpacity = isTraining ? 0.9 : 0.4;
          
          pulse.setAttribute('cx', pulseX.toString());
          pulse.setAttribute('cy', pulseY.toString());
          pulse.setAttribute('opacity', (fadeInOut * maxOpacity).toString());
        } else {
          pulse.setAttribute('opacity', '0');
        }
      });

      animationFrameId = requestAnimationFrame(animatePaths);
    };

    animationFrameId = requestAnimationFrame(animatePaths);
    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  const { subNodes, edges } = useMemo(() => {
    const sNodes: SubNode[] = [];
    const sEdges: Edge[] = [];
    
    if (!topology || !topology.nodes) return { subNodes: sNodes, edges: sEdges };

    const sortedNodes = [...topology.nodes].sort((a, b) => (a.layer_index || 0) - (b.layer_index || 0));

    const availableWidth = 90.0;
    const spacingX = Math.max(3.5, availableWidth / Math.max(1, sortedNodes.length - 1));
    const startX = 50 - ((sortedNodes.length - 1) * spacingX) / 2;
    
    let currentX = startX;
    const nodeColumns: Record<string, SubNode[]> = {};

    const getVisualCount = (actualDim: number) => {
      if (actualDim === 1) return 1;
      if (actualDim <= 4) return actualDim;
      const scaled = Math.round(Math.sqrt(actualDim) * 1.5);
      return Math.max(1, Math.min(scaled, 14));
    };

    sortedNodes.forEach((node, colIndex) => {
      const actualDim = node.output_dim || node.input_dim || 1;
      const visualDim = getVisualCount(actualDim);
      
      const colNodes: SubNode[] = [];
      const spacingY = 4.0;
      const centerY = 50;
      
      for (let i = 0; i < visualDim; i++) {
        const y = centerY + (i - (visualDim - 1) / 2) * spacingY;
        const isBottom = i === visualDim - 1;
        
        let nodeLabel = undefined;
        if (isBottom) {
          nodeLabel = node.kind === 'input' || node.kind === 'output' ? 
            node.kind.charAt(0).toUpperCase() + node.kind.slice(1) : node.kind;
          if (actualDim > visualDim) {
            nodeLabel += `\n(${actualDim})`;
          }
        }
        
        const sn: SubNode = {
          id: `${node.id}_${i}`,
          parent: node as any,
          layerIdx: colIndex,
          subIdx: i,
          totalInLayer: visualDim,
          x: currentX,
          y,
          label: nodeLabel,
          groupName: node.role
        };
        sNodes.push(sn);
        colNodes.push(sn);
      }
      
      nodeColumns[node.id] = colNodes;
      currentX += spacingX;
    });

    const connectGroups = (group1: SubNode[], group2: SubNode[], density: number) => {
      if (density <= 0) return;
      group1.forEach((n1, i1) => {
        group2.forEach((n2, i2) => {
          let shouldConnect = false;
          if (density >= 0.8) {
             shouldConnect = true;
          } else {
             const hash = (i1 * 31 + i2 * 17) % 100;
             shouldConnect = hash < (density * 100);
          }
          if (shouldConnect) {
             const edgeId = `${n1.id}::${n2.id}`;
             sEdges.push({ id: edgeId, sourceId: n1.id, targetId: n2.id, x1: n1.x, y1: n1.y, x2: n2.x, y2: n2.y, density });
          }
        });
      });
    };

    if (topology.edges) {
      topology.edges.forEach(edge => {
        const g1 = nodeColumns[edge.source];
        const g2 = nodeColumns[edge.target];
        if (g1 && g2 && g1.length > 0 && g2.length > 0) {
           const density = edge.density !== undefined ? edge.density : 1.0;
           connectGroups(g1, g2, density);
        }
      });
    }

    const processedGroups = new Set<string>();
    sNodes.forEach(n => {
       if (!processedGroups.has(n.groupName)) {
           processedGroups.add(n.groupName);
       }
    });

    if (topology.actual_layer_count !== undefined) {
       const xCoords = new Set(sNodes.map(n => n.x));
       const abstractLayerCount = xCoords.size;
       if (abstractLayerCount !== topology.actual_layer_count) {
          console.error('TOPOLOGY_RENDER_MISMATCH', {
             topology_revision: topology.topology_revision,
             backend_layer_count: topology.actual_layer_count,
             abstract_layer_count: abstractLayerCount,
             actual_layer_count: abstractLayerCount,
             missing_node_ids: [],
             extra_node_ids: []
          });
       }
    }

    return { subNodes: sNodes, edges: sEdges };
  }, [topology]);

  // Handle smooth transitions for GROW and PRUNE
  const [renderedNodes, setRenderedNodes] = useState<SubNode[]>([]);
  const [renderedEdges, setRenderedEdges] = useState<Edge[]>([]);
  
  useEffect(() => {
    setRenderedNodes(prev => {
      const next = [...subNodes];
      const prevMap = new Map(prev.map(n => [n.id, n]));
      const nextMap = new Map(next.map(n => [n.id, n]));
      
      let changed = false;
      prev.forEach(n => {
        if (!nextMap.has(n.id) && !n.isExiting) {
          next.push({ ...n, isExiting: true });
          changed = true;
        }
      });
      return changed || prev.length !== subNodes.length ? next : prev;
    });

    setRenderedEdges(prev => {
      const next = [...edges];
      const prevMap = new Map(prev.map(e => [e.id, e]));
      const nextMap = new Map(next.map(e => [e.id, e]));
      
      let changed = false;
      prev.forEach(e => {
        if (!nextMap.has(e.id) && !e.isExiting) {
          next.push({ ...e, isExiting: true });
          changed = true;
        }
      });
      return changed || prev.length !== edges.length ? next : prev;
    });
    
  }, [subNodes, edges]);

  // Cleanup exiting nodes/edges after animation completes
  useEffect(() => {
    const hasExiting = renderedNodes.some(n => n.isExiting) || renderedEdges.some(e => e.isExiting);
    if (hasExiting) {
      const timer = setTimeout(() => {
        setRenderedNodes(prev => prev.filter(n => !n.isExiting));
        setRenderedEdges(prev => prev.filter(e => !e.isExiting));
      }, 1000); // 1s fade out duration
      return () => clearTimeout(timer);
    }
  }, [renderedNodes, renderedEdges]);

  // Handle node click (open inspection panel)
  const handleNodeClick = (e: React.MouseEvent, node: NodeInfo | null) => {
    if (!node) return;
    const rect = (e.currentTarget as Element).getBoundingClientRect();
    const parentRect = (e.currentTarget.parentElement as Element).getBoundingClientRect();
    
    setSelectedNode({
      node,
      x: rect.left - parentRect.left + rect.width / 2,
      y: rect.top - parentRect.top + rect.height / 2
    });
  };

  const createPath = (x1: number, y1: number, x2: number, y2: number) => {
    // Wavy cubic bezier for organic feel
    const dx = Math.abs(x2 - x1) * 0.5;
    // Add a slight vertical bowing if y1 and y2 are perfectly flat to prevent straight lines
    const dy = (Math.abs(y1 - y2) < 2) ? ((x1 + y1) % 2 === 0 ? 12 : -12) : 0;
    return `M ${x1} ${y1} C ${x1 + dx} ${y1 + dy}, ${x2 - dx} ${y2 + dy}, ${x2} ${y2}`;
  };

  return (
    <div className="w-full h-full relative flex items-center justify-center" onClick={() => setSelectedNode(null)}>
      <svg ref={svgRef} className={`pointer-events-none transition-all duration-1000 ease-in-out ${isInitial ? 'w-[55%] max-w-[800px]' : 'w-[75%] max-w-[1200px]'}`} viewBox="0 150 1000 300" preserveAspectRatio="xMidYMid meet" style={{ overflow: 'visible' }}>
        
        {/* Region Labels removed per instructions */}
        {/* Draw Edges */}
        {renderedEdges.map((edge) => {
          const scaleX = (val: number) => (val / 100) * 1000;
          const scaleY = (val: number) => (val / 100) * 600;
          const path = createPath(scaleX(edge.x1), scaleY(edge.y1), scaleX(edge.x2), scaleY(edge.y2));
          // Make pseudo connections dimmer, module connections slightly brighter
          const isPseudo = edge.sourceId.startsWith('input') || edge.targetId.startsWith('output');
          const isModuleEdge = edge.sourceId.startsWith('module') || edge.targetId.startsWith('module');
          
          const baseColor = isPseudo ? "stroke-slate-700/30" : isModuleEdge ? "stroke-emerald-800/40" : "stroke-blue-900/40";
          const pulseColor = isPseudo ? "stroke-slate-500/80" : isModuleEdge ? "stroke-emerald-400/80" : "stroke-blue-400/80";
          
          // Generate a numeric hash for CSS animation delays if still needed temporarily
          let numericHash = 0;
          for (let i = 0; i < edge.id.length; i++) {
            numericHash = (numericHash << 5) - numericHash + edge.id.charCodeAt(i);
            numericHash |= 0;
          }
          const absHash = Math.abs(numericHash);
          const animDuration = `${3 + (absHash % 3)}s`;
          const animDelay = `${(absHash % 17) * -0.7}s`;

          // Calculate opacity based on exact mathematical density from backend
          let edgeOpacity = 100;
          let extraStroke = "";
          if (edge.density !== undefined) {
             // scale opacity so density 0 = 0 opacity, density 1.0 = full opacity
             // clamp between 10% and 100% for non-zero connections so they remain somewhat visible
             const scaledOp = Math.max(10, Math.min(100, Math.round(edge.density * 100)));
             edgeOpacity = scaledOp;
             
             if (edge.density > 0.8) {
               extraStroke = "stroke-[1.8px]"; // make strong connections visually thicker
             }
          }

          return (
            <g key={edge.id} className={`edge-group transition-all duration-1000 ${edge.isExiting ? 'opacity-0' : 'animate-in fade-in duration-1000'}`} style={{ opacity: edge.isExiting ? 0 : (edgeOpacity / 100) }}>
              {/* Base continuous wavy line */}
              <path 
                d={path} 
                fill="none" 
                className={`neural-connection-path ${baseColor} ${extraStroke} transition-all duration-700 ease-in-out`} 
                strokeWidth={extraStroke ? undefined : "1.2"} 
                data-x1={scaleX(edge.x1)} data-y1={scaleY(edge.y1)}
                data-x2={scaleX(edge.x2)} data-y2={scaleY(edge.y2)}
                data-edge-id={edge.id}
              />
              {/* Math-driven traveling pulse overlay */}
              <circle
                r={isPseudo ? "1.5" : "2"}
                className={`pulse-circle ${pulseColor.replace('stroke', 'fill')}`}
                opacity="0"
                style={{ filter: 'drop-shadow(0 0 3px currentColor)' }}
              />
            </g>
          );
        })}

        {/* Draw Nodes */}
        {renderedNodes.map((sn) => {
          const isInputOut = !sn.parent;
          const isModule = sn.parent?.role?.startsWith('module_');
          const isNew = sn.parent?.created_step && sn.parent.created_step > 0;
          const isHovered = selectedNode?.node.id === sn.parent?.id;

          let colorClass = 'text-blue-400 stroke-blue-500 fill-blue-950/80';
          let glowClass = 'drop-shadow-[0_0_4px_rgba(59,130,246,0.3)]';
          let innerClass = 'fill-blue-300';

          if (isInputOut) {
             colorClass = 'text-slate-400 stroke-slate-500 fill-[#040b16]';
             glowClass = '';
             innerClass = 'fill-slate-500';
          } else if (sn.parent?.role === 'router' || sn.parent?.role === 'head') {
            colorClass = 'text-purple-400 stroke-purple-500 fill-purple-950/80';
            glowClass = 'drop-shadow-[0_0_4px_rgba(168,85,247,0.3)]';
            innerClass = 'fill-purple-300';
          } else if (isModule) {
            if (isNew && sn.parent?.status === 'initializing') {
              colorClass = 'text-amber-400 stroke-amber-500 fill-amber-950/80 stroke-dashed';
              glowClass = 'drop-shadow-[0_0_8px_rgba(245,158,11,0.5)] animate-pulse';
              innerClass = 'fill-amber-400';
            } else {
              colorClass = 'text-emerald-400 stroke-emerald-500 fill-emerald-950/80';
              glowClass = 'drop-shadow-[0_0_4px_rgba(16,185,129,0.3)]';
              innerClass = 'fill-emerald-300';
            }
          }

          if (isHovered) {
             glowClass += ' drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]';
          }

          const scaleX = (val: number) => (val / 100) * 1000;
          const scaleY = (val: number) => (val / 100) * 600;
          const animDelay = `${(sn.subIdx * 0.3) + (sn.layerIdx * 0.1)}s`;

          return (
            <g 
              key={sn.id} 
              className={`transition-all duration-1000 ease-in-out origin-center ${glowClass} ${!isInputOut && !sn.isExiting ? 'animate-pulse-glow' : ''} ${sn.parent ? 'cursor-pointer pointer-events-auto hover:scale-125' : ''} ${sn.isExiting ? 'opacity-0 scale-50' : 'opacity-100 animate-in fade-in zoom-in duration-1000'}`}
              style={{ animationDelay: animDelay, transformOrigin: `${scaleX(sn.x)}px ${scaleY(sn.y)}px` }}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => handleNodeClick(e, sn.parent)}
            >
              {/* Very tiny dot neurons */}
              <circle cx={scaleX(sn.x)} cy={scaleY(sn.y)} r="6" className={`${colorClass} stroke-[1.2px] transition-all duration-500`} />
              {/* Inner core */}
              <circle cx={scaleX(sn.x)} cy={scaleY(sn.y)} r="1.5" className={`${innerClass} pointer-events-none opacity-90`} />
              
              {/* Optional Label */}
              {sn.label && (
                <text 
                  x={scaleX(sn.x)} 
                  y={scaleY(sn.y) + 20} 
                  textAnchor="middle" 
                  className={`text-[9px] tracking-[0.1em] font-mono pointer-events-none transition-all duration-500 uppercase ${
                     isNew && sn.parent?.status === 'initializing' ? 'fill-amber-400 font-bold' : 'fill-slate-500/70'
                  }`}
                  style={{ animation: 'none' }}
                >
                  {sn.label.split('\n').map((line, i) => (
                    <tspan key={i} x={scaleX(sn.x)} dy={i === 0 ? 0 : "1.2em"}>{line}</tspan>
                  ))}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* HTML Overlay for interactions */}
      <div className="absolute inset-0 pointer-events-none">
        {selectedNode && (
          <ModuleInspectionPanel 
            node={selectedNode.node} 
            x={selectedNode.x} 
            y={selectedNode.y} 
            onClose={() => setSelectedNode(null)} 
          />
        )}
      </div>

    </div>
  );
};
