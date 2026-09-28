'use client';
import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { TopologySnapshot, ActualLayer } from '../types';

interface BrainTopologyProps {
  topology: TopologySnapshot | null;
  isInitial?: boolean;
  uiState?: string;
}

// B64 Decoders
function decodeFloat32(b64: string): Float32Array {
  if (!b64) return new Float32Array(0);
  const bin = window.atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function decodeUint8(b64: string): Uint8Array {
  if (!b64) return new Uint8Array(0);
  const bin = window.atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

interface NodeColumn {
  id: string;
  x: number;
  label: string; 
  subLabels: string[]; 
  dimensions: string; 
  actualDim: number;
  isSparse: boolean;
  colorGroup: 'blue' | 'purple' | 'emerald' | 'slate';
  weight_b64?: string;
  mask_b64?: string;
  nodesY: number[];
}

export const ActualArchitectureView: React.FC<BrainTopologyProps> = ({ topology, isInitial, uiState = 'IDLE' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [prevTopology, setPrevTopology] = useState<any>(null);
  const [transitionStart, setTransitionStart] = useState<number>(0);

  useEffect(() => {
    // Trigger transition when revision changes
    if (topology && prevTopology && topology.topology_revision !== prevTopology.topology_revision) {
      setPrevTopology(topology);
      setTransitionStart(Date.now());
    } else if (topology && !prevTopology) {
      setPrevTopology(topology);
    }
  }, [topology, prevTopology]);

  const requestRef = useRef<number>();
  
  // Animation loop to force redraws during the transition period
  useEffect(() => {
    const animate = () => {
      const now = Date.now();
      if (now - transitionStart < 1600) {
        // Trigger a re-render
        setZoom(z => z); // Dummy state update to trigger render loop
        requestRef.current = requestAnimationFrame(animate);
      }
    };
    if (transitionStart > 0) {
      requestRef.current = requestAnimationFrame(animate);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [transitionStart]);
  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  const { cols, width, height, bounds } = useMemo(() => {
    const cList: NodeColumn[] = [];
    if (!topology || !topology.nodes) return { cols: cList, width: 1000, height: 600, bounds: { minX: 0, maxX: 1000, minY: 0, maxY: 600 } };

    const sortedNodes = [...topology.nodes].sort((a, b) => (a.layer_index || 0) - (b.layer_index || 0));

    const startX = 100;
    const spacingX = 160; 
    let currentX = startX;

    const buildColumn = (node: any, x: number): NodeColumn => {
      let actualDim = node.output_dim || node.input_dim || 1;
      
      let colorGroup: 'blue' | 'purple' | 'emerald' | 'slate' = 'slate';
      if (node.role === 'encoder') colorGroup = 'blue';
      else if (node.role === 'router' || node.role === 'head') colorGroup = 'purple';
      else if (node.role?.startsWith('module_')) colorGroup = 'emerald';

      let label = node.kind;
      if (node.kind === 'input' || node.kind === 'output') {
         label = node.kind.charAt(0).toUpperCase() + node.kind.slice(1);
      }

      const maxColHeight = 500;
      const minNodeSpacing = 3;
      const desiredSpacing = 12;
      const nodeSpacingY = Math.max(minNodeSpacing, Math.min(desiredSpacing, maxColHeight / Math.max(1, actualDim)));

      const nodesY = new Float32Array(actualDim);
      for (let i=0; i<actualDim; i++) {
        nodesY[i] = 300 + (i - (actualDim-1)/2) * nodeSpacingY;
      }
      return {
        id: node.id, x, label, subLabels: [],
        dimensions: `${actualDim} units`, actualDim,
        isSparse: !!node.sparse?.enabled,
        colorGroup, weight_b64: node.weight_b64, mask_b64: node.sparse?.mask_b64,
        nodesY: Array.from(nodesY)
      };
    };

    sortedNodes.forEach(node => {
       cList.push(buildColumn(node, currentX));
       currentX += spacingX;
    });

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    cList.forEach(c => {
       if (c.x < minX) minX = c.x;
       if (c.x > maxX) maxX = c.x;
       
       if (c.nodesY.length > 0) {
           const top = c.nodesY[0];
           const bottom = c.nodesY[c.nodesY.length - 1];
           if (top < minY) minY = top;
           if (bottom > maxY) maxY = bottom;
       }
    });

    if (minX === Infinity) {
      minX = 0; maxX = 1000; minY = 0; maxY = 600;
    }

    minX -= 50;
    maxX += 150;
    minY -= 80;
    maxY += 100;

    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;

    if (topology.actual_layer_count !== undefined) {
       const actualLayerCount = cList.length;
       if (actualLayerCount !== topology.actual_layer_count) {
          console.error('TOPOLOGY_RENDER_MISMATCH', {
             topology_revision: topology.topology_revision,
             backend_layer_count: topology.actual_layer_count,
             actual_layer_count: actualLayerCount,
             missing_node_ids: [],
             extra_node_ids: []
          });
       }
    }

    return { 
      cols: cList, 
      width: Math.max(contentWidth, currentX + 150), 
      height: Math.max(600, contentHeight), 
      bounds: { minX, maxX, minY, maxY } 
    };
  }, [topology]);

  // CANVAS RENDERER
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // HiDPI support
    const dpr = window.devicePixelRatio || 1;
    // Set actual canvas DOM size matches width/height internally
    // We make the canvas large enough to hold everything from 0 to width
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    // Draw Edges based on authoritative topology.edges
    if (topology && topology.edges) {
      topology.edges.forEach(edge => {
        const sourceCol = cols.find(c => c.id === edge.source);
        const targetCol = cols.find(c => c.id === edge.target);
        
        if (!sourceCol || !targetCol) return;

        if (!targetCol.weight_b64) {
          // Virtual connection (e.g. router to module input)
          ctx.beginPath();
          ctx.strokeStyle = "rgba(100, 150, 200, 0.05)";
          ctx.lineWidth = 1.0;
          const sY = sourceCol.nodesY[Math.floor(sourceCol.actualDim/2)];
          const tY = targetCol.nodesY[Math.floor(targetCol.actualDim/2)];
          ctx.moveTo(sourceCol.x, sY);
          ctx.lineTo(targetCol.x, tY);
          ctx.stroke();
          return;
        }

        const in_dim = sourceCol.actualDim;
        const out_dim = targetCol.actualDim;
        const W = decodeFloat32(targetCol.weight_b64);
        const M = targetCol.mask_b64 ? decodeUint8(targetCol.mask_b64) : null;
        
        if (W.length !== in_dim * out_dim) return; // Safety check
        
        const rewireDiff: any = (topology as any).rewire_diff;

        // To optimize Canvas 2D drawing of 4096+ lines, we bucket by alpha/color
        ctx.lineWidth = 0.5;
        const now = Date.now();
        const transitionProgress = Math.min(1.0, (now - transitionStart) / 1500.0); // 1.5s transition
        
        for (let o=0; o<out_dim; o++) {
          for (let i=0; i<in_dim; i++) {
            const idx = o * in_dim + i;
            
            const m_curr = M ? M[idx] : 1;
            
            const isAdded = rewireDiff?.added_edges?.some((e: any) => e.layer_id === targetCol.id && e.flat_index === idx);
            const isRemoved = rewireDiff?.removed_edges?.some((e: any) => e.layer_id === targetCol.id && e.flat_index === idx);
            
            if (m_curr === 0 && !isRemoved) continue; // structural connection does NOT exist
            
            let alphaMod = 1.0;
            let edgeColorOverride = null;
            let drawEdge = true;
            
            if (isAdded) {
                // ADDED EDGE
                if (transitionProgress < 1.0) {
                    edgeColorOverride = `rgba(52, 211, 153, ${Math.max(0.2, transitionProgress)})`; // Flash green
                    ctx.lineWidth = 1.5 - transitionProgress; // start thick, go normal
                }
            } else if (isRemoved) {
                // REMOVED EDGE
                if (transitionProgress < 1.0) {
                    edgeColorOverride = `rgba(244, 63, 94, ${1.0 - transitionProgress})`; // Fade red
                    ctx.lineWidth = 1.5 - transitionProgress;
                } else {
                    drawEdge = false;
                }
            } else {
                ctx.lineWidth = 0.5;
            }
            
            if (!drawEdge) continue;

            const w = W[idx];
            // "If mask == 1 the connection exists even if weight ~ 0"
            const alpha = Math.max(0.05, Math.min(Math.abs(w) * 2.0, 0.5)) * alphaMod;
            const isPos = w > 0;
            
            ctx.beginPath();
            
            if (edgeColorOverride) {
                ctx.strokeStyle = edgeColorOverride;
            } else {
                ctx.strokeStyle = isPos ? `rgba(96, 165, 250, ${alpha})` : `rgba(248, 113, 113, ${alpha})`;
                if (targetCol.colorGroup === 'emerald') {
                   ctx.strokeStyle = isPos ? `rgba(52, 211, 153, ${alpha})` : `rgba(244, 63, 94, ${alpha})`;
                }
            }

            ctx.moveTo(sourceCol.x, sourceCol.nodesY[i]);
            ctx.lineTo(targetCol.x, targetCol.nodesY[o]);
            ctx.stroke();
          }
        }
      });
    }

    // Draw Nodes
    cols.forEach(c => {
      ctx.fillStyle = c.colorGroup === 'emerald' ? '#064e3b' : c.colorGroup === 'blue' ? '#1e3a8a' : c.colorGroup === 'purple' ? '#4c1d95' : '#1e293b';
      ctx.strokeStyle = c.colorGroup === 'emerald' ? '#10b981' : c.colorGroup === 'blue' ? '#3b82f6' : c.colorGroup === 'purple' ? '#8b5cf6' : '#64748b';
      ctx.lineWidth = 1.0;
      
      ctx.beginPath();
      for (let i=0; i<c.actualDim; i++) {
        ctx.moveTo(c.x + 3, c.nodesY[i]);
        ctx.arc(c.x, c.nodesY[i], 3, 0, 2 * Math.PI);
      }
      ctx.fill();
      ctx.stroke();
    });
  }, [cols, width, height]);

  const fitToScreen = useCallback(() => {
    if (!containerRef.current) return;
    const { clientWidth, clientHeight } = containerRef.current;
    if (clientWidth === 0 || clientHeight === 0) return; // Wait for layout
    
    const { minX, maxX, minY, maxY } = bounds;
    const contentWidth = Math.max(1, maxX - minX);
    const contentHeight = Math.max(1, maxY - minY);
    
    // Exact padding per user instructions
    const paddingX = 80;
    const paddingY = 80;
    const usableWidth = Math.max(10, clientWidth - paddingX);
    const usableHeight = Math.max(10, clientHeight - paddingY);
    
    const scaleX = usableWidth / contentWidth;
    const scaleY = usableHeight / contentHeight;
    
    const fitScale = Math.min(scaleX, scaleY);
    
    // Avoid making it microscopic unless absolutely necessary to fit
    const scale = Math.max(0.1, Math.min(fitScale, 3.0));
    
    // Center bounding box precisely within the client rect
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    
    const px = (clientWidth / 2) - (cx * scale);
    const py = (clientHeight / 2) - (cy * scale);
    
    setZoom(scale);
    setPan({ x: px, y: py });
  }, [bounds]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Use ResizeObserver to automatically refit when dashboard changes height (e.g. READY mode)
    const observer = new ResizeObserver(() => {
      fitToScreen();
    });
    
    observer.observe(el);
    
    // Also trigger immediately just in case
    fitToScreen();

    return () => observer.disconnect();
  }, [fitToScreen]);

  const handleWheelNative = (e: WheelEvent) => {
    e.preventDefault();
    const zoomSensitivity = 0.0015;
    const delta = -e.deltaY * zoomSensitivity;
    setZoom((prevZoom) => {
      const newZoom = Math.max(0.1, Math.min(prevZoom * (1 + delta), 4.0));
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        setPan(prevPan => {
          const worldX = (mouseX - prevPan.x) / prevZoom;
          const worldY = (mouseY - prevPan.y) / prevZoom;
          return {
            x: mouseX - worldX * newZoom,
            y: mouseY - worldY * newZoom
          };
        });
      }
      return newZoom;
    });
  };

  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.addEventListener('wheel', handleWheelNative, { passive: false });
      return () => el.removeEventListener('wheel', handleWheelNative);
    }
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    setPan(p => ({ x: p.x + dx, y: p.y + dy }));
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  return (
    <div 
      ref={containerRef}
      className="w-full h-full relative overflow-hidden bg-transparent"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >

      <div 
        style={{ 
          width: width, 
          height: height, 
          position: 'absolute',
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: '0 0',
          cursor: isDragging.current ? 'grabbing' : 'grab'
        }}
      >
        <canvas 
          ref={canvasRef}
          style={{ width: `${width}px`, height: `${height}px`, position: 'absolute', top: 0, left: 0 }}
        />

        {/* Labels Overlay */}
        {cols.map(c => {
          let textColor = "text-slate-400";
          if (c.colorGroup === 'blue') textColor = "text-blue-400";
          else if (c.colorGroup === 'purple') textColor = "text-purple-400";
          else if (c.colorGroup === 'emerald') textColor = "text-emerald-400";

          const minY = c.nodesY[0];
          const maxY = c.nodesY[c.nodesY.length-1];

          return (
            <div key={`labels_${c.id}`} className="pointer-events-none">
              <div 
                className={`absolute flex flex-col items-center justify-start w-32 -ml-16 font-mono text-[10px] ${textColor} opacity-80 text-center`}
                style={{ left: c.x, top: maxY + 20 }}
              >
                <div className="font-bold text-white">{c.label}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
