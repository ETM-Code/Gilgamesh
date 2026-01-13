import { useEffect, useRef, useMemo, useCallback } from 'react';
import type { NeuronState, NetworkTopology } from '../lib/protocol';

interface NetworkCanvasProps {
  neurons: NeuronState[];
  topology: NetworkTopology | null;
  width: number;
  height: number;
}

interface NeuronPosition {
  x: number;
  y: number;
  layer: number;
  index: number;
}

// Layer colors - vibrant gradients
const LAYER_COLORS = [
  { base: '#3b82f6', glow: '#60a5fa', name: 'Input' },    // Blue
  { base: '#a855f7', glow: '#c084fc', name: 'Hidden' },   // Purple
  { base: '#f97316', glow: '#fb923c', name: 'Output' },   // Orange
];

export function NetworkCanvas({ neurons, topology, width, height }: NetworkCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const pulseTimeRef = useRef<number>(0);
  const prevNeuronsRef = useRef<Map<string, NeuronState>>(new Map());

  // Track active pulses for animation
  const pulsesRef = useRef<Array<{
    fromX: number;
    fromY: number;
    toX: number;
    toY: number;
    progress: number;
    color: string;
    weight: number;
  }>>([]);

  // Calculate neuron positions - vertical layout like reference
  const positions = useMemo(() => {
    if (!topology) return new Map<string, NeuronPosition>();

    const positions = new Map<string, NeuronPosition>();
    const layerCount = topology.layer_sizes.length;
    const padding = 80;
    const availableWidth = width - 2 * padding;
    const availableHeight = height - 2 * padding;

    topology.layer_sizes.forEach((size, layerIdx) => {
      const x = padding + (layerIdx / Math.max(1, layerCount - 1)) * availableWidth;

      // Show all neurons but space them nicely
      const maxVisible = Math.min(size, 50);
      const spacing = availableHeight / (maxVisible + 1);

      for (let i = 0; i < maxVisible; i++) {
        const neuronIdx = size <= maxVisible ? i : Math.floor((i / maxVisible) * size);
        const y = padding + (i + 1) * spacing;
        positions.set(`${layerIdx}-${neuronIdx}`, { x, y, layer: layerIdx, index: neuronIdx });
      }
    });

    return positions;
  }, [topology, width, height]);

  // Create neuron state map
  const neuronStates = useMemo(() => {
    const map = new Map<string, NeuronState>();
    neurons.forEach(n => {
      map.set(`${n.layer}-${n.index}`, n);
    });
    return map;
  }, [neurons]);

  // Detect new spikes and create pulses
  useEffect(() => {
    if (!topology) return;

    const newPulses: typeof pulsesRef.current = [];

    // Check for neurons that just started spiking
    neuronStates.forEach((state, key) => {
      const prevState = prevNeuronsRef.current.get(key);
      const justSpiked = state.spiking && (!prevState || !prevState.spiking);

      if (justSpiked && state.layer < topology.layer_sizes.length - 1) {
        const fromPos = positions.get(key);
        if (!fromPos) return;

        // Create pulses to all connected neurons in next layer
        const nextLayerSize = topology.layer_sizes[state.layer + 1];
        const maxConnections = Math.min(nextLayerSize, 20); // Limit for performance

        for (let i = 0; i < maxConnections; i++) {
          const targetIdx = Math.floor((i / maxConnections) * nextLayerSize);
          const toPos = positions.get(`${state.layer + 1}-${targetIdx}`);
          if (!toPos) continue;

          const layerColor = LAYER_COLORS[state.layer] || LAYER_COLORS[0];
          newPulses.push({
            fromX: fromPos.x,
            fromY: fromPos.y,
            toX: toPos.x,
            toY: toPos.y,
            progress: 0,
            color: layerColor.glow,
            weight: Math.random() * 0.5 + 0.5,
          });
        }
      }
    });

    if (newPulses.length > 0) {
      pulsesRef.current = [...pulsesRef.current.slice(-100), ...newPulses];
    }

    // Update previous states
    prevNeuronsRef.current = new Map(neuronStates);
  }, [neuronStates, positions, topology]);

  // Draw function
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !topology) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    pulseTimeRef.current += 0.016; // ~60fps

    // Clear with dark background
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);

    // Draw subtle grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw connections with bezier curves
    if (topology.synapses.length < 10000) {
      topology.synapses.forEach(synapse => {
        const fromPos = positions.get(`${synapse.from_layer}-${synapse.from_index}`);
        const toPos = positions.get(`${synapse.to_layer}-${synapse.to_index}`);
        if (!fromPos || !toPos) return;

        const fromState = neuronStates.get(`${synapse.from_layer}-${synapse.from_index}`);
        const isActive = fromState?.spiking || (fromState?.membrane ?? 0) > 0.5;

        // Base connection visibility
        const baseAlpha = isActive ? 0.15 : 0.03;
        const layerColor = LAYER_COLORS[synapse.from_layer] || LAYER_COLORS[0];

        ctx.strokeStyle = isActive
          ? `rgba(${hexToRgb(layerColor.glow)}, ${baseAlpha})`
          : `rgba(100, 100, 120, ${baseAlpha})`;
        ctx.lineWidth = Math.abs(synapse.weight) * 1.5 + 0.5;

        // Draw bezier curve
        ctx.beginPath();
        ctx.moveTo(fromPos.x, fromPos.y);
        const cpX = (fromPos.x + toPos.x) / 2;
        ctx.bezierCurveTo(cpX, fromPos.y, cpX, toPos.y, toPos.x, toPos.y);
        ctx.stroke();
      });
    }

    // Draw and update pulses
    pulsesRef.current = pulsesRef.current.filter(pulse => {
      pulse.progress += 0.03;
      if (pulse.progress > 1) return false;

      // Calculate pulse position along bezier
      const t = pulse.progress;
      const cpX = (pulse.fromX + pulse.toX) / 2;
      const x = bezierPoint(pulse.fromX, cpX, cpX, pulse.toX, t);
      const y = bezierPoint(pulse.fromY, pulse.fromY, pulse.toY, pulse.toY, t);

      // Draw pulse glow
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15);
      gradient.addColorStop(0, pulse.color);
      gradient.addColorStop(0.5, `rgba(${hexToRgb(pulse.color)}, 0.5)`);
      gradient.addColorStop(1, 'transparent');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 15, 0, Math.PI * 2);
      ctx.fill();

      // Draw bright core
      ctx.fillStyle = 'white';
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();

      return true;
    });

    // Draw neurons
    positions.forEach((pos, key) => {
      const state = neuronStates.get(key);
      const membrane = state?.membrane ?? 0;
      const spiking = state?.spiking ?? false;
      const layerColor = LAYER_COLORS[pos.layer] || LAYER_COLORS[0];

      const baseRadius = 6;
      const radius = spiking ? baseRadius * 1.3 : baseRadius;

      // Outer glow for active neurons
      if (membrane > 0.3 || spiking) {
        const glowRadius = spiking ? 30 : 15 * membrane;
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, glowRadius);
        gradient.addColorStop(0, `rgba(${hexToRgb(layerColor.glow)}, ${spiking ? 0.8 : 0.4 * membrane})`);
        gradient.addColorStop(0.5, `rgba(${hexToRgb(layerColor.glow)}, ${spiking ? 0.3 : 0.1 * membrane})`);
        gradient.addColorStop(1, 'transparent');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, glowRadius, 0, Math.PI * 2);
        ctx.fill();
      }

      // Neuron body
      const bodyGradient = ctx.createRadialGradient(
        pos.x - radius * 0.3, pos.y - radius * 0.3, 0,
        pos.x, pos.y, radius
      );

      if (spiking) {
        bodyGradient.addColorStop(0, '#ffffff');
        bodyGradient.addColorStop(0.3, layerColor.glow);
        bodyGradient.addColorStop(1, layerColor.base);
      } else {
        const intensity = membrane;
        bodyGradient.addColorStop(0, `rgba(${hexToRgb(layerColor.glow)}, ${0.3 + intensity * 0.7})`);
        bodyGradient.addColorStop(1, `rgba(${hexToRgb(layerColor.base)}, ${0.2 + intensity * 0.5})`);
      }

      ctx.fillStyle = bodyGradient;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.fill();

      // Bright center for spiking
      if (spiking) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius * 0.4, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw layer labels
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';

    topology.layer_sizes.forEach((size, idx) => {
      const x = 80 + (idx / Math.max(1, topology.layer_sizes.length - 1)) * (width - 160);
      const layerColor = LAYER_COLORS[idx] || LAYER_COLORS[0];

      // Count active neurons
      let activeCount = 0;
      for (let i = 0; i < size; i++) {
        const state = neuronStates.get(`${idx}-${i}`);
        if (state?.spiking || (state?.membrane ?? 0) > 0.5) activeCount++;
      }
      const activePercent = Math.round((activeCount / size) * 100);

      ctx.fillStyle = layerColor.glow;
      ctx.fillText(layerColor.name, x, 30);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.fillText(`${activePercent}%`, x, 45);
    });

    animationRef.current = requestAnimationFrame(draw);
  }, [neurons, topology, positions, neuronStates, width, height]);

  // Start animation loop
  useEffect(() => {
    animationRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animationRef.current);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="rounded-lg"
      style={{ background: '#0a0a0f' }}
    />
  );
}

// Helper: convert hex to rgb string
function hexToRgb(hex: string): string {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return '255, 255, 255';
  return `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}`;
}

// Helper: cubic bezier point
function bezierPoint(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const mt = 1 - t;
  return mt * mt * mt * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t * t * t * p3;
}
