import { useEffect, useRef, useMemo, useCallback } from 'react';
import type { NeuronState, NetworkTopology, SynapseInfo, PulseStyle } from '../lib/protocol';

interface NetworkCanvasProps {
  neurons: NeuronState[];
  topology: NetworkTopology | null;
  width: number;
  height: number;
  speed: number;
  pulseStyle: PulseStyle;
  showInputCurrent: boolean;
}

// Layer colors
const LAYER_COLORS = [
  { base: '#3b82f6', glow: '#60a5fa', name: 'Input' },    // Blue
  { base: '#a855f7', glow: '#c084fc', name: 'Hidden' },   // Purple
  { base: '#f97316', glow: '#fb923c', name: 'Output' },   // Orange
];

// Synapse colors
const EXCITATORY_COLOR = { base: '#3b82f6', glow: '#60a5fa' }; // Blue
const INHIBITORY_COLOR = { base: '#ef4444', glow: '#f87171' }; // Red

export function NetworkCanvas({
  neurons,
  topology,
  width,
  height,
  speed,
  pulseStyle,
  showInputCurrent,
}: NetworkCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const timeRef = useRef<number>(0);
  const pulsesRef = useRef<Array<{
    fromX: number;
    fromY: number;
    toX: number;
    toY: number;
    progress: number;
    color: string;
    weight: number;
  }>>([]);
  const prevSpikingRef = useRef<Set<string>>(new Set());

  // Build neuron state lookup
  const neuronStates = useMemo(() => {
    const map = new Map<string, NeuronState>();
    neurons.forEach(n => map.set(neuronKey(n.layer, n.index), n));
    return map;
  }, [neurons]);

  // Calculate positions for VISIBLE neurons only
  // Key insight: only create positions for neurons we're actually going to display
  const { positions, visibleNeurons } = useMemo(() => {
    if (!topology) {
      return { positions: new Map<string, { x: number; y: number }>(), visibleNeurons: new Map<number, number[]>() };
    }

    const positions = new Map<string, { x: number; y: number }>();
    const visibleNeurons = new Map<number, number[]>(); // layer -> visible indices
    const layerCount = topology.layer_sizes.length;
    const padding = 60;
    const availableWidth = width - 2 * padding;
    const availableHeight = height - 2 * padding;

    topology.layer_sizes.forEach((size, layerIdx) => {
      const x = padding + (layerIdx / Math.max(1, layerCount - 1)) * availableWidth;
      const maxVisible = Math.min(size, 50);
      const spacing = availableHeight / (maxVisible + 1);

      const indices: number[] = [];

      for (let slot = 0; slot < maxVisible; slot++) {
        // Map visual slot to neuron index
        const neuronIdx = size <= maxVisible
          ? slot
          : Math.floor((slot / (maxVisible - 1)) * (size - 1));

        const y = padding + (slot + 1) * spacing;
        const key = neuronKey(layerIdx, neuronIdx);

        positions.set(key, { x, y });
        indices.push(neuronIdx);
      }

      visibleNeurons.set(layerIdx, indices);
    });

    return { positions, visibleNeurons };
  }, [topology, width, height]);

  // Create pulses for active neurons
  useEffect(() => {
    if (!topology || !visibleNeurons.size) return;

    const newPulses: typeof pulsesRef.current = [];
    const currentSpiking = new Set<string>();

    // Check each visible neuron for activity
    visibleNeurons.forEach((indices, layerIdx) => {
      if (layerIdx >= topology.layer_sizes.length - 1) return; // Skip output layer

      const nextLayerIndices = visibleNeurons.get(layerIdx + 1) || [];

      indices.forEach(neuronIdx => {
        const key = neuronKey(layerIdx, neuronIdx);
        const state = neuronStates.get(key);
        if (!state) return;

        const fromPos = positions.get(key);
        if (!fromPos) return;

        // For input layer: emit based on membrane (constant current)
        // For other layers: emit on spike
        const isInput = layerIdx === 0;
        const shouldEmit = isInput
          ? (state.membrane > 0.2 && Math.random() < state.membrane * 0.4)
          : state.spiking;

        if (shouldEmit) {
          currentSpiking.add(key);

          // Only emit if this is a new spike (for non-input) or randomly (for input)
          const wasSpikingBefore = prevSpikingRef.current.has(key);
          if (isInput || !wasSpikingBefore) {
            // Create pulses to visible neurons in next layer
            const numPulses = Math.min(nextLayerIndices.length, isInput ? 3 : 8);
            for (let i = 0; i < numPulses; i++) {
              const targetIdx = nextLayerIndices[Math.floor(Math.random() * nextLayerIndices.length)];
              const toPos = positions.get(neuronKey(layerIdx + 1, targetIdx));
              if (!toPos) continue;

              // Find weight for this connection (approximate)
              const synapse = findSynapse(topology, layerIdx, neuronIdx, layerIdx + 1, targetIdx);
              const weight = synapse?.weight ?? 0.5;
              const isExcitatory = weight >= 0;

              newPulses.push({
                fromX: fromPos.x,
                fromY: fromPos.y,
                toX: toPos.x,
                toY: toPos.y,
                progress: 0,
                color: isExcitatory ? EXCITATORY_COLOR.glow : INHIBITORY_COLOR.glow,
                weight: Math.abs(weight),
              });
            }
          }
        }
      });
    });

    prevSpikingRef.current = currentSpiking;

    if (newPulses.length > 0) {
      pulsesRef.current = [...pulsesRef.current.slice(-150), ...newPulses];
    }
  }, [neuronStates, positions, topology, visibleNeurons]);

  // Draw function
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !topology) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dt = 0.016;
    timeRef.current += dt;

    // Clear
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);

    // Subtle grid
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

    // Draw synapses - ONLY between visible neurons
    visibleNeurons.forEach((fromIndices, fromLayer) => {
      const toLayer = fromLayer + 1;
      const toIndices = visibleNeurons.get(toLayer);
      if (!toIndices) return;

      fromIndices.forEach(fromIdx => {
        const fromKey = neuronKey(fromLayer, fromIdx);
        const fromPos = positions.get(fromKey);
        if (!fromPos) return;

        const fromState = neuronStates.get(fromKey);
        const active = isActive(fromState, 0.3);

        toIndices.forEach(toIdx => {
          const toKey = neuronKey(toLayer, toIdx);
          const toPos = positions.get(toKey);
          if (!toPos) return;

          // Find weight
          const synapse = findSynapse(topology, fromLayer, fromIdx, toLayer, toIdx);

          if (!synapse) return;

          const isExcitatory = synapse.weight >= 0;
          const weightMag = Math.abs(synapse.weight);
          const synapseColor = isExcitatory ? EXCITATORY_COLOR : INHIBITORY_COLOR;

          // Alpha and width based on weight
          const baseAlpha = 0.05 + weightMag * 0.2;
          const alpha = active ? Math.min(baseAlpha * 2.5, 0.6) : baseAlpha;
          const lineWidth = 0.3 + weightMag * 2;

          ctx.strokeStyle = `rgba(${hexToRgb(synapseColor.base)}, ${alpha})`;
          ctx.lineWidth = lineWidth;

          // Bezier curve
          ctx.beginPath();
          ctx.moveTo(fromPos.x, fromPos.y);
          const cpX = (fromPos.x + toPos.x) / 2;
          ctx.bezierCurveTo(cpX, fromPos.y, cpX, toPos.y, toPos.x, toPos.y);
          ctx.stroke();
        });
      });
    });

    // Draw and update pulses
    const pulseSpeed = 0.015 + speed * 0.025;

    pulsesRef.current = pulsesRef.current.filter(pulse => {
      pulse.progress += pulseSpeed;
      if (pulse.progress > 1) return false;

      const t = pulse.progress;
      const cpX = (pulse.fromX + pulse.toX) / 2;

      if (pulseStyle === 'ball') {
        const x = bezierPoint(pulse.fromX, cpX, cpX, pulse.toX, t);
        const y = bezierPoint(pulse.fromY, pulse.fromY, pulse.toY, pulse.toY, t);

        const glowSize = 10 + pulse.weight * 6;
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, glowSize);
        gradient.addColorStop(0, pulse.color);
        gradient.addColorStop(0.5, `rgba(${hexToRgb(pulse.color)}, 0.4)`);
        gradient.addColorStop(1, 'transparent');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, glowSize, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
      } else {
        // Electricity style
        const trailLength = 0.12;
        const startT = Math.max(0, t - trailLength);

        for (let i = 0; i < 8; i++) {
          const segT = startT + (t - startT) * (i / 8);
          const nextT = startT + (t - startT) * ((i + 1) / 8);

          const x1 = bezierPoint(pulse.fromX, cpX, cpX, pulse.toX, segT);
          const y1 = bezierPoint(pulse.fromY, pulse.fromY, pulse.toY, pulse.toY, segT);
          const x2 = bezierPoint(pulse.fromX, cpX, cpX, pulse.toX, nextT);
          const y2 = bezierPoint(pulse.fromY, pulse.fromY, pulse.toY, pulse.toY, nextT);

          const segAlpha = (i / 8) * 0.8;
          ctx.strokeStyle = `rgba(${hexToRgb(pulse.color)}, ${segAlpha})`;
          ctx.lineWidth = 1 + pulse.weight * 2 * (i / 8);
          ctx.lineCap = 'round';

          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }

        const headX = bezierPoint(pulse.fromX, cpX, cpX, pulse.toX, t);
        const headY = bezierPoint(pulse.fromY, pulse.fromY, pulse.toY, pulse.toY, t);

        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(headX, headY, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      return true;
    });

    // Draw neurons
    positions.forEach((pos, key) => {
      const state = neuronStates.get(key);
      const membrane = state?.membrane ?? 0;
      const spiking = state?.spiking ?? false;
      const layer = neuronKeyLayer(key);
      const color = layerColor(layer);
      const isInput = layer === 0;

      const baseRadius = 5;
      const radius = spiking ? baseRadius * 1.4 : baseRadius;

      // Input current visualization - pulsing rings
      if (isInput && showInputCurrent && membrane > 0.1) {
        const pulsePhase = (timeRef.current * 2 + pos.y * 0.01) % 1;
        const ringRadius = radius + pulsePhase * 20;
        const ringAlpha = (1 - pulsePhase) * membrane * 0.4;

        ctx.strokeStyle = `rgba(${hexToRgb(color.glow)}, ${ringAlpha})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, ringRadius, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Outer glow
      if (membrane > 0.2 || spiking) {
        const glowRadius = spiking ? 25 : 12 * membrane;
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, glowRadius);
        gradient.addColorStop(0, `rgba(${hexToRgb(color.glow)}, ${spiking ? 0.8 : 0.4 * membrane})`);
        gradient.addColorStop(0.6, `rgba(${hexToRgb(color.glow)}, ${spiking ? 0.2 : 0.1 * membrane})`);
        gradient.addColorStop(1, 'transparent');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, glowRadius, 0, Math.PI * 2);
        ctx.fill();
      }

      // Neuron body
      const intensity = membrane;
      const bodyGradient = ctx.createRadialGradient(
        pos.x - radius * 0.3, pos.y - radius * 0.3, 0,
        pos.x, pos.y, radius
      );

      if (spiking) {
        bodyGradient.addColorStop(0, '#ffffff');
        bodyGradient.addColorStop(0.4, color.glow);
        bodyGradient.addColorStop(1, color.base);
      } else {
        bodyGradient.addColorStop(0, `rgba(${hexToRgb(color.glow)}, ${0.3 + intensity * 0.7})`);
        bodyGradient.addColorStop(1, `rgba(${hexToRgb(color.base)}, ${0.2 + intensity * 0.5})`);
      }

      ctx.fillStyle = bodyGradient;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.fill();

      if (spiking) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius * 0.35, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Layer labels
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';

    topology.layer_sizes.forEach((size, idx) => {
      const layerIndices = visibleNeurons.get(idx) || [];
      if (layerIndices.length === 0) return;

      const firstPos = positions.get(neuronKey(idx, layerIndices[0]));
      if (!firstPos) return;

      const color = layerColor(idx);

      let activeCount = 0;
      layerIndices.forEach(nIdx => {
        if (isActive(neuronStates.get(neuronKey(idx, nIdx)), 0.5)) activeCount++;
      });
      const activePercent = Math.round((activeCount / size) * 100);

      ctx.fillStyle = color.glow;
      ctx.fillText(color.name, firstPos.x, 25);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.fillText(`${activePercent}%`, firstPos.x, 40);
    });

    // Legend
    const legendY = height - 25;
    ctx.font = '10px system-ui, sans-serif';

    ctx.fillStyle = EXCITATORY_COLOR.glow;
    ctx.beginPath();
    ctx.arc(width - 150, legendY, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.textAlign = 'left';
    ctx.fillText('Excitatory', width - 140, legendY + 3);

    ctx.fillStyle = INHIBITORY_COLOR.glow;
    ctx.beginPath();
    ctx.arc(width - 70, legendY, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.fillText('Inhibitory', width - 60, legendY + 3);

    animationRef.current = requestAnimationFrame(draw);
  }, [neurons, topology, positions, neuronStates, visibleNeurons, width, height, speed, pulseStyle, showInputCurrent]);

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

// Stable string key for a neuron position in the `${layer}-${index}` namespace.
const neuronKey = (layer: number, index: number) => `${layer}-${index}`;
const neuronKeyLayer = (key: string) => parseInt(key.split('-')[0]);

// LAYER_COLORS fallback: layers beyond the palette reuse the input (index 0) color.
const layerColor = (idx: number) => LAYER_COLORS[idx] || LAYER_COLORS[0];

// A neuron counts as "active" when spiking or its membrane exceeds the threshold.
const isActive = (state: NeuronState | undefined, threshold: number) =>
  !!state?.spiking || (state?.membrane ?? 0) > threshold;

// First synapse (if any) connecting (fromLayer, fromIndex) -> (toLayer, toIndex).
function findSynapse(
  topology: NetworkTopology,
  fromLayer: number,
  fromIndex: number,
  toLayer: number,
  toIndex: number,
): SynapseInfo | undefined {
  return topology.synapses.find(
    s => s.from_layer === fromLayer && s.from_index === fromIndex &&
         s.to_layer === toLayer && s.to_index === toIndex,
  );
}

function hexToRgb(hex: string): string {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return '255, 255, 255';
  return `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}`;
}

function bezierPoint(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const mt = 1 - t;
  return mt * mt * mt * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t * t * t * p3;
}
