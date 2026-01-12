import { useEffect, useRef, useMemo } from 'react';
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

export function NetworkCanvas({ neurons, topology, width, height }: NetworkCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Calculate neuron positions based on layer sizes
  const positions = useMemo(() => {
    if (!topology) return new Map<string, NeuronPosition>();

    const positions = new Map<string, NeuronPosition>();
    const layerCount = topology.layer_sizes.length;
    const padding = 60;
    const availableWidth = width - 2 * padding;
    const availableHeight = height - 2 * padding;

    topology.layer_sizes.forEach((size, layerIdx) => {
      const x = padding + (layerIdx / (layerCount - 1)) * availableWidth;

      // Limit visible neurons per layer for large networks
      const maxVisible = 40;
      const visibleCount = Math.min(size, maxVisible);
      const spacing = availableHeight / (visibleCount + 1);

      for (let i = 0; i < visibleCount; i++) {
        const neuronIdx = size <= maxVisible ? i : Math.floor((i / visibleCount) * size);
        const y = padding + (i + 1) * spacing;
        positions.set(`${layerIdx}-${neuronIdx}`, { x, y, layer: layerIdx, index: neuronIdx });
      }
    });

    return positions;
  }, [topology, width, height]);

  // Create a map of current neuron states
  const neuronStates = useMemo(() => {
    const map = new Map<string, NeuronState>();
    neurons.forEach(n => {
      map.set(`${n.layer}-${n.index}`, n);
    });
    return map;
  }, [neurons]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !topology) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Draw synapses (connections)
    if (topology.synapses.length < 5000) { // Only draw if not too many
      ctx.lineWidth = 0.5;
      topology.synapses.forEach(synapse => {
        const fromPos = positions.get(`${synapse.from_layer}-${synapse.from_index}`);
        const toPos = positions.get(`${synapse.to_layer}-${synapse.to_index}`);
        if (!fromPos || !toPos) return;

        const alpha = Math.min(0.3, Math.abs(synapse.weight) * 0.3);
        const color = synapse.weight > 0 ? `rgba(59, 130, 246, ${alpha})` : `rgba(239, 68, 68, ${alpha})`;

        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(fromPos.x, fromPos.y);
        ctx.lineTo(toPos.x, toPos.y);
        ctx.stroke();
      });
    }

    // Draw neurons
    positions.forEach((pos, key) => {
      const state = neuronStates.get(key);
      const membrane = state?.membrane ?? 0;
      const spiking = state?.spiking ?? false;

      // Neuron radius
      const radius = 8;

      // Background circle
      ctx.fillStyle = '#1e293b';
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.fill();

      // Membrane potential fill (from bottom)
      const fillHeight = membrane * radius * 2;
      ctx.save();
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.clip();

      const gradient = ctx.createLinearGradient(pos.x, pos.y + radius, pos.x, pos.y - radius);
      gradient.addColorStop(0, '#3b82f6');
      gradient.addColorStop(1, '#60a5fa');
      ctx.fillStyle = gradient;
      ctx.fillRect(pos.x - radius, pos.y + radius - fillHeight, radius * 2, fillHeight);
      ctx.restore();

      // Border
      ctx.strokeStyle = spiking ? '#fbbf24' : '#475569';
      ctx.lineWidth = spiking ? 3 : 1;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.stroke();

      // Spike glow
      if (spiking) {
        ctx.shadowColor = '#fbbf24';
        ctx.shadowBlur = 15;
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // Output layer: show class label
      if (pos.layer === (topology?.layer_sizes.length ?? 0) - 1) {
        const spikeCount = state?.spike_count ?? 0;
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`${pos.index}: ${spikeCount}`, pos.x + radius + 5, pos.y + 4);
      }
    });

    // Draw layer labels
    ctx.fillStyle = '#64748b';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';

    const layerNames = ['Input', 'Hidden', 'Output'];
    topology.layer_sizes.forEach((_, idx) => {
      const x = 60 + (idx / (topology.layer_sizes.length - 1)) * (width - 120);
      ctx.fillText(layerNames[idx] || `Layer ${idx}`, x, 25);
    });

  }, [neurons, topology, positions, neuronStates, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="rounded-lg"
    />
  );
}
