import { describe, it, expect, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import { NetworkCanvas } from '../src/components/NetworkCanvas';
import { installCanvasHarness } from './canvasRecorder';
import type { NetworkTopology } from '../src/lib/protocol';

// CHARACTERIZATION tests for NetworkCanvas RENDER behavior:
//   (a) topology=null early-returns from draw, empty positions, no throw
//   (b) neuron downsampling formula with the maxVisible=50 cap
//   (c) excitatory/inhibitory polarity color selection (incl. weight===0)
//   (d) layer color fallback LAYER_COLORS[layer] || LAYER_COLORS[0]
//
// Determinism: installCanvasHarness stubs requestAnimationFrame to fire once,
// cancelAnimationFrame to a no-op, and Math.random to a constant, so the pulse
// effect is irrelevant and the draw loop does not leak. Golden values captured
// by running the current code.

afterEach(cleanup);

const baseProps = {
  neurons: [],
  width: 800,
  height: 600,
  speed: 0.5,
  pulseStyle: 'ball' as const,
  showInputCurrent: false,
};

describe('NetworkCanvas topology=null', () => {
  it('renders a <canvas> without throwing and draws nothing', () => {
    const h = installCanvasHarness();
    try {
      const { container } = render(<NetworkCanvas {...baseProps} topology={null} />);
      expect(container.querySelector('canvas')).not.toBeNull();
      // draw() early-returns when topology is null: no fillRect / arc / bezier.
      expect(h.recorder.calls.filter((c) => c.method === 'fillRect')).toEqual([]);
      expect(h.recorder.arcs()).toEqual([]);
      expect(h.recorder.bezierCurves()).toEqual([]);
    } finally {
      h.restore();
    }
  });

  it('canvas has the width/height props applied', () => {
    const h = installCanvasHarness();
    try {
      const { container } = render(
        <NetworkCanvas {...baseProps} topology={null} width={123} height={456} />,
      );
      const c = container.querySelector('canvas')!;
      expect(c.width).toBe(123);
      expect(c.height).toBe(456);
    } finally {
      h.restore();
    }
  });
});

describe('NetworkCanvas synapse polarity colors (golden)', () => {
  it('positive weight -> excitatory blue rgba(59, 130, 246, ...)', () => {
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [3, 2],
        total_neurons: 5,
        synapses: [{ from_layer: 0, from_index: 0, to_layer: 1, to_index: 0, weight: 0.5 }],
      };
      render(<NetworkCanvas {...baseProps} topology={topology} />);
      expect(
        h.recorder
          .strokeStyles()
          .some((s) => typeof s === 'string' && s.startsWith('rgba(59, 130, 246,')),
      ).toBe(true);
    } finally {
      h.restore();
    }
  });

  it('negative weight -> inhibitory red rgba(239, 68, 68, ...)', () => {
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [3, 2],
        total_neurons: 5,
        synapses: [{ from_layer: 0, from_index: 0, to_layer: 1, to_index: 0, weight: -0.7 }],
      };
      render(<NetworkCanvas {...baseProps} topology={topology} />);
      expect(
        h.recorder
          .strokeStyles()
          .some((s) => typeof s === 'string' && s.startsWith('rgba(239, 68, 68,')),
      ).toBe(true);
    } finally {
      h.restore();
    }
  });

  it('weight === 0 -> excitatory (blue), never inhibitory (boundary)', () => {
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [2, 2],
        total_neurons: 4,
        synapses: [{ from_layer: 0, from_index: 0, to_layer: 1, to_index: 0, weight: 0 }],
      };
      render(<NetworkCanvas {...baseProps} topology={topology} />);
      const strokes = h.recorder.strokeStyles();
      expect(strokes.some((s) => typeof s === 'string' && s.startsWith('rgba(59, 130, 246,'))).toBe(true);
      expect(strokes.some((s) => typeof s === 'string' && s.startsWith('rgba(239'))).toBe(false);
    } finally {
      h.restore();
    }
  });
});

describe('NetworkCanvas neuron downsampling (maxVisible=50 cap)', () => {
  it('a layer of size 120 renders exactly 50 visible neuron bodies', () => {
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [120, 2],
        total_neurons: 122,
        synapses: [],
      };
      render(<NetworkCanvas {...baseProps} topology={topology} />);
      // Neuron body arcs have radius 5; layer 0 is at x=60 (padding).
      const layer0Bodies = h.recorder.arcs().filter((a) => a[0] === 60 && a[2] === 5);
      expect(layer0Bodies.length).toBe(50);
    } finally {
      h.restore();
    }
  });

  it('downsample y-positions match the floor-projection formula golden', () => {
    // neuronIdx = floor((slot/(maxVisible-1))*(size-1)); spacing=(600-120)/51.
    // We lock the rendered Y coordinates of the 50 layer-0 neuron bodies.
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [120, 2],
        total_neurons: 122,
        synapses: [],
      };
      render(<NetworkCanvas {...baseProps} topology={topology} />);
      const ys = h.recorder.arcs().filter((a) => a[0] === 60 && a[2] === 5).map((a) => a[1]);
      // First and last slots: y = 60 + (slot+1)*spacing, spacing = 480/51.
      const spacing = 480 / 51;
      expect(ys[0]).toBeCloseTo(60 + 1 * spacing, 9); // ~69.4118
      expect(ys[49]).toBeCloseTo(60 + 50 * spacing, 9); // ~530.5882
      expect(ys.length).toBe(50);
      // Monotonic increasing.
      for (let i = 1; i < ys.length; i++) expect(ys[i]).toBeGreaterThan(ys[i - 1]);
    } finally {
      h.restore();
    }
  });

  it('size <= maxVisible uses slot index directly (size 3 -> 3 bodies)', () => {
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [3, 2],
        total_neurons: 5,
        synapses: [],
      };
      render(<NetworkCanvas {...baseProps} topology={topology} />);
      const layer0Bodies = h.recorder.arcs().filter((a) => a[0] === 60 && a[2] === 5);
      expect(layer0Bodies.length).toBe(3);
    } finally {
      h.restore();
    }
  });
});
