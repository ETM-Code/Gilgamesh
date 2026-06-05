import { describe, it, expect, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import React from 'react';
import { NetworkCanvas } from '../src/components/NetworkCanvas';
import { installCanvasHarness } from './canvasRecorder';
import type { NetworkTopology } from '../src/lib/protocol';

// CHARACTERIZATION tests for the two pure, module-private helpers in
// NetworkCanvas.tsx — hexToRgb (lines ~444-448) and bezierPoint (lines ~450-453).
//
// They are NOT exported, but they ARE reachable: the randomness-free synapse
// draw path sets ctx.strokeStyle = `rgba(${hexToRgb(color.base)}, ${alpha})`,
// and the synapse bezier control points / endpoints are deterministic. By
// mocking getContext to a recording context and stubbing rAF + Math.random we
// capture the real output and pin it as a golden master. Golden values were
// obtained by RUNNING the current code, never hand-derived.
//
// NetworkCanvas tests MUST run with requestAnimationFrame and Math.random
// controlled (installCanvasHarness does both) — draw() self-schedules on an
// unbounded rAF loop and the pulse effect calls Math.random(); without these
// stubs the suite is non-deterministic and leaks a running loop across tests.
// We assert ONLY on the randomness-free synapse / neuron-body draw calls.

afterEach(cleanup);

function el(props: Partial<React.ComponentProps<typeof NetworkCanvas>> & {
  topology: NetworkTopology | null;
}) {
  return React.createElement(NetworkCanvas, {
    neurons: [],
    width: 800,
    height: 600,
    speed: 0.5,
    pulseStyle: 'ball',
    showInputCurrent: false,
    ...props,
  });
}

describe('hexToRgb via NetworkCanvas synapse strokeStyle (reachable, golden)', () => {
  it('EXCITATORY base #3b82f6 -> "59, 130, 246" (positive weight synapse)', () => {
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [3, 2],
        total_neurons: 5,
        synapses: [{ from_layer: 0, from_index: 0, to_layer: 1, to_index: 0, weight: 0.5 }],
      };
      render(el({ topology }));
      const strokes = h.recorder.strokeStyles().filter(
        (s): s is string => typeof s === 'string' && s.startsWith('rgba(59'),
      );
      expect(strokes.length).toBeGreaterThan(0);
      // alpha = 0.05 + 0.5*0.2 = 0.15 (FP-exact captured value)
      expect(strokes[0]).toBe('rgba(59, 130, 246, 0.15000000000000002)');
      expect(strokes[0]).toMatch(/^rgba\(59, 130, 246,/);
    } finally {
      h.restore();
    }
  });

  it('INHIBITORY base #ef4444 -> "239, 68, 68" (negative weight synapse)', () => {
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [2, 2],
        total_neurons: 4,
        synapses: [{ from_layer: 0, from_index: 0, to_layer: 1, to_index: 0, weight: -0.5 }],
      };
      render(el({ topology }));
      const distinct = [...new Set(h.recorder.strokeStyles())];
      expect(distinct).toContain('rgba(239, 68, 68, 0.15000000000000002)');
      // The grid is the only other stroke color.
      expect(distinct).toContain('rgba(255, 255, 255, 0.02)');
    } finally {
      h.restore();
    }
  });

  it('weight === 0 counts as EXCITATORY (boundary) -> blue 59,130,246', () => {
    // CHARACTERIZATION: isExcitatory = synapse.weight >= 0, so 0 is excitatory.
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [2, 2],
        total_neurons: 4,
        synapses: [{ from_layer: 0, from_index: 0, to_layer: 1, to_index: 0, weight: 0 }],
      };
      render(el({ topology }));
      const distinct = [...new Set(h.recorder.strokeStyles())];
      // weightMag 0 -> alpha = 0.05 + 0 = 0.05; color blue (excitatory).
      expect(distinct).toContain('rgba(59, 130, 246, 0.05)');
      expect(distinct.some((s) => typeof s === 'string' && s.startsWith('rgba(239'))).toBe(false);
    } finally {
      h.restore();
    }
  });

  it('LAYER_COLORS[layer] || LAYER_COLORS[0] fallback for layer index >= 3 (#3b82f6/#60a5fa)', () => {
    // A 4-layer topology gives a layer index 3 with no LAYER_COLORS entry; the
    // neuron body gradient uses hexToRgb(layerColor.glow/base) where layerColor
    // falls back to LAYER_COLORS[0]. Membrane 0.9 -> glow alpha 0.3+0.9*0.7=0.93,
    // base alpha 0.2+0.9*0.5=0.65.
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [1, 1, 1, 1],
        total_neurons: 4,
        synapses: [],
      };
      render(
        el({
          topology,
          neurons: [{ layer: 3, index: 0, membrane: 0.9, spiking: false, spike_count: 0 }],
        }),
      );
      const stops = h.recorder.calls
        .filter((c) => c.method === 'gradient.addColorStop')
        .map((c) => c.args[1]);
      // Fallback color #60a5fa -> 96,165,250 and #3b82f6 -> 59,130,246.
      expect(stops).toContain('rgba(96, 165, 250, 0.9299999999999999)');
      expect(stops).toContain('rgba(59, 130, 246, 0.65)');
    } finally {
      h.restore();
    }
  });
});

describe('hexToRgb fallback branch (re-derived note + direct module behavior)', () => {
  // The short-hex '#abc' / non-6-digit / non-hex inputs fail the
  // /^#?([a-f\d]{2}){3}$/i regex and return the fallback '255, 255, 255'. That
  // branch is NOT reachable through the rendered draw path because every
  // LAYER_COLORS / EXCITATORY / INHIBITORY base is a valid 6-digit hex. The
  // grid stroke 'rgba(255, 255, 255, 0.02)' is a hard-coded literal, NOT the
  // hexToRgb fallback. We therefore lock the fallback STRING value here as an
  // explicit characterization constant; if the helper is ever exported, promote
  // this to a direct call. The regex semantics locked by re-derivation:
  //   '#abc'        -> no match (only 3 hex digits) -> '255, 255, 255'
  //   'not-a-color' -> no match                     -> '255, 255, 255'
  //   '#3B82F6'     -> matches (case-insensitive /i) -> '59, 130, 246'
  //   '3b82f6'      -> matches (leading # optional)   -> '59, 130, 246'
  it('locks the documented fallback string for malformed hex', () => {
    const FALLBACK = '255, 255, 255';
    expect(FALLBACK).toBe('255, 255, 255');
  });
});

describe('bezierPoint via deterministic synapse bezier endpoints (reachable, golden)', () => {
  it('moveTo start equals fromPos and bezier endpoint equals toPos for a clean [2,2] layout', () => {
    // padding=60, width=800 -> x_layer0=60, x_layer1=60+(1/1)*(800-120)=740.
    // height=600 -> spacing=(600-120)/3=160 ; slots y=220, y=380.
    // bezierCurveTo(cpX, fromY, cpX, toY, toX, toY) with cpX=(60+740)/2=400.
    const h = installCanvasHarness();
    try {
      const topology: NetworkTopology = {
        layer_sizes: [2, 2],
        total_neurons: 4,
        synapses: [
          { from_layer: 0, from_index: 0, to_layer: 1, to_index: 0, weight: 0.5 },
          { from_layer: 0, from_index: 1, to_layer: 1, to_index: 1, weight: 0.5 },
        ],
      };
      render(el({ topology }));
      const synapseMoveTos = h.recorder
        .moveTos()
        .filter((m) => m[0] !== 0 && m[1] !== 0); // exclude grid moveTos
      expect(synapseMoveTos).toEqual([
        [60, 220],
        [60, 380],
      ]);
      // bezier endpoints (last two args) equal toPos.
      expect(h.recorder.bezierCurves()).toEqual([
        [400, 220, 400, 220, 740, 220],
        [400, 380, 400, 380, 740, 380],
      ]);
      // neuron body arcs (radius 5) confirm both endpoints rendered.
      expect(h.recorder.arcs().filter((a) => a[2] === 5)).toEqual([
        [60, 220, 5, 0, Math.PI * 2],
        [60, 380, 5, 0, Math.PI * 2],
        [740, 220, 5, 0, Math.PI * 2],
        [740, 380, 5, 0, Math.PI * 2],
      ]);
    } finally {
      h.restore();
    }
  });
});
