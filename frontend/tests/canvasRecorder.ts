// Shared test helper: a recording 2D-canvas context mock.
//
// jsdom does not implement HTMLCanvasElement.prototype.getContext, so any
// component whose draw path needs a real CanvasRenderingContext2D is otherwise
// untestable (the effect early-returns on a null ctx, or jsdom logs a
// "Not implemented" error). This installs a recording mock that captures every
// call and the assigned fillStyle / strokeStyle at call time, which lets us pin
// the randomness-free draw output (hexToRgb output, bezier coords, fillRect
// tuples, etc.) as golden masters.

import { vi } from 'vitest';

export interface DrawCall {
  method: string;
  args: unknown[];
  fillStyle: unknown;
  strokeStyle: unknown;
  lineWidth: unknown;
}

export interface Recorder {
  calls: DrawCall[];
  ctx: Record<string, unknown>;
  /** fillRect tuples as [fillStyle, x, y, w, h] captured at call time. */
  fillRects: () => Array<[unknown, number, number, number, number]>;
  /** strokeStyle strings captured at each stroke() call. */
  strokeStyles: () => unknown[];
  /** all bezierCurveTo arg arrays. */
  bezierCurves: () => number[][];
  /** all moveTo arg arrays. */
  moveTos: () => number[][];
  /** all arc() arg arrays. */
  arcs: () => number[][];
}

export function createRecorder(): Recorder {
  const calls: DrawCall[] = [];
  let fillStyle: unknown = '';
  let strokeStyle: unknown = '';
  let lineWidth: unknown = 1;

  const record = (method: string) => (...args: unknown[]) => {
    calls.push({ method, args, fillStyle, strokeStyle, lineWidth });
  };

  const gradient = {
    addColorStop: record('gradient.addColorStop'),
  };

  const ctx: Record<string, unknown> = {
    fillRect: record('fillRect'),
    strokeRect: record('strokeRect'),
    clearRect: record('clearRect'),
    beginPath: record('beginPath'),
    closePath: record('closePath'),
    moveTo: record('moveTo'),
    lineTo: record('lineTo'),
    bezierCurveTo: record('bezierCurveTo'),
    quadraticCurveTo: record('quadraticCurveTo'),
    arc: record('arc'),
    fill: record('fill'),
    stroke: record('stroke'),
    fillText: record('fillText'),
    save: record('save'),
    restore: record('restore'),
    translate: record('translate'),
    scale: record('scale'),
    rotate: record('rotate'),
    setLineDash: record('setLineDash'),
    createRadialGradient: (...args: unknown[]) => {
      calls.push({ method: 'createRadialGradient', args, fillStyle, strokeStyle, lineWidth });
      return gradient;
    },
    createLinearGradient: (...args: unknown[]) => {
      calls.push({ method: 'createLinearGradient', args, fillStyle, strokeStyle, lineWidth });
      return gradient;
    },
  };

  Object.defineProperty(ctx, 'fillStyle', {
    get: () => fillStyle,
    set: (v) => { fillStyle = v; },
  });
  Object.defineProperty(ctx, 'strokeStyle', {
    get: () => strokeStyle,
    set: (v) => { strokeStyle = v; },
  });
  Object.defineProperty(ctx, 'lineWidth', {
    get: () => lineWidth,
    set: (v) => { lineWidth = v; },
  });
  // Plain writable props the draw code sets but we do not assert on.
  ctx.font = '';
  ctx.textAlign = '';
  ctx.lineCap = '';
  ctx.globalAlpha = 1;

  return {
    calls,
    ctx,
    fillRects: () =>
      calls
        .filter((c) => c.method === 'fillRect')
        .map((c) => [c.fillStyle, c.args[0], c.args[1], c.args[2], c.args[3]] as [unknown, number, number, number, number]),
    strokeStyles: () =>
      calls.filter((c) => c.method === 'stroke').map((c) => c.strokeStyle),
    bezierCurves: () =>
      calls.filter((c) => c.method === 'bezierCurveTo').map((c) => c.args as number[]),
    moveTos: () =>
      calls.filter((c) => c.method === 'moveTo').map((c) => c.args as number[]),
    arcs: () =>
      calls.filter((c) => c.method === 'arc').map((c) => c.args as number[]),
  };
}

/**
 * Install the recorder as the 2D context for ALL canvases in the test, plus
 * deterministic requestAnimationFrame (fires the callback exactly once) and a
 * fixed Math.random. Returns the recorder. Caller must restore via the returned
 * restore() (or rely on vi.restoreAllMocks()/afterEach).
 */
export function installCanvasHarness(opts?: { randomValue?: number }): {
  recorder: Recorder;
  restore: () => void;
} {
  const recorder = createRecorder();

  const getContextSpy = vi
    .spyOn(HTMLCanvasElement.prototype, 'getContext')
    .mockImplementation(((type: string) => {
      if (type === '2d') return recorder.ctx as unknown as CanvasRenderingContext2D;
      return null;
    }) as typeof HTMLCanvasElement.prototype.getContext);

  // rAF fires its callback exactly once so a single deterministic frame draws,
  // then returns a handle. draw() self-schedules via requestAnimationFrame at
  // its tail, so we must NOT re-invoke on subsequent calls or it recurses
  // forever. Only the first scheduled callback runs; later ones are dropped.
  // cancelAnimationFrame is a no-op. This prevents an unbounded self-scheduling
  // loop from leaking across tests.
  let rafFired = false;
  const rafSpy = vi
    .spyOn(globalThis, 'requestAnimationFrame')
    .mockImplementation(((cb: FrameRequestCallback) => {
      if (!rafFired) {
        rafFired = true;
        cb(0);
      }
      return 1;
    }) as typeof requestAnimationFrame);

  const cafSpy = vi
    .spyOn(globalThis, 'cancelAnimationFrame')
    .mockImplementation(() => {});

  const randomSpy = vi
    .spyOn(Math, 'random')
    .mockReturnValue(opts?.randomValue ?? 0.999);

  return {
    recorder,
    restore: () => {
      getContextSpy.mockRestore();
      rafSpy.mockRestore();
      cafSpy.mockRestore();
      randomSpy.mockRestore();
    },
  };
}
