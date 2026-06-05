import { describe, it, expect, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import { MnistDisplay } from '../src/components/MnistDisplay';
import { installCanvasHarness } from './canvasRecorder';

// CHARACTERIZATION tests for MnistDisplay. Under jsdom, canvas.getContext('2d')
// returns null, so the drawing useEffect is a no-op and only the surrounding
// markup (a pure function of props) is asserted. The canvas width/height come
// from displaySize = size === 7 ? 140 : 112.

afterEach(cleanup);

function canvas(container: HTMLElement): HTMLCanvasElement {
  return container.querySelector('canvas')!;
}

describe('MnistDisplay text and correctness marker', () => {
  it('renders the label number', () => {
    const { container } = render(
      <MnistDisplay pixels={[]} size={28} label={7} prediction={2} correct={false} />,
    );
    // Label value sits in the first mono span (the prediction is in the
    // red/green pred span). Use distinct label/prediction to disambiguate.
    const labelOuter = container.querySelector<HTMLElement>('span.text-slate-400');
    expect(labelOuter!.textContent).toContain('Label:');
    expect(labelOuter!.querySelector('span.font-mono')!.textContent).toBe('7');
  });

  it('correct=true -> container span has text-green-400 and trailing " ✓"', () => {
    const { container } = render(
      <MnistDisplay pixels={[]} size={28} label={3} prediction={3} correct />,
    );
    const predSpan = container.querySelector<HTMLElement>('span.text-green-400');
    expect(predSpan).not.toBeNull();
    expect(predSpan!.textContent).toContain('✓');
    expect(container.querySelector('span.text-red-400')).toBeNull();
  });

  it('correct=false -> container span has text-red-400 and trailing " ✗"', () => {
    const { container } = render(
      <MnistDisplay pixels={[]} size={28} label={5} prediction={3} correct={false} />,
    );
    const predSpan = container.querySelector<HTMLElement>('span.text-red-400');
    expect(predSpan).not.toBeNull();
    expect(predSpan!.textContent).toContain('✗');
    expect(container.querySelector('span.text-green-400')).toBeNull();
  });

  it('renders the prediction number', () => {
    const { container } = render(
      <MnistDisplay pixels={[]} size={28} label={5} prediction={3} correct={false} />,
    );
    const predSpan = container.querySelector<HTMLElement>('span.text-red-400');
    expect(predSpan!.textContent).toContain('3');
  });
});

describe('MnistDisplay canvas displaySize magic-number branch', () => {
  it('size=7 -> canvas 140x140 (the only special-cased size)', () => {
    // CHARACTERIZATION: displaySize = size === 7 ? 140 : 112. Lock the magic
    // numbers; only 7 maps to 140.
    const { container } = render(
      <MnistDisplay pixels={[]} size={7} label={0} prediction={0} correct />,
    );
    expect(canvas(container).width).toBe(140);
    expect(canvas(container).height).toBe(140);
  });

  it('size=28 -> canvas 112x112', () => {
    const { container } = render(
      <MnistDisplay pixels={[]} size={28} label={0} prediction={0} correct />,
    );
    expect(canvas(container).width).toBe(112);
    expect(canvas(container).height).toBe(112);
  });

  it('size=14 -> canvas 112x112 (not special-cased)', () => {
    const { container } = render(
      <MnistDisplay pixels={[]} size={14} label={0} prediction={0} correct />,
    );
    expect(canvas(container).width).toBe(112);
    expect(canvas(container).height).toBe(112);
  });

  it('non-empty pixels under jsdom renders without throwing (effect no-op, ctx null)', () => {
    // pixels.length>0 enters the effect but getContext('2d') is null in jsdom,
    // so it returns early. Lock that this does not throw.
    expect(() =>
      render(
        <MnistDisplay
          pixels={[0, 0.5, 1, 0.25]}
          size={2}
          label={1}
          prediction={1}
          correct
        />,
      ),
    ).not.toThrow();
  });
});

describe('MnistDisplay pixel-drawing useEffect (recording-context golden)', () => {
  // With getContext mocked to a recording context the per-pixel draw loop runs.
  // scale = canvas.width/size; intensity = Math.floor(value*255);
  // fillStyle = `rgb(i,i,i)`; fillRect(x*scale, y*scale, scale, scale).
  // Golden tuples captured by running the current code.

  it('pixels=[0,0.5,1,0.25] size=2 -> 4 fillRects with Math.floor flooring (scale 56)', () => {
    const h = installCanvasHarness();
    try {
      // size=2 not special-cased -> displaySize 112 -> scale = 112/2 = 56.
      // Math.floor: 0.5*255=127.5->127 ; 0.25*255=63.75->63 ; 1*255=255 ; 0->0.
      render(<MnistDisplay pixels={[0, 0.5, 1, 0.25]} size={2} label={1} prediction={1} correct />);
      expect(h.recorder.fillRects()).toEqual([
        ['rgb(0, 0, 0)', 0, 0, 56, 56],
        ['rgb(127, 127, 127)', 56, 0, 56, 56],
        ['rgb(255, 255, 255)', 0, 56, 56, 56],
        ['rgb(63, 63, 63)', 56, 56, 56, 56],
      ]);
    } finally {
      h.restore();
    }
  });

  it('pixels=[] skips the loop (zero fillRect calls)', () => {
    const h = installCanvasHarness();
    try {
      render(<MnistDisplay pixels={[]} size={2} label={0} prediction={0} correct />);
      expect(h.recorder.fillRects()).toEqual([]);
    } finally {
      h.restore();
    }
  });

  it('size=7 uses scale=140/7=20 (the special-cased 140px canvas)', () => {
    const h = installCanvasHarness();
    try {
      // 49 pixels all value 1 -> rgb(255,255,255), scale 20.
      const pixels = new Array(49).fill(1);
      render(<MnistDisplay pixels={pixels} size={7} label={0} prediction={0} correct />);
      const rects = h.recorder.fillRects();
      expect(rects.length).toBe(49);
      // First pixel (x=0,y=0) and a known offset pixel.
      expect(rects[0]).toEqual(['rgb(255, 255, 255)', 0, 0, 20, 20]);
      // pixel (x=6,y=6) -> 6*20=120.
      expect(rects[48]).toEqual(['rgb(255, 255, 255)', 120, 120, 20, 20]);
    } finally {
      h.restore();
    }
  });
});
