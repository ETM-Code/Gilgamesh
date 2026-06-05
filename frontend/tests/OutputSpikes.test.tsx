import { describe, it, expect, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import { OutputSpikes } from '../src/components/OutputSpikes';

// CHARACTERIZATION tests for OutputSpikes. The component is a pure function of
// props: bar width = (count / Math.max(...spikes, 1)) * 100 + '%'. Golden
// width strings below were captured by running the real component in jsdom,
// not hand-derived (the repeating-decimal floats come straight from JS).

afterEach(cleanup);

// The inner bar div is the element carrying an inline width style. We locate
// each row's bar by walking the rendered rows in order.
function barWidths(container: HTMLElement): string[] {
  // Each row's track is the `div.flex-1` whose child is the colored bar.
  const bars = container.querySelectorAll<HTMLElement>(
    'div.flex-1 > div',
  );
  return Array.from(bars).map((b) => b.style.width);
}

describe('OutputSpikes bar widths', () => {
  it('locks width percentages for [0,3,1,6,2] (max=6)', () => {
    const { container } = render(
      <OutputSpikes spikes={[0, 3, 1, 6, 2]} prediction={3} />,
    );
    // Golden values observed from the running component (max = 6).
    expect(barWidths(container)).toEqual([
      '0%',
      '50%',
      '16.666666666666664%',
      '100%',
      '33.33333333333333%',
    ]);
  });

  it('EDGE: empty spikes renders zero bars (Math.max(...[],1)=1, no rows)', () => {
    const { container } = render(<OutputSpikes spikes={[]} prediction={0} />);
    expect(barWidths(container)).toEqual([]);
  });

  it('EDGE: all-zero spikes [0,0,0] -> max floored to 1 -> all 0% (no divide-by-zero)', () => {
    // CHARACTERIZATION: maxSpikes = Math.max(...spikes, 1) floors the
    // denominator at 1 so an all-zero input yields 0% widths instead of NaN.
    const { container } = render(
      <OutputSpikes spikes={[0, 0, 0]} prediction={1} />,
    );
    expect(barWidths(container)).toEqual(['0%', '0%', '0%']);
  });

  it('predicted row bar uses bg-green-500, others bg-blue-500', () => {
    const { container } = render(
      <OutputSpikes spikes={[0, 3, 1, 6, 2]} prediction={3} />,
    );
    const bars = container.querySelectorAll<HTMLElement>('div.flex-1 > div');
    expect(bars[3].className).toContain('bg-green-500');
    expect(bars[0].className).toContain('bg-blue-500');
    expect(bars[1].className).toContain('bg-blue-500');
    expect(bars[2].className).toContain('bg-blue-500');
    expect(bars[4].className).toContain('bg-blue-500');
    expect(bars[3].className).not.toContain('bg-blue-500');
  });

  it('predicted index label has text-green-400 font-bold; others text-slate-500', () => {
    const { container } = render(
      <OutputSpikes spikes={[0, 3, 1, 6, 2]} prediction={3} />,
    );
    // The index label is the first <span> inside each row.
    const rows = container.querySelectorAll<HTMLElement>('div.flex.items-center');
    const labelSpan = (row: HTMLElement) => row.querySelector('span') as HTMLElement;
    expect(labelSpan(rows[3]).className).toContain('text-green-400');
    expect(labelSpan(rows[3]).className).toContain('font-bold');
    expect(labelSpan(rows[0]).className).toContain('text-slate-500');
    expect(labelSpan(rows[0]).className).not.toContain('text-green-400');
  });

  it('renders the index and count text for each row', () => {
    const { container } = render(
      <OutputSpikes spikes={[0, 3, 1, 6, 2]} prediction={3} />,
    );
    const rows = container.querySelectorAll<HTMLElement>('div.flex.items-center');
    // Each row has [indexSpan, track, countSpan].
    const text = (row: HTMLElement) =>
      Array.from(row.querySelectorAll('span')).map((s) => s.textContent);
    expect(text(rows[0])).toEqual(['0', '0']);
    expect(text(rows[1])).toEqual(['1', '3']);
    expect(text(rows[3])).toEqual(['3', '6']);
  });

  it('renders the "Output Spikes" heading', () => {
    const { container } = render(
      <OutputSpikes spikes={[1]} prediction={0} />,
    );
    expect(container.querySelector('h3')?.textContent).toBe('Output Spikes');
  });
});
