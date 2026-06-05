import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, fireEvent, screen } from '@testing-library/react';
import { ControlPanel } from '../src/components/ControlPanel';
import type { ClientMessage } from '../src/lib/protocol';

// CHARACTERIZATION tests for ControlPanel. All rendered text and the
// ClientMessage emitted on each click are pure functions of props.

afterEach(cleanup);

function renderPanel(
  overrides: Partial<{
    paused: boolean;
    currentStep: number;
    totalSteps: number;
    sampleIndex: number;
    totalSamples: number;
    send: (msg: ClientMessage) => void;
  }> = {},
) {
  const send = overrides.send ?? vi.fn();
  const utils = render(
    <ControlPanel
      paused={overrides.paused ?? true}
      currentStep={overrides.currentStep ?? 5}
      totalSteps={overrides.totalSteps ?? 25}
      sampleIndex={overrides.sampleIndex ?? 0}
      totalSamples={overrides.totalSamples ?? 10000}
      send={send}
    />,
  );
  return { send, ...utils };
}

function progressBarWidth(container: HTMLElement): string {
  // The progress bar is the gradient div carrying an inline width.
  const bar = container.querySelector<HTMLElement>(
    'div.bg-gradient-to-r',
  );
  return bar!.style.width;
}

describe('ControlPanel text', () => {
  it('locks sample counter text = (sampleIndex+1) / totalSamples', () => {
    renderPanel({ sampleIndex: 0, totalSamples: 10000 });
    expect(screen.getByText('1 / 10000')).toBeInTheDocument();
  });

  it('locks step text = currentStep/totalSteps', () => {
    const { container } = renderPanel({ currentStep: 5, totalSteps: 25 });
    expect(container.textContent).toContain('5/25');
  });

  it('locks progress bar width = (currentStep/totalSteps)*100 + "%"', () => {
    const { container } = renderPanel({ currentStep: 5, totalSteps: 25 });
    expect(progressBarWidth(container)).toBe('20%');
  });

  it('EDGE: currentStep=0 -> progress width 0%', () => {
    const { container } = renderPanel({ currentStep: 0, totalSteps: 25 });
    expect(progressBarWidth(container)).toBe('0%');
  });

  it('EDGE: totalSteps=0, currentStep=0 -> width interpolates "NaN%", rejected by jsdom CSS -> reads back ""', () => {
    // CHARACTERIZATION: current behavior, possibly a bug, locked to detect
    // change. App.tsx defaults totalSteps to 25 (never 0), but if a caller
    // passes 0 with currentStep 0, the template literal yields the inline
    // style "NaN%". jsdom's CSSOM treats "NaN%" as an invalid CSS length and
    // drops it, so element.style.width reads back as the empty string.
    // (A real browser behaves the same: the invalid value is ignored.)
    const { container } = renderPanel({ currentStep: 0, totalSteps: 0 });
    expect(progressBarWidth(container)).toBe('');
  });

  it('EDGE: totalSteps=0, currentStep>0 -> "Infinity%", rejected by jsdom CSS -> reads back ""', () => {
    // CHARACTERIZATION: current behavior, possibly a bug, locked to detect
    // change. (5/0)*100 = Infinity -> inline style "Infinity%", an invalid CSS
    // length that jsdom (and real browsers) drop, so width reads back as "".
    const { container } = renderPanel({ currentStep: 5, totalSteps: 0 });
    expect(progressBarWidth(container)).toBe('');
  });
});

describe('ControlPanel Play/Pause button', () => {
  it('paused=true -> label "▶ Play" with bg-blue-500/80', () => {
    renderPanel({ paused: true });
    const btn = screen.getByText('▶ Play');
    expect(btn).toBeInTheDocument();
    expect(btn.className).toContain('bg-blue-500/80');
  });

  it('paused=false -> label "⏸ Pause" with bg-orange-500/80', () => {
    renderPanel({ paused: false });
    const btn = screen.getByText('⏸ Pause');
    expect(btn).toBeInTheDocument();
    expect(btn.className).toContain('bg-orange-500/80');
  });
});

describe('ControlPanel click -> ClientMessage', () => {
  it('Prev click sends {type:"PrevSample"}', () => {
    const { send } = renderPanel();
    fireEvent.click(screen.getByText('← Prev'));
    expect(send).toHaveBeenCalledWith({ type: 'PrevSample' });
  });

  it('Next click sends {type:"NextSample"}', () => {
    const { send } = renderPanel();
    fireEvent.click(screen.getByText('Next →'));
    expect(send).toHaveBeenCalledWith({ type: 'NextSample' });
  });

  it('Restart (↻) click sends {type:"RestartSample"}', () => {
    const { send } = renderPanel();
    fireEvent.click(screen.getByText('↻'));
    expect(send).toHaveBeenCalledWith({ type: 'RestartSample' });
  });

  it('Random (🎲) click sends {type:"RandomSample"}', () => {
    const { send } = renderPanel();
    fireEvent.click(screen.getByText('🎲'));
    expect(send).toHaveBeenCalledWith({ type: 'RandomSample' });
  });

  it('when paused, Play click sends {type:"Resume"}', () => {
    const { send } = renderPanel({ paused: true });
    fireEvent.click(screen.getByText('▶ Play'));
    expect(send).toHaveBeenCalledWith({ type: 'Resume' });
  });

  it('when not paused, Pause click sends {type:"Pause"}', () => {
    const { send } = renderPanel({ paused: false });
    fireEvent.click(screen.getByText('⏸ Pause'));
    expect(send).toHaveBeenCalledWith({ type: 'Pause' });
  });
});
