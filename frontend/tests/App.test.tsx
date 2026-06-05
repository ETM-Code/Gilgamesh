import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, cleanup, fireEvent, within } from '@testing-library/react';

// CHARACTERIZATION tests for App.tsx: keyboard shortcuts, the input[type=text]
// guard, and the header stats derivation. useWebSocket is fully mocked (no real
// WebSocket) so the test is deterministic. NetworkCanvas / MnistDisplay draw
// effects are harmless here because their canvases get a null 2D context under
// jsdom (effects early-return). Golden values captured by running the code.

// Controllable mock state for the hook.
const sendSpy = vi.fn();
let mockState: {
  connected: boolean;
  frame: any;
  status: any;
  topology: any;
  send: typeof sendSpy;
};

vi.mock('../src/hooks/useWebSocket', () => ({
  useWebSocket: () => mockState,
}));

import App from '../src/App';

beforeEach(() => {
  sendSpy.mockClear();
  mockState = {
    connected: false,
    frame: null,
    status: null,
    topology: null,
    send: sendSpy,
  };
});

afterEach(cleanup);

// A complete AnimationFrame fixture. When App has a frame it renders the
// MnistDisplay + OutputSpikes sidebar, which require these fields (OutputSpikes
// spreads output_spikes into Math.max). Override `paused` per test.
function makeFrame(overrides: Record<string, unknown> = {}) {
  return {
    type: 'AnimationFrame',
    neurons: [],
    step: 0,
    total_steps: 25,
    sample_index: 0,
    label: 0,
    prediction: 0,
    correct: true,
    output_spikes: [0, 1, 2],
    image_pixels: [],
    image_size: 28,
    paused: false,
    ...overrides,
  };
}

describe('App keyboard shortcuts', () => {
  it('ArrowLeft and "a" send PrevSample', () => {
    render(<App />);
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(sendSpy).toHaveBeenLastCalledWith({ type: 'PrevSample' });
    fireEvent.keyDown(window, { key: 'a' });
    expect(sendSpy).toHaveBeenLastCalledWith({ type: 'PrevSample' });
  });

  it('ArrowRight and "d" send NextSample', () => {
    render(<App />);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(sendSpy).toHaveBeenLastCalledWith({ type: 'NextSample' });
    fireEvent.keyDown(window, { key: 'd' });
    expect(sendSpy).toHaveBeenLastCalledWith({ type: 'NextSample' });
  });

  it('"r" sends RestartSample', () => {
    render(<App />);
    fireEvent.keyDown(window, { key: 'r' });
    expect(sendSpy).toHaveBeenLastCalledWith({ type: 'RestartSample' });
  });

  it('Space sends Resume when frame.paused is true', () => {
    mockState.frame = makeFrame({ paused: true });
    render(<App />);
    const ev = new KeyboardEvent('keydown', { key: ' ', cancelable: true });
    const prevented = !window.dispatchEvent(ev);
    expect(sendSpy).toHaveBeenLastCalledWith({ type: 'Resume' });
    // preventDefault is honored.
    expect(prevented).toBe(true);
  });

  it('Space sends Pause when frame.paused is false', () => {
    mockState.frame = makeFrame({ paused: false });
    render(<App />);
    fireEvent.keyDown(window, { key: ' ' });
    expect(sendSpy).toHaveBeenLastCalledWith({ type: 'Pause' });
  });

  it('Space sends Pause when frame is null (frame?.paused is undefined -> falsy)', () => {
    // CHARACTERIZATION: frame?.paused is undefined when frame is null, so the
    // ternary takes the Pause branch.
    render(<App />);
    fireEvent.keyDown(window, { key: ' ' });
    expect(sendSpy).toHaveBeenLastCalledWith({ type: 'Pause' });
  });

  it('an unmapped key sends nothing', () => {
    render(<App />);
    fireEvent.keyDown(window, { key: 'x' });
    expect(sendSpy).not.toHaveBeenCalled();
  });

  it('keydown from an <input type="text"> is ignored (guard)', () => {
    const { container } = render(<App />);
    const input = document.createElement('input');
    input.type = 'text';
    container.appendChild(input);
    fireEvent.keyDown(input, { key: 'ArrowRight' });
    expect(sendSpy).not.toHaveBeenCalled();
  });
});

describe('App header stats derivation', () => {
  function header(container: HTMLElement) {
    return container.querySelector('header')!;
  }

  it('defaults to 0 neurons / 0 synapses / 0 active when topology and frame are null', () => {
    const { container } = render(<App />);
    const h = within(header(container));
    expect(h.getByText('NEURONS').previousSibling!.textContent).toBe('0');
    expect(h.getByText('SYNAPSES').previousSibling!.textContent).toBe('0');
    expect(h.getByText('ACTIVE').previousSibling!.textContent).toBe('0');
  });

  it('FPS shows "0" when disconnected', () => {
    const { container } = render(<App />);
    const h = within(header(container));
    expect(h.getByText('FPS').previousSibling!.textContent).toBe('0');
  });

  it('FPS shows "60" when connected', () => {
    mockState.connected = true;
    const { container } = render(<App />);
    const h = within(header(container));
    expect(h.getByText('FPS').previousSibling!.textContent).toBe('60');
  });

  it('reflects topology total_neurons / synapses.length and frame active spiking count', () => {
    mockState.topology = {
      total_neurons: 794,
      synapses: [{}, {}, {}],
      layer_sizes: [784, 10],
    };
    mockState.frame = {
      neurons: [
        { spiking: true },
        { spiking: false },
        { spiking: true },
      ],
      paused: true,
      image_pixels: [],
      image_size: 28,
      label: 0,
      prediction: 0,
      correct: true,
      output_spikes: [],
      step: 0,
      total_steps: 25,
      sample_index: 0,
    };
    const { container } = render(<App />);
    const h = within(header(container));
    expect(h.getByText('NEURONS').previousSibling!.textContent).toBe('794');
    expect(h.getByText('SYNAPSES').previousSibling!.textContent).toBe('3');
    expect(h.getByText('ACTIVE').previousSibling!.textContent).toBe('2');
  });
});

describe('App handleBehaviorChange (cast bypasses ClientMessage union)', () => {
  it('CHARACTERIZATION: selecting an end-of-sample radio sends SetEndOfSampleBehavior via `send(msg as any)` (the cast bypasses the discriminated union)', () => {
    const { container } = render(<App />);
    // The "Stop" radio (value="stop"); default is auto-advance.
    const stopRadio = container.querySelector<HTMLInputElement>('input[type="radio"][value="stop"]')!;
    fireEvent.click(stopRadio);
    expect(sendSpy).toHaveBeenLastCalledWith({ type: 'SetEndOfSampleBehavior', behavior: 'stop' });
  });
});
