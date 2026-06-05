import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useWebSocket } from '../src/hooks/useWebSocket';

// CHARACTERIZATION tests for useWebSocket. A deterministic fake WebSocket is
// installed on globalThis (no network). We drive onopen / onclose / onmessage /
// send() manually and lock the observed state transitions and the exact wire
// JSON. Golden values captured by running the current code.

// Minimal fake matching the bits the hook uses.
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState = FakeWebSocket.CONNECTING;
  onopen: ((ev: unknown) => void) | null = null;
  onclose: ((ev: unknown) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  sent: string[] = [];
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.closed = true;
    this.readyState = FakeWebSocket.CLOSED;
  }

  // test helpers
  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.({});
  }

  message(data: unknown) {
    this.onmessage?.({ data: typeof data === 'string' ? data : JSON.stringify(data) });
  }

  // Simulates the browser firing onclose after the socket has actually closed
  // (readyState transitions to CLOSED before the handler runs).
  triggerClose() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({});
  }
}

let origWebSocket: typeof globalThis.WebSocket;

beforeEach(() => {
  FakeWebSocket.instances = [];
  origWebSocket = globalThis.WebSocket;
  // @ts-expect-error overriding global for the test
  globalThis.WebSocket = FakeWebSocket;
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  globalThis.WebSocket = origWebSocket;
  vi.restoreAllMocks();
  vi.useRealTimers();
});

function last(): FakeWebSocket {
  return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
}

describe('useWebSocket connection lifecycle', () => {
  it('constructs a WebSocket with the given url and starts disconnected', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    expect(FakeWebSocket.instances.length).toBe(1);
    expect(last().url).toBe('ws://test/ws');
    expect(result.current.connected).toBe(false);
  });

  it('onopen sets connected true and clears error', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    act(() => last().open());
    expect(result.current.connected).toBe(true);
    expect(result.current.error).toBeNull();
  });

  it('onerror sets error to "Connection error"', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    act(() => last().onerror?.({}));
    expect(result.current.error).toBe('Connection error');
  });

  it('onclose sets connected false and schedules a 2000ms reconnect (new WebSocket)', () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    act(() => last().open());
    expect(result.current.connected).toBe(true);

    act(() => last().triggerClose());
    expect(result.current.connected).toBe(false);
    expect(FakeWebSocket.instances.length).toBe(1);

    act(() => {
      vi.advanceTimersByTime(2000);
    });
    // A new WebSocket was constructed by the scheduled reconnect. (connect()
    // guards on readyState===OPEN; the closed socket is CLOSED, so a new one
    // IS created.)
    expect(FakeWebSocket.instances.length).toBe(2);
  });
});

describe('useWebSocket message routing', () => {
  it('topology message (has layer_sizes) sets topology and leaves frame untouched', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    const topo = { layer_sizes: [2, 3], total_neurons: 5, synapses: [] };
    act(() => last().message(topo));
    expect(result.current.topology).toEqual(topo);
    expect(result.current.frame).toBeNull();
    expect(result.current.status).toBeNull();
  });

  it('AnimationFrame sets frame slice', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    const frame = {
      type: 'AnimationFrame',
      neurons: [],
      step: 1,
      total_steps: 25,
      sample_index: 0,
      label: 3,
      prediction: 3,
      correct: true,
      output_spikes: [0, 1, 2],
      image_pixels: [],
      image_size: 28,
      paused: false,
    };
    act(() => last().message(frame));
    expect(result.current.frame).toEqual(frame);
    expect(result.current.topology).toBeNull();
  });

  it('Status sets status slice', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    const status = {
      type: 'Status',
      mode: 'inference',
      config: {},
      total_samples: 10000,
      checkpoint_loaded: null,
    };
    act(() => last().message(status));
    expect(result.current.status).toEqual(status);
  });

  it('TrainingUpdate sets training slice', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    const training = {
      type: 'TrainingUpdate',
      epoch: 1,
      total_epochs: 10,
      loss: 0.5,
      train_accuracy: 0.9,
      test_accuracy: 0.85,
      learning_rate: 0.001,
      best_test_accuracy: 0.85,
    };
    act(() => last().message(training));
    expect(result.current.training).toEqual(training);
  });

  it('Error sets error to the message field', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    act(() => last().message({ type: 'Error', message: 'boom' }));
    expect(result.current.error).toBe('boom');
  });

  it('Ack is a no-op: all state slices remain null', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    act(() => last().message({ type: 'Ack', command: 'NextSample' }));
    expect(result.current.frame).toBeNull();
    expect(result.current.status).toBeNull();
    expect(result.current.topology).toBeNull();
    expect(result.current.training).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('WeightMatrix is a no-op: all state slices remain null', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    act(() =>
      last().message({
        type: 'WeightMatrix',
        layer: 'fc1',
        data: [1, 2, 3, 4],
        rows: 2,
        cols: 2,
        min: 1,
        max: 4,
      }),
    );
    expect(result.current.frame).toBeNull();
    expect(result.current.status).toBeNull();
    expect(result.current.topology).toBeNull();
    expect(result.current.training).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('CHARACTERIZATION: a message with BOTH layer_sizes AND type:Status routes to TOPOLOGY (structural-typing quirk, possibly surprising, locked)', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    const hybrid = {
      type: 'Status',
      layer_sizes: [4, 4],
      total_neurons: 8,
      synapses: [],
      mode: 'idle',
    };
    act(() => last().message(hybrid));
    // 'layer_sizes' in msg returns BEFORE the type switch -> topology set,
    // status NOT set.
    expect(result.current.topology).toEqual(hybrid);
    expect(result.current.status).toBeNull();
  });

  it('malformed JSON is swallowed (no throw, state unchanged)', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    expect(() => act(() => last().message('{bad json'))).not.toThrow();
    expect(result.current.frame).toBeNull();
    expect(result.current.topology).toBeNull();
    expect(result.current.error).toBeNull();
  });
});

describe('useWebSocket send()', () => {
  it('before OPEN: does NOT call ws.send (dropped + warned)', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    // readyState is CONNECTING, not OPEN.
    act(() => result.current.send({ type: 'NextSample' }));
    expect(last().sent).toEqual([]);
  });

  it('when OPEN: sends the exact JSON for the ClientMessage', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    act(() => last().open());
    act(() => result.current.send({ type: 'SetSpeed', speed: 1.5 }));
    expect(last().sent).toEqual(['{"type":"SetSpeed","speed":1.5}']);
  });

  it('when OPEN: a discriminated-union message serializes its full shape', () => {
    const { result } = renderHook(() => useWebSocket('ws://test/ws'));
    act(() => last().open());
    act(() => result.current.send({ type: 'JumpToSample', index: 42 }));
    expect(last().sent).toEqual(['{"type":"JumpToSample","index":42}']);
  });
});
