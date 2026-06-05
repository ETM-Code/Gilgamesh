import { describe, it, expect } from 'vitest';
import type {
  ServerMessage,
  ClientMessage,
  AnimationFrame,
  TrainingUpdate,
  WeightMatrix,
  Status,
  ErrorMsg,
  Ack,
  SimulationMode,
  EndOfSampleBehavior,
} from '../src/lib/protocol';

// CHARACTERIZATION test suite for the WebSocket wire protocol shared with the
// Rust backend. protocol.ts is types-only (zero runtime values), so these
// tests lock the contract via `satisfies` (compile-time assignability) plus
// runtime assertions on discriminant strings and serialized shapes. Any
// rename/removal of a variant, discriminant, or field will break either tsc
// (the build-smoke test) or one of these runtime golden assertions.

describe('protocol: ServerMessage discriminated union', () => {
  // One typed fixture per ServerMessage variant. `satisfies` keeps these
  // assignable at compile time without widening their literal `type`.
  const animationFrame = {
    type: 'AnimationFrame',
    neurons: [
      { layer: 0, index: 0, membrane: 0.5, spiking: true, spike_count: 3 },
    ],
    step: 5,
    total_steps: 25,
    sample_index: 0,
    label: 7,
    prediction: 7,
    correct: true,
    output_spikes: [0, 3, 1, 6, 2],
    image_pixels: [0, 0.5, 1],
    image_size: 28,
    paused: false,
  } satisfies AnimationFrame;

  const trainingUpdate = {
    type: 'TrainingUpdate',
    epoch: 1,
    total_epochs: 10,
    loss: 0.25,
    train_accuracy: 0.9,
    test_accuracy: 0.88,
    learning_rate: 0.001,
    best_test_accuracy: 0.9,
  } satisfies TrainingUpdate;

  const weightMatrix = {
    type: 'WeightMatrix',
    layer: 'fc1',
    data: [0.1, -0.2, 0.3],
    rows: 1,
    cols: 3,
    min: -0.2,
    max: 0.3,
  } satisfies WeightMatrix;

  const status = {
    type: 'Status',
    mode: 'idle',
    config: {},
    total_samples: 10000,
    checkpoint_loaded: null,
  } satisfies Status;

  const errorMsg = {
    type: 'Error',
    message: 'boom',
  } satisfies ErrorMsg;

  const ack = {
    type: 'Ack',
    command: 'NextSample',
  } satisfies Ack;

  const allServer: ServerMessage[] = [
    animationFrame,
    trainingUpdate,
    weightMatrix,
    status,
    errorMsg,
    ack,
  ];

  it('locks the exact discriminant string of every ServerMessage variant', () => {
    expect(animationFrame.type).toBe('AnimationFrame');
    expect(trainingUpdate.type).toBe('TrainingUpdate');
    expect(weightMatrix.type).toBe('WeightMatrix');
    expect(status.type).toBe('Status');
    // CHARACTERIZATION: deliberate asymmetry — the ErrorMsg interface's
    // discriminant is the string 'Error' (NOT 'ErrorMsg'). Locked to detect
    // change. The Rust backend serializes this variant as "Error".
    expect(errorMsg.type).toBe('Error');
    expect(ack.type).toBe('Ack');
  });

  it('locks the full sorted set of ServerMessage discriminants', () => {
    const tags = allServer.map((m) => m.type).sort();
    expect(tags).toEqual([
      'Ack',
      'AnimationFrame',
      'Error',
      'Status',
      'TrainingUpdate',
      'WeightMatrix',
    ]);
  });

  it('locks the full key set of a fully-populated AnimationFrame', () => {
    expect(Object.keys(animationFrame).sort()).toEqual([
      'correct',
      'image_pixels',
      'image_size',
      'label',
      'neurons',
      'output_spikes',
      'paused',
      'prediction',
      'sample_index',
      'step',
      'total_steps',
      'type',
    ]);
  });

  it('locks the NeuronState key set inside AnimationFrame.neurons', () => {
    expect(Object.keys(animationFrame.neurons[0]).sort()).toEqual([
      'index',
      'layer',
      'membrane',
      'spike_count',
      'spiking',
    ]);
  });

  it('locks the TrainingUpdate key set', () => {
    expect(Object.keys(trainingUpdate).sort()).toEqual([
      'best_test_accuracy',
      'epoch',
      'learning_rate',
      'loss',
      'test_accuracy',
      'total_epochs',
      'train_accuracy',
      'type',
    ]);
  });

  it('locks the WeightMatrix key set', () => {
    expect(Object.keys(weightMatrix).sort()).toEqual([
      'cols',
      'data',
      'layer',
      'max',
      'min',
      'rows',
      'type',
    ]);
  });

  it('locks the Status key set and null checkpoint default', () => {
    expect(Object.keys(status).sort()).toEqual([
      'checkpoint_loaded',
      'config',
      'mode',
      'total_samples',
      'type',
    ]);
    expect(status.checkpoint_loaded).toBeNull();
  });

  it('locks the ErrorMsg and Ack key sets', () => {
    expect(Object.keys(errorMsg).sort()).toEqual(['message', 'type']);
    expect(Object.keys(ack).sort()).toEqual(['command', 'type']);
  });
});

describe('protocol: ClientMessage variant tags and payloads', () => {
  // One typed fixture per ClientMessage variant.
  const getStatus = { type: 'GetStatus' } satisfies ClientMessage;
  const nextSample = { type: 'NextSample' } satisfies ClientMessage;
  const prevSample = { type: 'PrevSample' } satisfies ClientMessage;
  const jumpToSample = { type: 'JumpToSample', index: 42 } satisfies ClientMessage;
  const randomSample = { type: 'RandomSample' } satisfies ClientMessage;
  const pause = { type: 'Pause' } satisfies ClientMessage;
  const resume = { type: 'Resume' } satisfies ClientMessage;
  const setSpeed = { type: 'SetSpeed', speed: 1.5 } satisfies ClientMessage;
  const setEob = {
    type: 'SetEndOfSampleBehavior',
    behavior: 'auto-advance',
  } satisfies ClientMessage;
  const restartSample = { type: 'RestartSample' } satisfies ClientMessage;
  const startTraining = {
    type: 'StartTraining',
    config: { epochs: 10 },
  } satisfies ClientMessage;
  const stopTraining = { type: 'StopTraining' } satisfies ClientMessage;
  const loadCheckpoint = {
    type: 'LoadCheckpoint',
    path: '/ckpt/best.pt',
  } satisfies ClientMessage;
  const updateConfig = {
    type: 'UpdateConfig',
    config: { lr: 0.001 },
  } satisfies ClientMessage;
  const getWeights = { type: 'GetWeights' } satisfies ClientMessage;

  const allClient: ClientMessage[] = [
    getStatus,
    nextSample,
    prevSample,
    jumpToSample,
    randomSample,
    pause,
    resume,
    setSpeed,
    setEob,
    restartSample,
    startTraining,
    stopTraining,
    loadCheckpoint,
    updateConfig,
    getWeights,
  ];

  it('locks the full sorted set of ClientMessage tags', () => {
    const tags = allClient.map((m) => m.type).sort();
    expect(tags).toEqual([
      'GetStatus',
      'GetWeights',
      'JumpToSample',
      'LoadCheckpoint',
      'NextSample',
      'Pause',
      'PrevSample',
      'RandomSample',
      'RestartSample',
      'Resume',
      'SetEndOfSampleBehavior',
      'SetSpeed',
      'StartTraining',
      'StopTraining',
      'UpdateConfig',
    ]);
  });

  it('locks parameterized payload key sets', () => {
    expect(Object.keys(jumpToSample).sort()).toEqual(['index', 'type']);
    expect(Object.keys(setSpeed).sort()).toEqual(['speed', 'type']);
    expect(Object.keys(setEob).sort()).toEqual(['behavior', 'type']);
    expect(Object.keys(startTraining).sort()).toEqual(['config', 'type']);
    expect(Object.keys(loadCheckpoint).sort()).toEqual(['path', 'type']);
    expect(Object.keys(updateConfig).sort()).toEqual(['config', 'type']);
  });

  it('locks the exact JSON wire string of each ClientMessage fixture', () => {
    // JSON.stringify preserves insertion order; these strings are what cross
    // the WebSocket to the Rust server. A field rename or reorder of the
    // fixture literal would change these.
    expect(JSON.stringify(getStatus)).toBe('{"type":"GetStatus"}');
    expect(JSON.stringify(nextSample)).toBe('{"type":"NextSample"}');
    expect(JSON.stringify(prevSample)).toBe('{"type":"PrevSample"}');
    expect(JSON.stringify(jumpToSample)).toBe('{"type":"JumpToSample","index":42}');
    expect(JSON.stringify(randomSample)).toBe('{"type":"RandomSample"}');
    expect(JSON.stringify(pause)).toBe('{"type":"Pause"}');
    expect(JSON.stringify(resume)).toBe('{"type":"Resume"}');
    expect(JSON.stringify(setSpeed)).toBe('{"type":"SetSpeed","speed":1.5}');
    expect(JSON.stringify(setEob)).toBe(
      '{"type":"SetEndOfSampleBehavior","behavior":"auto-advance"}',
    );
    expect(JSON.stringify(restartSample)).toBe('{"type":"RestartSample"}');
    expect(JSON.stringify(startTraining)).toBe(
      '{"type":"StartTraining","config":{"epochs":10}}',
    );
    expect(JSON.stringify(stopTraining)).toBe('{"type":"StopTraining"}');
    expect(JSON.stringify(loadCheckpoint)).toBe(
      '{"type":"LoadCheckpoint","path":"/ckpt/best.pt"}',
    );
    expect(JSON.stringify(updateConfig)).toBe(
      '{"type":"UpdateConfig","config":{"lr":0.001}}',
    );
    expect(JSON.stringify(getWeights)).toBe('{"type":"GetWeights"}');
  });
});

describe('protocol: string-literal unions', () => {
  it('locks SimulationMode literals and ordering', () => {
    const modes: readonly SimulationMode[] = ['idle', 'inference', 'training'];
    expect(modes).toEqual(['idle', 'inference', 'training']);
  });

  it('locks EndOfSampleBehavior literals and ordering', () => {
    const behaviors: readonly EndOfSampleBehavior[] = [
      'auto-advance',
      'stop',
      'loop',
    ];
    expect(behaviors).toEqual(['auto-advance', 'stop', 'loop']);
  });
});
