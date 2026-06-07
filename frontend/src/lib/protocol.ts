// WebSocket protocol types - mirrors Rust definitions

export type SimulationMode = 'idle' | 'inference' | 'training';

export interface NeuronState {
  layer: number;
  index: number;
  membrane: number;
  spiking: boolean;
  spike_count: number;
}

export interface SynapseInfo {
  from_layer: number;
  from_index: number;
  to_layer: number;
  to_index: number;
  weight: number;
}

export interface NetworkTopology {
  layer_sizes: number[];
  total_neurons: number;
  synapses: SynapseInfo[];
}

// Server -> Client messages
export type ServerMessage =
  | AnimationFrame
  | TrainingUpdate
  | WeightMatrix
  | Status
  | ErrorMsg
  | Ack;

export interface AnimationFrame {
  type: 'AnimationFrame';
  neurons: NeuronState[];
  step: number;
  total_steps: number;
  sample_index: number;
  label: number;
  prediction: number;
  correct: boolean;
  output_spikes: number[];
  image_pixels: number[];
  image_size: number;
  paused: boolean;
}

export interface TrainingUpdate {
  type: 'TrainingUpdate';
  epoch: number;
  total_epochs: number;
  loss: number;
  train_accuracy: number;
  test_accuracy: number;
  learning_rate: number;
  best_test_accuracy: number;
}

export interface WeightMatrix {
  type: 'WeightMatrix';
  layer: string;
  data: number[];
  rows: number;
  cols: number;
  min: number;
  max: number;
}

export interface Status {
  type: 'Status';
  mode: SimulationMode;
  config: Record<string, unknown>;
  total_samples: number;
  checkpoint_loaded: string | null;
}

export interface ErrorMsg {
  type: 'Error';
  message: string;
}

export interface Ack {
  type: 'Ack';
  command: string;
}

// End of sample behavior
export type EndOfSampleBehavior = 'auto-advance' | 'stop' | 'loop';

// Pulse rendering style for the network visualization
export type PulseStyle = 'ball' | 'electricity';

// Client -> Server messages
export type ClientMessage =
  | { type: 'GetStatus' }
  | { type: 'NextSample' }
  | { type: 'PrevSample' }
  | { type: 'JumpToSample'; index: number }
  | { type: 'RandomSample' }
  | { type: 'Pause' }
  | { type: 'Resume' }
  | { type: 'SetSpeed'; speed: number }
  | { type: 'SetEndOfSampleBehavior'; behavior: EndOfSampleBehavior }
  | { type: 'RestartSample' }
  | { type: 'StartTraining'; config: Record<string, unknown> }
  | { type: 'StopTraining' }
  | { type: 'LoadCheckpoint'; path: string }
  | { type: 'UpdateConfig'; config: Record<string, unknown> }
  | { type: 'GetWeights' };
