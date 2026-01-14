//! WebSocket message protocol for web UI communication
//!
//! Defines all message types exchanged between the Rust backend and web frontend.

use serde::{Deserialize, Serialize};

use crate::config::Config;

/// Messages sent from server to client
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    /// Animation frame with neuron states (sent at ~30fps during simulation)
    AnimationFrame {
        neurons: Vec<NeuronState>,
        step: usize,
        total_steps: usize,
        sample_index: usize,
        label: u8,
        prediction: usize,
        correct: bool,
        output_spikes: Vec<u32>,
        image_pixels: Vec<f32>,
        image_size: usize,
        paused: bool,
    },

    /// Training progress update (sent per epoch)
    TrainingUpdate {
        epoch: usize,
        total_epochs: usize,
        loss: f64,
        train_accuracy: f64,
        test_accuracy: f64,
        learning_rate: f64,
        best_test_accuracy: f64,
    },

    /// Weight matrix snapshot for visualization
    WeightMatrix {
        layer: String,
        data: Vec<f32>,
        rows: usize,
        cols: usize,
        min: f32,
        max: f32,
    },

    /// System status and configuration
    Status {
        mode: SimulationMode,
        config: Config,
        total_samples: usize,
        checkpoint_loaded: Option<String>,
    },

    /// Error message
    Error { message: String },

    /// Confirmation of command receipt
    Ack { command: String },
}

/// Compact neuron state for efficient wire transfer
#[derive(Debug, Clone, Serialize)]
pub struct NeuronState {
    /// Layer index (0 = input, 1 = hidden, 2 = output)
    pub layer: usize,
    /// Neuron index within layer
    pub index: usize,
    /// Membrane potential normalized to 0-1
    pub membrane: f32,
    /// Whether neuron is currently spiking
    pub spiking: bool,
    /// Accumulated spike count (meaningful for output neurons)
    pub spike_count: u32,
}

/// Synapse state for visualization
#[derive(Debug, Clone, Serialize)]
pub struct SynapseState {
    /// Source neuron (layer, index)
    pub from: (usize, usize),
    /// Target neuron (layer, index)
    pub to: (usize, usize),
    /// Weight value
    pub weight: f32,
    /// Animation progress for spike propagation (0-1)
    pub propagation: f32,
}

/// Current simulation mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SimulationMode {
    /// No simulation running
    Idle,
    /// Running inference animation
    Inference,
    /// Training in progress
    Training,
}

/// Behavior when a sample finishes playing
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum EndOfSampleBehavior {
    /// Automatically advance to the next sample
    #[default]
    AutoAdvance,
    /// Stop and stay paused
    Stop,
    /// Loop the current sample
    Loop,
}

/// Messages sent from client to server
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    /// Request current status
    GetStatus,

    /// Navigation commands
    NextSample,
    PrevSample,
    JumpToSample { index: usize },
    RandomSample,

    /// Playback control
    Pause,
    Resume,
    SetSpeed { speed: f32 },
    SetEndOfSampleBehavior { behavior: EndOfSampleBehavior },
    RestartSample,

    /// Training control
    StartTraining { config: Config },
    StopTraining,

    /// Configuration
    LoadCheckpoint { path: String },
    UpdateConfig { config: Config },

    /// Request weight matrices
    GetWeights,
}

/// Initial handshake message with full network topology
#[derive(Debug, Clone, Serialize)]
pub struct NetworkTopology {
    /// Layer sizes (e.g., [49, 9, 10] for input -> hidden -> output)
    pub layer_sizes: Vec<usize>,
    /// Total number of neurons
    pub total_neurons: usize,
    /// Synapse data: (from_layer, from_idx, to_layer, to_idx, weight)
    pub synapses: Vec<SynapseInfo>,
}

/// Static synapse information (topology + weights)
#[derive(Debug, Clone, Serialize)]
pub struct SynapseInfo {
    pub from_layer: usize,
    pub from_index: usize,
    pub to_layer: usize,
    pub to_index: usize,
    pub weight: f32,
}
