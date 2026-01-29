//! Simulation bridge for web UI
//!
//! Manages network state and broadcasts updates to connected clients.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ndarray::{Array2, Axis};
use tokio::time::sleep;

use super::protocol::{EndOfSampleBehavior, NetworkTopology, NeuronState, ServerMessage, SimulationMode, SynapseInfo};
use super::server::AppState;
use crate::checkpoint::Checkpoint;
use crate::config::Config;
use crate::data::MnistDataset;
use crate::network::Network;
use crate::neurons::LeakyState;

/// Simulation state shared between server and simulation loop
pub struct SimulationState {
    /// Current mode
    pub mode: SimulationMode,
    /// Network configuration
    pub config: Config,
    /// Loaded network (if any)
    pub network: Option<Network>,
    /// Test images
    pub test_images: Option<Array2<f32>>,
    /// Test labels
    pub test_labels: Option<Vec<usize>>,
    /// Total number of test samples
    pub total_samples: usize,
    /// Current sample index
    pub current_sample: usize,
    /// Current simulation step
    pub current_step: usize,
    /// Total steps per sample
    pub total_steps: usize,
    /// Animation speed multiplier
    pub speed: f32,
    /// Is simulation paused
    pub paused: bool,
    /// Behavior when sample ends
    pub end_of_sample_behavior: EndOfSampleBehavior,
    /// Request to change sample
    pub sample_request: Option<SampleRequest>,
    /// Layer sizes [input, hidden, output]
    pub layer_sizes: Vec<usize>,
    /// Path to loaded checkpoint
    pub checkpoint_path: Option<String>,
    /// Accumulated output spikes for current sample
    pub output_spikes: Vec<u32>,
    /// Hidden layer neuron state
    hidden_state: Option<LeakyState>,
    /// Output layer neuron state
    output_state: Option<LeakyState>,
    /// Last hidden layer spikes (for visualization)
    last_hidden_spikes: Vec<bool>,
    /// Last output layer spikes (for visualization)
    last_output_spikes: Vec<bool>,
}

#[derive(Debug, Clone, Copy)]
pub enum SampleRequest {
    Next,
    Prev,
    Jump(usize),
    Random,
    Restart,
}

impl SimulationState {
    pub fn new() -> Self {
        Self {
            mode: SimulationMode::Idle,
            config: Config::default(),
            network: None,
            test_images: None,
            test_labels: None,
            total_samples: 0,
            current_sample: 0,
            current_step: 0,
            total_steps: 25,
            speed: 1.0,
            paused: false,
            end_of_sample_behavior: EndOfSampleBehavior::default(),
            sample_request: None,
            layer_sizes: vec![],
            checkpoint_path: None,
            output_spikes: vec![0; 10],
            hidden_state: None,
            output_state: None,
            last_hidden_spikes: vec![],
            last_output_spikes: vec![],
        }
    }

    /// Load a checkpoint and MNIST data
    pub fn load_checkpoint(&mut self, path: &Path, data_dir: &Path) -> anyhow::Result<()> {
        let cp = Checkpoint::load(path)?;
        let network = cp.to_network()?;

        let input_size = cp.architecture.input_size;
        let hidden_size = cp.architecture.hidden_size;
        let output_size = cp.architecture.output_size;

        self.layer_sizes = vec![input_size, hidden_size, output_size];
        self.config = Config::default();
        self.config.network.input_size = input_size;
        self.config.network.hidden_size = hidden_size;
        self.config.network.output_size = output_size;

        // Load MNIST if not already loaded
        if self.test_images.is_none() {
            let dataset = MnistDataset::load(data_dir.to_str().unwrap_or("./data"))?;
            self.test_images = Some(dataset.test_images);
            self.test_labels = Some(dataset.test_labels);
            self.total_samples = self.test_images.as_ref().map(|i| i.nrows()).unwrap_or(0);
        }

        // Initialize neuron states before moving network
        let hidden_state = network.lif1.init_state(1);
        let output_state = network.lif2.init_state(1);

        self.network = Some(network);
        self.checkpoint_path = Some(path.to_string_lossy().to_string());
        self.mode = SimulationMode::Inference;
        self.current_sample = 0;
        self.current_step = 0;
        self.output_spikes = vec![0; output_size];
        self.hidden_state = Some(hidden_state);
        self.output_state = Some(output_state);
        self.last_hidden_spikes = vec![false; hidden_size];
        self.last_output_spikes = vec![false; output_size];

        Ok(())
    }

    /// Get network topology for initial handshake
    pub fn get_topology(&self) -> Option<NetworkTopology> {
        let network = self.network.as_ref()?;

        let mut synapses = Vec::new();

        // FC1: input -> hidden
        for (to_idx, row) in network.fc1.weight.outer_iter().enumerate() {
            for (from_idx, &weight) in row.iter().enumerate() {
                synapses.push(SynapseInfo {
                    from_layer: 0,
                    from_index: from_idx,
                    to_layer: 1,
                    to_index: to_idx,
                    weight,
                });
            }
        }

        // FC2: hidden -> output
        for (to_idx, row) in network.fc2.weight.outer_iter().enumerate() {
            for (from_idx, &weight) in row.iter().enumerate() {
                synapses.push(SynapseInfo {
                    from_layer: 1,
                    from_index: from_idx,
                    to_layer: 2,
                    to_index: to_idx,
                    weight,
                });
            }
        }

        let total_neurons: usize = self.layer_sizes.iter().sum();

        Some(NetworkTopology {
            layer_sizes: self.layer_sizes.clone(),
            total_neurons,
            synapses,
        })
    }

    /// Get weight matrices for visualization
    pub fn get_weight_matrices(&self) -> Option<Vec<(String, Vec<f32>, usize, usize)>> {
        let network = self.network.as_ref()?;

        let fc1_data: Vec<f32> = network.fc1.weight.iter().cloned().collect();
        let fc1_shape = network.fc1.weight.dim();

        let fc2_data: Vec<f32> = network.fc2.weight.iter().cloned().collect();
        let fc2_shape = network.fc2.weight.dim();

        Some(vec![
            ("fc1".to_string(), fc1_data, fc1_shape.0, fc1_shape.1),
            ("fc2".to_string(), fc2_data, fc2_shape.0, fc2_shape.1),
        ])
    }

    /// Move to next sample
    pub fn next_sample(&mut self) {
        self.sample_request = Some(SampleRequest::Next);
    }

    /// Move to previous sample
    pub fn prev_sample(&mut self) {
        self.sample_request = Some(SampleRequest::Prev);
    }

    /// Jump to specific sample
    pub fn jump_to_sample(&mut self, index: usize) {
        self.sample_request = Some(SampleRequest::Jump(index));
    }

    /// Jump to random sample
    pub fn random_sample(&mut self) {
        self.sample_request = Some(SampleRequest::Random);
    }

    /// Restart current sample
    pub fn restart_sample(&mut self) {
        self.sample_request = Some(SampleRequest::Restart);
    }

    /// Set end-of-sample behavior
    pub fn set_end_of_sample_behavior(&mut self, behavior: EndOfSampleBehavior) {
        self.end_of_sample_behavior = behavior;
    }

    /// Reset for a new sample
    fn reset_for_sample(&mut self, sample_idx: usize) {
        self.current_sample = sample_idx;
        self.current_step = 0;
        let output_size = self.layer_sizes.get(2).copied().unwrap_or(10);
        let hidden_size = self.layer_sizes.get(1).copied().unwrap_or(9);
        self.output_spikes = vec![0; output_size];
        self.last_hidden_spikes = vec![false; hidden_size];
        self.last_output_spikes = vec![false; output_size];

        // Reset neuron states
        if let Some(ref net) = self.network {
            self.hidden_state = Some(net.lif1.init_state(1));
            self.output_state = Some(net.lif2.init_state(1));
        }
    }

    /// Get current image pixels
    pub fn get_image_pixels(&self) -> Vec<f32> {
        self.test_images
            .as_ref()
            .map(|images| images.row(self.current_sample).to_vec())
            .unwrap_or_default()
    }

    /// Get current label
    pub fn get_label(&self) -> usize {
        self.test_labels
            .as_ref()
            .and_then(|labels| labels.get(self.current_sample).copied())
            .unwrap_or(0)
    }
}

impl Default for SimulationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Main simulation loop - runs in background and broadcasts state
pub async fn run_simulation_loop(state: Arc<AppState>) {
    loop {
        // Get current mode and speed
        let (mode, speed) = {
            let sim = state.simulation.read().await;
            (sim.mode, sim.speed)
        };

        // Adjust frame rate based on speed (slower = longer between frames)
        let frame_delay = (100.0 / speed.max(0.1)) as u64; // Base ~100ms per step at 1x

        match mode {
            SimulationMode::Idle => {
                sleep(Duration::from_millis(100)).await;
            }

            SimulationMode::Inference => {
                run_inference_step(&state).await;
                sleep(Duration::from_millis(frame_delay)).await;
            }

            SimulationMode::Training => {
                sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// Run one step of inference simulation
async fn run_inference_step(state: &Arc<AppState>) {
    let mut sim = state.simulation.write().await;

    // Handle sample change requests
    if let Some(request) = sim.sample_request.take() {
        let new_sample = match request {
            SampleRequest::Next => (sim.current_sample + 1).min(sim.total_samples.saturating_sub(1)),
            SampleRequest::Prev => sim.current_sample.saturating_sub(1),
            SampleRequest::Jump(idx) => idx.min(sim.total_samples.saturating_sub(1)),
            SampleRequest::Random => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                rng.gen_range(0..sim.total_samples.max(1))
            }
            SampleRequest::Restart => sim.current_sample,
        };
        sim.reset_for_sample(new_sample);
    }

    // Check if paused
    if sim.paused {
        // Still broadcast current state even when paused
        broadcast_frame(&sim, state);
        return;
    }

    // Check if network is loaded and we have images
    let has_network = sim.network.is_some();
    let has_images = sim.test_images.is_some();
    if !has_network || !has_images {
        return;
    }

    let current_sample = sim.current_sample;
    let images = sim.test_images.as_ref().unwrap();
    if current_sample >= images.nrows() {
        return;
    }

    // Get sample data
    let sample_image = images.row(current_sample).to_owned();
    let sample_batch = sample_image.insert_axis(Axis(0));

    // Take states first to avoid borrow checker issues
    let hidden_state = sim.hidden_state.take();
    let output_state = sim.output_state.take();

    // Run forward pass
    let network = sim.network.as_ref().unwrap();

    // FC1 -> LIF1
    let fc1_out = network.fc1.forward(&sample_batch);
    let hidden_state = hidden_state.unwrap_or_else(|| network.lif1.init_state(1));
    let (hidden_spikes, new_hidden_state, _) = network.lif1.forward(&fc1_out, &hidden_state);

    // FC2 -> LIF2
    let fc2_out = network.fc2.forward(&hidden_spikes);
    let output_state = output_state.unwrap_or_else(|| network.lif2.init_state(1));
    let (output_spikes_arr, new_output_state, _) = network.lif2.forward(&fc2_out, &output_state);

    // Update state
    sim.hidden_state = Some(new_hidden_state);
    sim.output_state = Some(new_output_state);

    // Record spikes for visualization
    sim.last_hidden_spikes = hidden_spikes.row(0).iter().map(|&v| v > 0.5).collect();
    sim.last_output_spikes = output_spikes_arr.row(0).iter().map(|&v| v > 0.5).collect();

    // Accumulate output spikes
    for (i, &spike) in output_spikes_arr.row(0).iter().enumerate() {
        if spike > 0.5 && i < sim.output_spikes.len() {
            sim.output_spikes[i] += 1;
        }
    }

    sim.current_step += 1;
    let current_step = sim.current_step;
    let total_steps = sim.total_steps;

    // Broadcast current state
    broadcast_frame(&sim, state);

    // Check if sample is complete
    if current_step >= total_steps {
        let total_samples = sim.total_samples;
        let current_sample = sim.current_sample;
        let speed = sim.speed;

        // Drop the lock before sleeping
        drop(sim);

        // Longer pause between samples so user can see the result
        let pause_time = (2000.0 / speed.max(0.1)) as u64;
        sleep(Duration::from_millis(pause_time)).await;

        // Handle based on end-of-sample behavior (read fresh value after sleep)
        let mut sim = state.simulation.write().await;
        println!("End of sample reached. Behavior: {:?}", sim.end_of_sample_behavior);
        match sim.end_of_sample_behavior {
            EndOfSampleBehavior::AutoAdvance => {
                let next = (current_sample + 1) % total_samples.max(1);
                sim.reset_for_sample(next);
            }
            EndOfSampleBehavior::Stop => {
                println!("Stopping - setting paused=true");
                sim.paused = true;
                sim.reset_for_sample(current_sample);
            }
            EndOfSampleBehavior::Loop => {
                println!("Looping current sample");
                sim.reset_for_sample(current_sample);
            }
        }
    }
}

/// Broadcast current frame to all clients
fn broadcast_frame(sim: &SimulationState, state: &AppState) {
    if sim.test_images.is_none() || sim.network.is_none() {
        return;
    }

    // Build neuron states
    let mut neurons = Vec::new();

    // Input layer - use image pixels as "membrane"
    let image_pixels = sim.get_image_pixels();
    for (i, &pixel) in image_pixels.iter().enumerate() {
        neurons.push(NeuronState {
            layer: 0,
            index: i,
            membrane: pixel,
            spiking: pixel > 0.5,
            spike_count: 0,
        });
    }

    // Hidden layer
    if let Some(ref hidden_state) = sim.hidden_state {
        for (i, &mem) in hidden_state.mem.row(0).iter().enumerate() {
            let spiking = sim.last_hidden_spikes.get(i).copied().unwrap_or(false);
            neurons.push(NeuronState {
                layer: 1,
                index: i,
                membrane: mem.clamp(0.0, 1.0),
                spiking,
                spike_count: 0,
            });
        }
    }

    // Output layer
    if let Some(ref output_state) = sim.output_state {
        for (i, &mem) in output_state.mem.row(0).iter().enumerate() {
            let spiking = sim.last_output_spikes.get(i).copied().unwrap_or(false);
            neurons.push(NeuronState {
                layer: 2,
                index: i,
                membrane: mem.clamp(0.0, 1.0),
                spiking,
                spike_count: sim.output_spikes.get(i).copied().unwrap_or(0),
            });
        }
    }

    // Calculate prediction
    let prediction = sim
        .output_spikes
        .iter()
        .enumerate()
        .max_by_key(|(_, &count)| count)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let label = sim.get_label();
    let correct = prediction == label;

    // Determine image size from pixel count (6x6=36, 7x7=49, 28x28=784)
    let image_size = (image_pixels.len() as f32).sqrt() as usize;

    let msg = ServerMessage::AnimationFrame {
        neurons,
        step: sim.current_step,
        total_steps: sim.total_steps,
        sample_index: sim.current_sample,
        label: label as u8,
        prediction,
        correct,
        output_spikes: sim.output_spikes.clone(),
        image_pixels,
        image_size,
        paused: sim.paused,
    };

    state.broadcast(msg);
}
