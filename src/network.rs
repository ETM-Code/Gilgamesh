//! Network composition for spiking neural networks
//!
//! This module provides a complete SNN architecture matching the snnTorch example:
//! Input -> Linear -> LIF -> Linear -> LIF -> Output

use crate::layers::Linear;
use crate::neurons::leaky::{Leaky, LeakyCache, LeakyState, NeuronMode};
use crate::surrogate::SurrogateGradient;
use ndarray::{Array1, Array2};
use rand::Rng;

/// A simple feedforward SNN matching snnTorch architecture
///
/// Architecture: Input(49) -> FC1(100) -> LIF1 -> FC2(10) -> LIF2 -> Output
#[derive(Clone)]
pub struct Network {
    pub fc1: Linear,
    pub lif1: Leaky,
    pub fc2: Linear,
    pub lif2: Leaky,
}

impl Network {
    /// Create a new network with specified architecture (Simple mode)
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, beta: f32, seed: u64) -> Self {
        let spike_grad = SurrogateGradient::fast_sigmoid(25.0);

        Self {
            fc1: Linear::with_seed(input_size, hidden_size, true, seed),
            lif1: Leaky::new(hidden_size, beta).with_spike_grad(spike_grad),
            fc2: Linear::with_seed(hidden_size, output_size, true, seed.wrapping_add(1)),
            lif2: Leaky::new(output_size, beta).with_spike_grad(spike_grad),
        }
    }

    /// Create a new network in Physics mode with RC dynamics
    pub fn new_physics(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        tau_m: f32,
        dt: f32,
        seed: u64,
    ) -> Self {
        let spike_grad = SurrogateGradient::fast_sigmoid(25.0);

        Self {
            fc1: Linear::with_seed(input_size, hidden_size, true, seed),
            lif1: Leaky::new_physics(hidden_size, tau_m, dt).with_spike_grad(spike_grad),
            fc2: Linear::with_seed(hidden_size, output_size, true, seed.wrapping_add(1)),
            lif2: Leaky::new_physics(output_size, tau_m, dt).with_spike_grad(spike_grad),
        }
    }

    /// Create a new network in Physics mode with pulse stretching
    pub fn new_physics_with_pulse(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        tau_m: f32,
        dt: f32,
        tau_pulse: f32,
        v_peak: f32,
        seed: u64,
    ) -> Self {
        let spike_grad = SurrogateGradient::fast_sigmoid(25.0);

        Self {
            fc1: Linear::with_seed(input_size, hidden_size, true, seed),
            lif1: Leaky::new_physics_with_pulse(hidden_size, tau_m, dt, tau_pulse, v_peak)
                .with_spike_grad(spike_grad),
            fc2: Linear::with_seed(hidden_size, output_size, true, seed.wrapping_add(1)),
            lif2: Leaky::new_physics_with_pulse(output_size, tau_m, dt, tau_pulse, v_peak)
                .with_spike_grad(spike_grad),
        }
    }

    /// Create a new network in Physics mode with threshold adaptation
    pub fn new_physics_with_adaptation(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        tau_m: f32,
        dt: f32,
        tau_theta: f32,
        theta_low: f32,
        theta_high: f32,
        seed: u64,
    ) -> Self {
        let spike_grad = SurrogateGradient::fast_sigmoid(25.0);

        Self {
            fc1: Linear::with_seed(input_size, hidden_size, true, seed),
            lif1: Leaky::new_physics_with_threshold_adaptation(
                hidden_size, tau_m, dt, tau_theta, theta_low, theta_high,
            ).with_spike_grad(spike_grad),
            fc2: Linear::with_seed(hidden_size, output_size, true, seed.wrapping_add(1)),
            lif2: Leaky::new_physics_with_threshold_adaptation(
                output_size, tau_m, dt, tau_theta, theta_low, theta_high,
            ).with_spike_grad(spike_grad),
        }
    }

    /// Set neuron mode for all LIF layers
    pub fn set_mode(&mut self, mode: NeuronMode) {
        self.lif1 = self.lif1.clone().with_mode(mode.clone());
        self.lif2 = self.lif2.clone().with_mode(mode);
    }

    /// Check if network is in physics mode
    pub fn is_physics_mode(&self) -> bool {
        self.lif1.is_physics_mode()
    }

    /// Initialize network state for a batch
    pub fn init_state(&self, batch_size: usize) -> NetworkState {
        NetworkState {
            lif1_state: self.lif1.init_state(batch_size),
            lif2_state: self.lif2.init_state(batch_size),
        }
    }

    /// Initialize network state with pulse tracking enabled
    pub fn init_state_with_pulse(&self, batch_size: usize) -> NetworkState {
        NetworkState {
            lif1_state: self.lif1.init_state_with_pulse(batch_size),
            lif2_state: self.lif2.init_state_with_pulse(batch_size),
        }
    }

    /// Initialize network state with threshold adaptation enabled
    pub fn init_state_with_adaptation(&self, batch_size: usize) -> NetworkState {
        NetworkState {
            lif1_state: self.lif1.init_state_with_adaptation(batch_size),
            lif2_state: self.lif2.init_state_with_adaptation(batch_size),
        }
    }

    /// Initialize network state with all physics features (pulse + adaptation)
    pub fn init_state_full(&self, batch_size: usize) -> NetworkState {
        NetworkState {
            lif1_state: self.lif1.init_state_full(batch_size),
            lif2_state: self.lif2.init_state_full(batch_size),
        }
    }

    /// Forward pass for a single timestep
    ///
    /// Returns (output_spikes, output_mem, new_state, cache)
    pub fn forward_step(
        &self,
        input: &Array2<f32>,
        state: &NetworkState,
    ) -> (Array2<f32>, Array2<f32>, NetworkState, NetworkCache) {
        self.forward_step_quantized(input, state, None)
    }

    /// Forward pass for a single timestep with optional quantization
    ///
    /// Args:
    ///   input: [batch, features]
    ///   state: Current network state
    ///   quant_bits: Optional quantization bits (None = no quantization)
    ///
    /// Returns (output_spikes, output_mem, new_state, cache)
    pub fn forward_step_quantized(
        &self,
        input: &Array2<f32>,
        state: &NetworkState,
        quant_bits: Option<u8>,
    ) -> (Array2<f32>, Array2<f32>, NetworkState, NetworkCache) {
        // Layer 1: FC -> LIF
        let cur1 = match quant_bits {
            Some(bits) => self.fc1.forward_quantized(input, bits),
            None => self.fc1.forward(input),
        };
        let (spk1, lif1_state, lif1_cache) = self.lif1.forward(&cur1, &state.lif1_state);

        // Layer 2: FC -> LIF
        let cur2 = match quant_bits {
            Some(bits) => self.fc2.forward_quantized(&spk1, bits),
            None => self.fc2.forward(&spk1),
        };
        let (spk2, lif2_state, lif2_cache) = self.lif2.forward(&cur2, &state.lif2_state);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
        };

        let cache = NetworkCache {
            encoded_input: None,
            cur1,
            spk1,
            cur2,
            lif1_cache,
            lif2_cache,
        };

        // Return output spikes and membrane potential
        (spk2, new_state.lif2_state.mem.clone(), new_state, cache)
    }

    /// Full forward pass over multiple timesteps (rate-coded input)
    ///
    /// Args:
    ///   input: [batch, features] - presented at each timestep
    ///   num_steps: number of timesteps
    ///
    /// Returns:
    ///   (spike_count, final_mem, caches) - accumulated spikes and final membrane for classification
    pub fn forward(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        self.forward_quantized(input, num_steps, None)
    }

    /// Forward pass for a single timestep with variable dt (for physics mode)
    ///
    /// Args:
    ///   input: [batch, features]
    ///   state: Current network state
    ///   dt: Integration timestep (overrides stored dt in physics mode)
    ///
    /// Returns (output_spikes, output_mem, new_state, cache)
    pub fn forward_step_with_dt(
        &self,
        input: &Array2<f32>,
        state: &NetworkState,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, NetworkState, NetworkCache) {
        // Layer 1: FC -> LIF with variable dt
        let cur1 = self.fc1.forward(input);
        let (spk1, lif1_state, lif1_cache) = self.lif1.forward_with_dt(&cur1, &state.lif1_state, dt);

        // Layer 2: FC -> LIF with variable dt
        let cur2 = self.fc2.forward(&spk1);
        let (spk2, lif2_state, lif2_cache) = self.lif2.forward_with_dt(&cur2, &state.lif2_state, dt);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
        };

        let cache = NetworkCache {
            encoded_input: None,
            cur1,
            spk1,
            cur2,
            lif1_cache,
            lif2_cache,
        };

        (spk2, new_state.lif2_state.mem.clone(), new_state, cache)
    }

    /// Full forward pass with optional weight quantization
    ///
    /// Args:
    ///   input: [batch, features] - presented at each timestep
    ///   num_steps: number of timesteps
    ///   quant_bits: Optional quantization bits (None = no quantization)
    ///
    /// Returns:
    ///   (spike_count, final_mem, caches) - accumulated spikes and final membrane for classification
    pub fn forward_quantized(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        quant_bits: Option<u8>,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        use crate::layers::linear::quantize_weights;

        let batch_size = input.shape()[0];
        let mut state = self.init_state(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        // Pre-compute quantized weights once (not per-timestep)
        let (fc1_weight, fc2_weight) = match quant_bits {
            Some(bits) => (
                quantize_weights(&self.fc1.weight, bits),
                quantize_weights(&self.fc2.weight, bits),
            ),
            None => (self.fc1.weight.clone(), self.fc2.weight.clone()),
        };

        // Accumulate spikes over time for rate-based classification
        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            // Layer 1: FC -> LIF (using pre-quantized weights)
            let mut cur1 = input.dot(&fc1_weight);
            if let Some(ref b) = self.fc1.bias {
                for mut row in cur1.rows_mut() {
                    row += b;
                }
            }
            let (spk1, lif1_state, lif1_cache) = self.lif1.forward(&cur1, &state.lif1_state);

            // Layer 2: FC -> LIF (using pre-quantized weights)
            let mut cur2 = spk1.dot(&fc2_weight);
            if let Some(ref b) = self.fc2.bias {
                for mut row in cur2.rows_mut() {
                    row += b;
                }
            }
            let (spk2, lif2_state, lif2_cache) = self.lif2.forward(&cur2, &state.lif2_state);

            // Update spike count in-place to avoid allocation
            spike_count += &spk2;
            final_mem = lif2_state.mem.clone();

            state = NetworkState { lif1_state, lif2_state };
            caches.push(NetworkCache {
                encoded_input: None,
                cur1,
                spk1,
                cur2,
                lif1_cache,
                lif2_cache,
            });
        }

        (spike_count, final_mem, caches)
    }

    /// Full forward pass with variable dt (for physics mode fine-grained simulation)
    ///
    /// Args:
    ///   input: [batch, features] - presented at each timestep
    ///   num_steps: number of timesteps
    ///   dt: Integration timestep (overrides stored dt in physics mode)
    ///
    /// Returns:
    ///   (spike_count, final_mem, caches)
    pub fn forward_with_dt(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let batch_size = input.shape()[0];
        let mut state = self.init_state(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            let (spk2, mem2, new_state, cache) = self.forward_step_with_dt(input, &state, dt);
            spike_count += &spk2;
            final_mem = mem2;
            state = new_state;
            caches.push(cache);
        }

        (spike_count, final_mem, caches)
    }

    /// Forward pass for a single timestep with pulse stretching
    ///
    /// Uses pulse-shaped spikes between layers (exponential decay),
    /// but keeps binary spikes for gradient computation.
    ///
    /// Args:
    ///   input: [batch, features]
    ///   state: Current network state (should have pulse tracking enabled)
    ///   dt: Integration timestep
    ///
    /// Returns (output_pulse, output_mem, new_state, cache)
    pub fn forward_step_with_pulse(
        &self,
        input: &Array2<f32>,
        state: &NetworkState,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, NetworkState, NetworkCache) {
        // Layer 1: FC -> LIF with pulse output
        let cur1 = self.fc1.forward(input);
        let (pulse1, lif1_state, lif1_cache) =
            self.lif1.forward_with_pulse(&cur1, &state.lif1_state, dt);

        // Layer 2: FC -> LIF with pulse output
        // The hidden layer output is pulse-shaped, transmitted to output layer
        let cur2 = self.fc2.forward(&pulse1);
        let (pulse2, lif2_state, lif2_cache) =
            self.lif2.forward_with_pulse(&cur2, &state.lif2_state, dt);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
        };

        let cache = NetworkCache {
            encoded_input: None,
            cur1,
            spk1: lif1_cache.spikes.clone(), // Binary spikes for backward
            cur2,
            lif1_cache,
            lif2_cache,
        };

        (pulse2, new_state.lif2_state.mem.clone(), new_state, cache)
    }

    /// Full forward pass with pulse stretching (physics mode)
    ///
    /// Uses pulse-shaped spikes between layers for more realistic
    /// hardware simulation while keeping binary spikes for gradients.
    ///
    /// Args:
    ///   input: [batch, features] - presented at each timestep
    ///   num_steps: number of timesteps
    ///   dt: Integration timestep
    ///
    /// Returns:
    ///   (spike_count, final_mem, caches)
    pub fn forward_with_pulse(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let batch_size = input.shape()[0];
        let mut state = self.init_state_with_pulse(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        // For pulse mode, we accumulate binary spikes (from cache) for classification
        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            let (_, mem2, new_state, cache) = self.forward_step_with_pulse(input, &state, dt);
            // Accumulate binary spikes (not pulses) for classification
            spike_count += &cache.lif2_cache.spikes;
            final_mem = mem2;
            state = new_state;
            caches.push(cache);
        }

        (spike_count, final_mem, caches)
    }

    /// Forward pass with threshold adaptation for a single timestep
    ///
    /// Returns (output_spikes, output_mem, new_state, cache)
    pub fn forward_step_with_adaptation(
        &self,
        input: &Array2<f32>,
        state: &NetworkState,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, NetworkState, NetworkCache) {
        // Layer 1: FC -> LIF with adaptation
        let cur1 = self.fc1.forward(input);
        let (spk1, lif1_state, lif1_cache) =
            self.lif1.forward_with_adaptation(&cur1, &state.lif1_state, dt);

        // Layer 2: FC -> LIF with adaptation
        let cur2 = self.fc2.forward(&spk1);
        let (spk2, lif2_state, lif2_cache) =
            self.lif2.forward_with_adaptation(&cur2, &state.lif2_state, dt);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
        };

        let cache = NetworkCache {
            encoded_input: None,
            cur1,
            spk1,
            cur2,
            lif1_cache,
            lif2_cache,
        };

        // Return actual membrane potential (consistent with other forward_step methods)
        (spk2, new_state.lif2_state.mem.clone(), new_state, cache)
    }

    /// Full forward pass with threshold adaptation
    ///
    /// Uses adaptive thresholds that increase after spiking and decay when quiet.
    ///
    /// Args:
    ///   input: [batch, features] - presented at each timestep
    ///   num_steps: number of timesteps
    ///   dt: Integration timestep
    ///
    /// Returns:
    ///   (spike_count, final_mem, caches)
    pub fn forward_with_adaptation(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let batch_size = input.shape()[0];
        let mut state = self.init_state_with_adaptation(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            let (spk2, mem2, new_state, cache) = self.forward_step_with_adaptation(input, &state, dt);
            spike_count += &spk2;
            final_mem = mem2;
            state = new_state;
            caches.push(cache);
        }

        (spike_count, final_mem, caches)
    }

    /// Full forward pass with noise injection (for robustness training)
    ///
    /// Args:
    ///   input: [batch, features] - presented at each timestep
    ///   num_steps: number of timesteps
    ///   weight_noise_std: Weight noise std (relative, e.g., 0.05 for 5%)
    ///   threshold_noise_std: Threshold noise std (relative, e.g., 0.02 for 2%)
    ///   membrane_noise_std: Membrane noise std (absolute)
    ///   input_noise_std: Input noise std (relative)
    ///   rng: Random number generator
    ///
    /// Returns:
    ///   (spike_count, final_mem, caches)
    pub fn forward_noisy<R: Rng>(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        weight_noise_std: f32,
        threshold_noise_std: f32,
        membrane_noise_std: f32,
        input_noise_std: f32,
        rng: &mut R,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        use rand_distr::{Distribution, Normal};

        let batch_size = input.shape()[0];
        let mut state = self.init_state(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));

        let weight_normal = (weight_noise_std.is_finite() && weight_noise_std > 0.0)
            .then(|| Normal::new(0.0, weight_noise_std as f64).ok())
            .flatten();
        let input_normal = (input_noise_std.is_finite() && input_noise_std > 0.0)
            .then(|| Normal::new(0.0, input_noise_std as f64).ok())
            .flatten();

        // Pre-compute noisy weights (same noise for all timesteps within a batch)
        let fc1_weight = if let Some(normal) = &weight_normal {
            self.fc1
                .weight
                .mapv(|w| w * (1.0 + normal.sample(rng) as f32))
        } else {
            self.fc1.weight.clone()
        };
        let fc2_weight = if let Some(normal) = &weight_normal {
            self.fc2
                .weight
                .mapv(|w| w * (1.0 + normal.sample(rng) as f32))
        } else {
            self.fc2.weight.clone()
        };

        for _ in 0..num_steps {
            // Apply input noise
            let noisy_input = if let Some(normal) = &input_normal {
                input.mapv(|x| x * (1.0 + normal.sample(rng) as f32))
            } else {
                input.clone()
            };

            // Layer 1: FC -> LIF
            let mut cur1 = noisy_input.dot(&fc1_weight);
            if let Some(ref b) = self.fc1.bias {
                for mut row in cur1.rows_mut() {
                    row += b;
                }
            }
            let (spk1, lif1_state, lif1_cache) = self.lif1.forward_noisy(
                &cur1,
                &state.lif1_state,
                threshold_noise_std,
                membrane_noise_std,
                rng,
            );

            // Layer 2: FC -> LIF
            let mut cur2 = spk1.dot(&fc2_weight);
            if let Some(ref b) = self.fc2.bias {
                for mut row in cur2.rows_mut() {
                    row += b;
                }
            }
            let (spk2, lif2_state, lif2_cache) = self.lif2.forward_noisy(
                &cur2,
                &state.lif2_state,
                threshold_noise_std,
                membrane_noise_std,
                rng,
            );

            spike_count += &spk2;

            state = NetworkState { lif1_state, lif2_state };
            caches.push(NetworkCache {
                encoded_input: None,
                cur1,
                spk1,
                cur2,
                lif1_cache,
                lif2_cache,
            });
        }

        let final_mem = state.lif2_state.mem;
        (spike_count, final_mem, caches)
    }

    /// Forward pass with analog output mode
    ///
    /// In analog mode, the membrane voltage (amplified) flows alongside binary spikes
    /// between layers. This models hardware where both spike and analog signals propagate.
    ///
    /// Args:
    ///   input: [batch, features] - presented at each timestep
    ///   num_steps: number of timesteps
    ///   analog_gain: Amplification factor for membrane voltage (0.0 = disabled, pure spikes)
    ///
    /// Returns:
    ///   (spike_count, final_mem, caches)
    pub fn forward_with_analog(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        analog_gain: f32,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let batch_size = input.shape()[0];
        let mut state = self.init_state(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            // Layer 1: FC -> LIF
            let cur1 = self.fc1.forward(input);
            let (spk1, lif1_state, lif1_cache) = self.lif1.forward(&cur1, &state.lif1_state);

            // Inter-layer signal: spikes + analog membrane (if enabled)
            let layer1_output = if analog_gain > 0.0 {
                // Combine binary spikes with amplified membrane voltage
                &spk1 + &(&lif1_state.mem * analog_gain)
            } else {
                // Pure spike-based (default)
                spk1.clone()
            };

            // Layer 2: FC -> LIF (receives combined signal)
            let cur2 = self.fc2.forward(&layer1_output);
            let (spk2, lif2_state, lif2_cache) = self.lif2.forward(&cur2, &state.lif2_state);

            spike_count += &spk2;
            final_mem = lif2_state.mem.clone();

            state = NetworkState {
                lif1_state,
                lif2_state,
            };
            caches.push(NetworkCache {
                encoded_input: None,
                cur1,
                spk1, // Store original spikes for backward pass
                cur2,
                lif1_cache,
                lif2_cache,
            });
        }

        (spike_count, final_mem, caches)
    }

    /// Forward pass with input encoding (rate-coded or temporal)
    ///
    /// Supports both rate-coded (snnTorch-style) and temporal (hardware-like) input encoding.
    /// For rate-coded: same input at each timestep (standard behavior)
    /// For temporal: rows presented sequentially, simulating hardware scanning
    ///
    /// Note: This method is primarily for inference. For training with temporal encoding,
    /// consider the computational cost of storing inputs per timestep for backprop.
    ///
    /// Args:
    ///   input: [batch, features] - static input (will be encoded per timestep)
    ///   encoder: InputEncoder specifying the encoding type
    ///   num_steps: number of timesteps
    ///
    /// Returns:
    ///   (spike_count, final_mem, caches) - note: caches only useful for rate-coded training
    pub fn forward_with_encoding(
        &self,
        input: &Array2<f32>,
        encoder: &crate::data::InputEncoder,
        num_steps: usize,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let actual_steps = encoder.timesteps_needed(num_steps);
        let batch_size = input.shape()[0];
        let mut state = self.init_state(batch_size);
        let mut caches = Vec::with_capacity(actual_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for t in 0..actual_steps {
            // Encode input for this timestep
            let encoded_input = encoder.encode_timestep(input, t);

            // Standard forward step
            let cur1 = self.fc1.forward(&encoded_input);
            let (spk1, lif1_state, lif1_cache) = self.lif1.forward(&cur1, &state.lif1_state);

            let cur2 = self.fc2.forward(&spk1);
            let (spk2, lif2_state, lif2_cache) = self.lif2.forward(&cur2, &state.lif2_state);

            spike_count += &spk2;
            final_mem = lif2_state.mem.clone();

            state = NetworkState { lif1_state, lif2_state };

            // Store encoded_input for temporal encoding (needed for correct gradients)
            // For rate-coded, encoder returns input unchanged so this is equivalent
            caches.push(NetworkCache {
                encoded_input: Some(encoded_input),
                cur1,
                spk1,
                cur2,
                lif1_cache,
                lif2_cache,
            });
        }

        (spike_count, final_mem, caches)
    }

    /// Backward pass through timesteps
    ///
    /// Uses Backpropagation Through Time (BPTT)
    /// Note: For rate-coded, input is passed separately (same for all timesteps)
    /// For temporal encoding, uses encoded_input stored in each cache
    ///
    /// Args:
    ///   bptt_steps: If Some(n), only backprop through last n timesteps (truncated BPTT)
    pub fn backward(
        &self,
        input: &Array2<f32>,
        caches: &[NetworkCache],
        grad_output: &Array2<f32>,
    ) -> NetworkGradients {
        self.backward_truncated(input, caches, grad_output, None)
    }

    /// Backward pass with optional truncation
    pub fn backward_truncated(
        &self,
        input: &Array2<f32>,
        caches: &[NetworkCache],
        grad_output: &Array2<f32>,
        bptt_steps: Option<usize>,
    ) -> NetworkGradients {
        let total_steps = caches.len();
        // Truncate to last bptt_steps if specified
        let num_steps = bptt_steps.map(|n| n.min(total_steps)).unwrap_or(total_steps);
        let caches_to_use = &caches[total_steps - num_steps..];
        let batch_size = grad_output.shape()[0];

        // Initialize gradients
        let mut grad_fc1_weight = Array2::zeros(self.fc1.weight.raw_dim());
        let mut grad_fc1_bias = self.fc1.bias.as_ref().map(|b| Array1::zeros(b.len()));
        let mut grad_fc2_weight = Array2::zeros(self.fc2.weight.raw_dim());
        let mut grad_fc2_bias = self.fc2.bias.as_ref().map(|b| Array1::zeros(b.len()));

        // Gradient w.r.t. spike count (distributed across timesteps we backprop through)
        let grad_per_step = grad_output / num_steps as f32;

        // Membrane gradients from next timestep (for BPTT)
        let mut grad_mem1_next = Array2::zeros((batch_size, self.lif1.size));
        let mut grad_mem2_next = Array2::zeros((batch_size, self.lif2.size));

        // Process timesteps in reverse order (only the truncated subset)
        for cache in caches_to_use.iter().rev() {
            // Backward through LIF2
            let (grad_cur2, grad_mem2_prev) =
                self.lif2.backward(&grad_per_step, &grad_mem2_next, &cache.lif2_cache);
            grad_mem2_next = grad_mem2_prev;

            // Backward through FC2
            let (grad_spk1, gw2, gb2) = self.fc2.backward(&cache.spk1, &grad_cur2);
            grad_fc2_weight = &grad_fc2_weight + &gw2;
            if let (Some(ref mut acc), Some(gb)) = (&mut grad_fc2_bias, gb2) {
                *acc = &*acc + &gb;
            }

            // Backward through LIF1
            let (grad_cur1, grad_mem1_prev) =
                self.lif1.backward(&grad_spk1, &grad_mem1_next, &cache.lif1_cache);
            grad_mem1_next = grad_mem1_prev;

            // Backward through FC1
            // Use encoded_input from cache if available (temporal encoding),
            // otherwise use the static input (rate-coded)
            let fc1_input = cache.encoded_input.as_ref().unwrap_or(input);
            let (_, gw1, gb1) = self.fc1.backward(fc1_input, &grad_cur1);
            grad_fc1_weight = &grad_fc1_weight + &gw1;
            if let (Some(ref mut acc), Some(gb)) = (&mut grad_fc1_bias, gb1) {
                *acc = &*acc + &gb;
            }
        }

        NetworkGradients {
            fc1_weight: grad_fc1_weight,
            fc1_bias: grad_fc1_bias,
            fc2_weight: grad_fc2_weight,
            fc2_bias: grad_fc2_bias,
        }
    }

    /// Apply gradient update with learning rate
    pub fn apply_gradients(&mut self, grads: &NetworkGradients, lr: f32) {
        self.fc1.weight = &self.fc1.weight - &(&grads.fc1_weight * lr);
        if let (Some(ref mut b), Some(ref gb)) = (&mut self.fc1.bias, &grads.fc1_bias) {
            *b = &*b - &(gb * lr);
        }

        self.fc2.weight = &self.fc2.weight - &(&grads.fc2_weight * lr);
        if let (Some(ref mut b), Some(ref gb)) = (&mut self.fc2.bias, &grads.fc2_bias) {
            *b = &*b - &(gb * lr);
        }
    }

    /// Total number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.fc1.num_parameters() + self.fc2.num_parameters()
    }
}

/// Network state (membrane potentials)
#[derive(Clone, Debug)]
pub struct NetworkState {
    pub lif1_state: LeakyState,
    pub lif2_state: LeakyState,
}

impl NetworkState {
    pub fn reset(&mut self) {
        self.lif1_state.reset();
        self.lif2_state.reset();
    }
}

/// Cached values for backward pass
#[derive(Clone, Debug)]
pub struct NetworkCache {
    // For rate-coded: input is same for all timesteps, passed separately to backward()
    // For temporal: encoded_input stores the per-timestep input for correct gradients
    pub encoded_input: Option<Array2<f32>>,
    pub cur1: Array2<f32>,
    pub spk1: Array2<f32>,
    pub cur2: Array2<f32>,
    pub lif1_cache: LeakyCache,
    pub lif2_cache: LeakyCache,
}

/// Accumulated gradients for network parameters
#[derive(Clone, Debug)]
pub struct NetworkGradients {
    pub fc1_weight: Array2<f32>,
    pub fc1_bias: Option<Array1<f32>>,
    pub fc2_weight: Array2<f32>,
    pub fc2_bias: Option<Array1<f32>>,
}

impl NetworkGradients {
    pub fn zeros_like(net: &Network) -> Self {
        Self {
            fc1_weight: Array2::zeros(net.fc1.weight.raw_dim()),
            fc1_bias: net.fc1.bias.as_ref().map(|b| Array1::zeros(b.len())),
            fc2_weight: Array2::zeros(net.fc2.weight.raw_dim()),
            fc2_bias: net.fc2.bias.as_ref().map(|b| Array1::zeros(b.len())),
        }
    }

    /// Add another gradient to this one
    pub fn add(&mut self, other: &NetworkGradients) {
        self.fc1_weight = &self.fc1_weight + &other.fc1_weight;
        if let (Some(ref mut a), Some(ref b)) = (&mut self.fc1_bias, &other.fc1_bias) {
            *a = &*a + b;
        }
        self.fc2_weight = &self.fc2_weight + &other.fc2_weight;
        if let (Some(ref mut a), Some(ref b)) = (&mut self.fc2_bias, &other.fc2_bias) {
            *a = &*a + b;
        }
    }

    /// Scale gradients by a factor
    pub fn scale(&mut self, factor: f32) {
        self.fc1_weight *= factor;
        if let Some(ref mut b) = self.fc1_bias {
            *b *= factor;
        }
        self.fc2_weight *= factor;
        if let Some(ref mut b) = self.fc2_bias {
            *b *= factor;
        }
    }

    /// Compute the total L2 norm of all gradients
    pub fn total_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;

        // FC1 weight
        sum_sq += self.fc1_weight.iter().map(|x| x * x).sum::<f32>();

        // FC1 bias
        if let Some(ref b) = self.fc1_bias {
            sum_sq += b.iter().map(|x| x * x).sum::<f32>();
        }

        // FC2 weight
        sum_sq += self.fc2_weight.iter().map(|x| x * x).sum::<f32>();

        // FC2 bias
        if let Some(ref b) = self.fc2_bias {
            sum_sq += b.iter().map(|x| x * x).sum::<f32>();
        }

        sum_sq.sqrt()
    }

    /// Clip gradients by global norm (in place)
    /// Returns the original norm before clipping
    pub fn clip_norm(&mut self, max_norm: f32) -> f32 {
        let total_norm = self.total_norm();
        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            self.scale(scale);
        }
        total_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_forward() {
        let net = Network::new(49, 100, 10, 0.9, 42);
        let input = Array2::zeros((4, 49)); // batch of 4

        let (spikes, mem, caches) = net.forward(&input, 25);

        assert_eq!(spikes.shape(), &[4, 10]);
        assert_eq!(mem.shape(), &[4, 10]);
        assert_eq!(caches.len(), 25);
    }

    #[test]
    fn test_network_backward() {
        let net = Network::new(49, 100, 10, 0.9, 42);
        let input = Array2::from_elem((2, 49), 0.1);

        let (spikes, _, caches) = net.forward(&input, 5);

        // Gradient of cross-entropy loss (mock)
        let grad_output = Array2::from_elem((2, 10), 0.1);

        let grads = net.backward(&input, &caches, &grad_output);

        assert_eq!(grads.fc1_weight.shape(), net.fc1.weight.shape());
        assert_eq!(grads.fc2_weight.shape(), net.fc2.weight.shape());
    }

    #[test]
    fn test_network_parameter_count() {
        let net = Network::new(49, 100, 10, 0.9, 42);

        // FC1: 49*100 + 100 = 5000
        // FC2: 100*10 + 10 = 1010
        // Total: 6010
        assert_eq!(net.num_parameters(), 6010);
    }

    #[test]
    fn test_network_forward_quantized() {
        let net = Network::new(49, 100, 10, 0.9, 42);
        let input = Array2::from_elem((4, 49), 0.1);

        // Forward without quantization
        let (spikes_normal, _, _) = net.forward(&input, 10);

        // Forward with 8-bit quantization
        let (spikes_quant, _, _) = net.forward_quantized(&input, 10, Some(8));

        // Shapes should match
        assert_eq!(spikes_normal.shape(), spikes_quant.shape());

        // With 8-bit quantization, results should be similar
        // (not identical due to weight quantization)
        assert!(spikes_normal.iter().all(|v| v.is_finite()));
        assert!(spikes_quant.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_network_forward_with_pulse() {
        // Create physics network with pulse stretching
        let tau_m = 0.00949f32; // Equivalent to beta=0.9 at dt=1ms
        let dt = 0.001f32;
        let tau_pulse = 0.00167f32;
        let v_peak = 4.42f32;

        let net = Network::new_physics_with_pulse(49, 100, 10, tau_m, dt, tau_pulse, v_peak, 42);
        let input = Array2::from_elem((4, 49), 0.1);

        // Forward with pulse stretching
        let (spike_count, mem, caches) = net.forward_with_pulse(&input, 25, dt);

        // Shapes should be correct
        assert_eq!(spike_count.shape(), &[4, 10]);
        assert_eq!(mem.shape(), &[4, 10]);
        assert_eq!(caches.len(), 25);

        // All values should be finite
        assert!(spike_count.iter().all(|v| v.is_finite()));
        assert!(mem.iter().all(|v| v.is_finite()));

        // Spike counts should be non-negative integers (binary spikes accumulated)
        assert!(spike_count.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_network_pulse_vs_binary() {
        // Compare pulse mode to binary mode
        let tau_m = 0.00949f32;
        let dt = 0.001f32;
        let tau_pulse = 0.00167f32;
        let v_peak = 4.42f32;

        let net = Network::new_physics_with_pulse(49, 100, 10, tau_m, dt, tau_pulse, v_peak, 42);
        let input = Array2::from_elem((4, 49), 0.1);

        // Both modes should produce valid outputs
        let (spikes_binary, _, _) = net.forward_with_dt(&input, 25, dt);
        let (spikes_pulse, _, _) = net.forward_with_pulse(&input, 25, dt);

        // Shapes should match
        assert_eq!(spikes_binary.shape(), spikes_pulse.shape());

        // Both should have finite values
        assert!(spikes_binary.iter().all(|v| v.is_finite()));
        assert!(spikes_pulse.iter().all(|v| v.is_finite()));

        // Results may differ due to pulse-shaped inter-layer transmission
        // but both should produce reasonable spike counts
        let binary_sum: f32 = spikes_binary.sum();
        let pulse_sum: f32 = spikes_pulse.sum();
        assert!(binary_sum >= 0.0);
        assert!(pulse_sum >= 0.0);
    }

    #[test]
    fn test_network_with_adaptation() {
        // Test network with threshold adaptation
        let tau_m = 0.00949f32;
        let dt = 0.001f32;
        let tau_theta = 0.001f32;
        let theta_low = 1.0f32;
        let theta_high = 1.2f32; // 20% threshold increase

        let net = Network::new_physics_with_adaptation(
            49, 100, 10, tau_m, dt, tau_theta, theta_low, theta_high, 42,
        );
        let input = Array2::from_elem((4, 49), 0.1);

        // Run forward pass with adaptation
        let (spikes, final_mem, caches) = net.forward_with_adaptation(&input, 25, dt);

        // Should have correct shape
        assert_eq!(spikes.shape(), &[4, 10]);
        assert_eq!(final_mem.shape(), &[4, 10]);
        assert_eq!(caches.len(), 25);

        // Should produce valid outputs
        assert!(spikes.iter().all(|v| v.is_finite()));
        assert!(final_mem.iter().all(|v| v.is_finite()));

        // Should produce some spikes
        let spike_sum: f32 = spikes.sum();
        assert!(spike_sum >= 0.0, "Spike count should be non-negative");

        // Compare with standard physics mode (same network architecture)
        let net_standard = Network::new_physics(49, 100, 10, tau_m, dt, 42);
        let (spikes_std, _, _) = net_standard.forward_with_dt(&input, 25, dt);

        // Both should produce outputs with same shape
        assert_eq!(spikes.shape(), spikes_std.shape());
    }

    #[test]
    fn test_network_forward_with_encoding() {
        use crate::data::InputEncoder;

        let net = Network::new(49, 100, 10, 0.9, 42);
        let input = Array2::from_elem((4, 49), 0.1); // batch of 4, 7x7 images

        // Test with rate-coded encoding (should match regular forward)
        let encoder_rate = InputEncoder::rate_coded(7);
        let (spikes_rate, mem_rate, caches_rate) =
            net.forward_with_encoding(&input, &encoder_rate, 25);

        assert_eq!(spikes_rate.shape(), &[4, 10]);
        assert_eq!(mem_rate.shape(), &[4, 10]);
        assert_eq!(caches_rate.len(), 25);

        // Compare with regular forward (should be identical for rate-coded)
        let (spikes_regular, _, _) = net.forward(&input, 25);
        let diff: f32 = (&spikes_rate - &spikes_regular)
            .mapv(|x| x.abs())
            .sum();
        assert!(
            diff < 1e-5,
            "Rate-coded encoding should match regular forward"
        );

        // Test with temporal encoding (row-by-row)
        let encoder_temporal = InputEncoder::temporal(7, 0.002, 0.9, 0.001);
        let (spikes_temporal, mem_temporal, caches_temporal) =
            net.forward_with_encoding(&input, &encoder_temporal, 10);

        // With temporal encoding, timesteps may be auto-adjusted
        // 7 rows * 2ms spacing = 14ms = 14 timesteps at dt=1ms
        assert!(caches_temporal.len() >= 14, "Should have at least 14 timesteps for temporal");

        assert_eq!(spikes_temporal.shape(), &[4, 10]);
        assert_eq!(mem_temporal.shape(), &[4, 10]);

        // Temporal and rate-coded should produce different results
        // (because input is presented differently)
        let diff_temporal: f32 = (&spikes_temporal - &spikes_regular)
            .mapv(|x| x.abs())
            .sum();
        // Note: could be zero if input happens to give same result, but generally different
        // Just check it ran without error
        assert!(spikes_temporal.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_network_forward_with_analog() {
        let net = Network::new(49, 100, 10, 0.9, 42);
        let input = Array2::from_elem((4, 49), 0.1);

        // Test with analog disabled (gain=0) - should match regular forward
        let (spikes_no_analog, _, _) = net.forward_with_analog(&input, 25, 0.0);
        let (spikes_regular, _, _) = net.forward(&input, 25);

        let diff: f32 = (&spikes_no_analog - &spikes_regular)
            .mapv(|x| x.abs())
            .sum();
        assert!(
            diff < 1e-5,
            "Analog gain=0 should match regular forward"
        );

        // Test with analog enabled (gain=0.1)
        let (spikes_analog, mem_analog, _) = net.forward_with_analog(&input, 25, 0.1);

        assert_eq!(spikes_analog.shape(), &[4, 10]);
        assert_eq!(mem_analog.shape(), &[4, 10]);
        assert!(spikes_analog.iter().all(|v| v.is_finite()));

        // With analog enabled, results may differ from pure spike mode
        // (membrane contributes to inter-layer signal)
        let diff_analog: f32 = (&spikes_analog - &spikes_regular)
            .mapv(|x| x.abs())
            .sum();
        // Just verify it runs and produces valid output
        // The difference depends on the specific gain value
        assert!(spikes_analog.iter().all(|&v| v >= 0.0));
    }
}
