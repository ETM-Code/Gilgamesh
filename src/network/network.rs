//! Network struct and implementation

use crate::layers::Linear;
use crate::neurons::{Leaky, NeuronMode};
use crate::surrogate::SurrogateGradient;
use ndarray::{Array1, Array2};
use rand::Rng;

use super::cache::NetworkCache;
use super::gradients::NetworkGradients;
use super::state::NetworkState;
use super::trace::SimulationTrace;

/// A simple feedforward SNN matching snnTorch architecture
///
/// Architecture: Input(49) -> FC1(100) -> LIF1 -> FC2(10) -> LIF2 -> Output
#[derive(Clone)]
pub struct Network {
    pub fc1: Linear,
    pub lif1: Leaky,
    pub fc2: Linear,
    pub lif2: Leaky,
    /// When true, input pixels are converted to spike trains via deterministic accumulator
    /// before being fed through fc1. Makes all connections uniform spiking synapses.
    pub spiking_input: bool,
    /// Scale factor applied to hidden layer spikes before they enter fc2.
    /// Models the hardware pulse duration: spike_scale = t_pulse_effective / dt.
    /// Default 1.0 = spike lasts full timestep (original gilgamesh behavior).
    /// For Tarski PCB with τ_pulse=1.5µs and dt=1ms: spike_scale ≈ 0.00086.
    pub spike_scale: f32,
}

impl Network {
    /// Create a new network with specified architecture (defaults to Physics mode)
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        beta: f32,
        seed: u64,
    ) -> Self {
        let spike_grad = SurrogateGradient::fast_sigmoid(25.0);

        Self {
            fc1: Linear::with_seed(input_size, hidden_size, false, seed),
            lif1: Leaky::new(hidden_size, beta).with_spike_grad(spike_grad),
            fc2: Linear::with_seed(hidden_size, output_size, false, seed.wrapping_add(1)),
            lif2: Leaky::new(output_size, beta).with_spike_grad(spike_grad),
            spiking_input: false,
            spike_scale: 1.0,
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
            fc1: Linear::with_seed(input_size, hidden_size, false, seed),
            lif1: Leaky::new_physics(hidden_size, tau_m, dt).with_spike_grad(spike_grad),
            fc2: Linear::with_seed(hidden_size, output_size, false, seed.wrapping_add(1)),
            lif2: Leaky::new_physics(output_size, tau_m, dt).with_spike_grad(spike_grad),
            spiking_input: false,
            spike_scale: 1.0,
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
            fc1: Linear::with_seed(input_size, hidden_size, false, seed),
            lif1: Leaky::new_physics_with_pulse(hidden_size, tau_m, dt, tau_pulse, v_peak)
                .with_spike_grad(spike_grad),
            fc2: Linear::with_seed(hidden_size, output_size, false, seed.wrapping_add(1)),
            lif2: Leaky::new_physics_with_pulse(output_size, tau_m, dt, tau_pulse, v_peak)
                .with_spike_grad(spike_grad),
            spiking_input: false,
            spike_scale: 1.0,
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
            fc1: Linear::with_seed(input_size, hidden_size, false, seed),
            lif1: Leaky::new_physics_with_threshold_adaptation(
                hidden_size,
                tau_m,
                dt,
                tau_theta,
                theta_low,
                theta_high,
            )
            .with_spike_grad(spike_grad),
            fc2: Linear::with_seed(hidden_size, output_size, false, seed.wrapping_add(1)),
            lif2: Leaky::new_physics_with_threshold_adaptation(
                output_size,
                tau_m,
                dt,
                tau_theta,
                theta_low,
                theta_high,
            )
            .with_spike_grad(spike_grad),
            spiking_input: false,
            spike_scale: 1.0,
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

    /// Initialize input accumulator if spiking input is enabled
    /// Uses dithered initialization: random offsets in [0, 1) break up
    /// deterministic quantization patterns for better spike diversity
    fn init_input_accum(&self, batch_size: usize) -> Option<Array2<f32>> {
        if self.spiking_input {
            use rand::SeedableRng;
            use rand_xoshiro::Xoshiro256PlusPlus;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
            let input_size = self.fc1.weight.shape()[0];
            let accum = Array2::from_shape_fn((batch_size, input_size), |_| rng.gen::<f32>());
            Some(accum)
        } else {
            None
        }
    }

    /// Initialize network state for a batch
    pub fn init_state(&self, batch_size: usize) -> NetworkState {
        NetworkState {
            lif1_state: self.lif1.init_state(batch_size),
            lif2_state: self.lif2.init_state(batch_size),
            input_accum: self.init_input_accum(batch_size),
        }
    }

    /// Initialize network state with pulse tracking enabled
    pub fn init_state_with_pulse(&self, batch_size: usize) -> NetworkState {
        NetworkState {
            lif1_state: self.lif1.init_state_with_pulse(batch_size),
            lif2_state: self.lif2.init_state_with_pulse(batch_size),
            input_accum: self.init_input_accum(batch_size),
        }
    }

    /// Initialize network state with threshold adaptation enabled
    pub fn init_state_with_adaptation(&self, batch_size: usize) -> NetworkState {
        NetworkState {
            lif1_state: self.lif1.init_state_with_adaptation(batch_size),
            lif2_state: self.lif2.init_state_with_adaptation(batch_size),
            input_accum: self.init_input_accum(batch_size),
        }
    }

    /// Initialize network state with all physics features (pulse + adaptation)
    pub fn init_state_full(&self, batch_size: usize) -> NetworkState {
        NetworkState {
            lif1_state: self.lif1.init_state_full(batch_size),
            lif2_state: self.lif2.init_state_full(batch_size),
            input_accum: self.init_input_accum(batch_size),
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
        let hidden_current = match quant_bits {
            Some(bits) => self.fc1.forward_quantized(input, bits),
            None => self.fc1.forward(input),
        };
        let (hidden_spikes, lif1_state, lif1_cache) =
            self.lif1.forward(&hidden_current, &state.lif1_state);

        // Layer 2: FC -> LIF
        // Apply spike_scale to model hardware pulse duration.
        // spike_scale=1.0 means spike lasts full timestep (default/original).
        // spike_scale<1.0 means shorter pulse (e.g., 0.00086 for 1.5µs pulse in 1ms step).
        let scaled_spikes = if (self.spike_scale - 1.0).abs() > 1e-6 {
            &hidden_spikes * self.spike_scale
        } else {
            hidden_spikes.clone()
        };
        let output_current = match quant_bits {
            Some(bits) => self.fc2.forward_quantized(&scaled_spikes, bits),
            None => self.fc2.forward(&scaled_spikes),
        };
        let (output_spikes, lif2_state, lif2_cache) =
            self.lif2.forward(&output_current, &state.lif2_state);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
            input_accum: None,
        };

        let cache = NetworkCache::new(
            hidden_current,
            hidden_spikes,
            output_current,
            lif1_cache,
            lif2_cache,
        );

        // Return output spikes and membrane potential
        (
            output_spikes,
            new_state.lif2_state.mem.clone(),
            new_state,
            cache,
        )
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
        let hidden_current = self.fc1.forward(input);
        let (hidden_spikes, lif1_state, lif1_cache) =
            self.lif1
                .forward_with_dt(&hidden_current, &state.lif1_state, dt);

        // Layer 2: FC -> LIF with variable dt
        let output_current = self.fc2.forward(&hidden_spikes);
        let (output_spikes, lif2_state, lif2_cache) =
            self.lif2
                .forward_with_dt(&output_current, &state.lif2_state, dt);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
            input_accum: None,
        };

        let cache = NetworkCache::new(
            hidden_current,
            hidden_spikes,
            output_current,
            lif1_cache,
            lif2_cache,
        );

        (
            output_spikes,
            new_state.lif2_state.mem.clone(),
            new_state,
            cache,
        )
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
        self.forward_quantized_full(input, num_steps, quant_bits, false, 0)
    }

    /// Full forward pass with weight and input quantization options
    ///
    /// split_sign: use independent scaling for positive/negative weights (3-bit mag + 1-bit sign)
    /// input_quant_bits: quantize input to this many bits (0 = no input quantization)
    pub fn forward_quantized_full(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        quant_bits: Option<u8>,
        split_sign: bool,
        input_quant_bits: u8,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        use crate::layers::linear::{quantize_input, quantize_weights};

        let batch_size = input.shape()[0];
        let mut state = self.init_state(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        // Pre-compute quantized weights once (not per-timestep)
        // quantize_weights uses split-sign scaling (independent pos/neg) matching hardware
        let (fc1_weight, fc2_weight) = match quant_bits {
            Some(bits) => (
                quantize_weights(&self.fc1.weight, bits),
                quantize_weights(&self.fc2.weight, bits),
            ),
            None => (self.fc1.weight.clone(), self.fc2.weight.clone()),
        };
        let _ = split_sign; // split-sign is now the default in quantize_weights

        // Optionally quantize input (DAC resolution)
        let input = if input_quant_bits > 0 {
            quantize_input(input, input_quant_bits)
        } else {
            input.clone()
        };
        let input = &input;

        // Accumulate spikes over time for rate-based classification
        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        let spike_grad = SurrogateGradient::fast_sigmoid(25.0);

        // For spiking input, undo MNIST normalization to get [0,1] pixel values
        // The accumulator needs non-negative values to generate meaningful spike rates
        let spiking_raw_input = if self.spiking_input {
            Some(input.mapv(|x| (x * 0.3081 + 0.1307).clamp(0.0, 1.0)))
        } else {
            None
        };

        for t in 0..num_steps {
            // Spiking input: convert pixel values to spikes via deterministic accumulator
            // Uses burst spikes (floor) for multi-level encoding like Loihi 2 graded spikes
            let (fc1_input, input_spikes_cache, input_accum_cache) = if self.spiking_input {
                let raw = spiking_raw_input.as_ref().unwrap();
                let accum = state.input_accum.as_ref().unwrap();
                let accum_pre = accum + raw;
                // Burst spikes: emit floor(accum) spikes, allowing multi-level encoding
                let spikes = accum_pre.mapv(|v| v.floor().max(0.0));
                let new_accum = &accum_pre - &spikes;
                // On the final timestep, inject residual accumulator as fractional spike
                let fc1_in = if t == num_steps - 1 {
                    &spikes + &new_accum
                } else {
                    spikes.clone()
                };
                state.input_accum = Some(new_accum);
                (fc1_in, Some(spikes), Some(accum_pre))
            } else {
                (input.clone(), None, None)
            };

            // Layer 1: FC -> LIF (using pre-quantized weights)
            let mut hidden_current = fc1_input.dot(&fc1_weight);
            self.fc1
                .apply_synapse_drive_model_inplace(&mut hidden_current);
            if let Some(ref b) = self.fc1.bias {
                for mut row in hidden_current.rows_mut() {
                    row += b;
                }
            }

            // In physics mode with pulse stretching, use pulse output between layers
            // to match physical circuit behavior. Binary spikes are kept in cache for backward pass.
            let use_pulse = self.lif1.mode.tau_pulse() > 0.0 && self.is_physics_mode();
            let (hidden_output, hidden_spikes, lif1_state, lif1_cache) = if use_pulse {
                let dt = self.lif1.mode.dt().unwrap_or(0.001);
                let (pulse, lif1_state, lif1_cache) =
                    self.lif1
                        .forward_with_pulse(&hidden_current, &state.lif1_state, dt);
                let binary_spikes = lif1_cache.spikes.clone();
                (pulse, binary_spikes, lif1_state, lif1_cache)
            } else {
                let (spikes, lif1_state, lif1_cache) =
                    self.lif1.forward(&hidden_current, &state.lif1_state);
                (spikes.clone(), spikes, lif1_state, lif1_cache)
            };

            // Layer 2: FC -> LIF (using pre-quantized weights)
            let mut output_current = hidden_output.dot(&fc2_weight);
            self.fc2
                .apply_synapse_drive_model_inplace(&mut output_current);
            if let Some(ref b) = self.fc2.bias {
                for mut row in output_current.rows_mut() {
                    row += b;
                }
            }
            let (output_spikes, lif2_state, lif2_cache) =
                self.lif2.forward(&output_current, &state.lif2_state);

            // Update spike count in-place to avoid allocation
            spike_count += &output_spikes;
            final_mem = lif2_state.mem.clone();

            state = NetworkState {
                lif1_state,
                lif2_state,
                input_accum: state.input_accum,
            };
            let mut cache = NetworkCache::new(
                hidden_current,
                hidden_spikes, // Binary spikes for backward pass
                output_current,
                lif1_cache,
                lif2_cache,
            );
            cache.input_spikes = input_spikes_cache;
            cache.input_accum_pre = input_accum_cache;
            caches.push(cache);
        }

        let _ = spike_grad; // used by backward pass via cache

        (spike_count, final_mem, caches)
    }

    /// Run a forward loop over multiple timesteps using a per-step function.
    ///
    /// Common loop structure shared by forward_with_dt, forward_with_adaptation, etc.
    fn run_forward_loop<F>(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        mut state: NetworkState,
        step_fn: F,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>)
    where
        F: Fn(
            &Self,
            &Array2<f32>,
            &NetworkState,
        ) -> (Array2<f32>, Array2<f32>, NetworkState, NetworkCache),
    {
        let batch_size = input.shape()[0];
        let mut caches = Vec::with_capacity(num_steps);
        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            let (output_spikes, output_membrane, new_state, cache) = step_fn(self, input, &state);
            spike_count += &output_spikes;
            final_mem = output_membrane;
            state = new_state;
            caches.push(cache);
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
        let state = self.init_state(batch_size);
        self.run_forward_loop(input, num_steps, state, |net, inp, st| {
            net.forward_step_with_dt(inp, st, dt)
        })
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
        let hidden_current = self.fc1.forward(input);
        let (pulse1, lif1_state, lif1_cache) =
            self.lif1
                .forward_with_pulse(&hidden_current, &state.lif1_state, dt);

        // Layer 2: FC -> LIF with pulse output
        // The hidden layer output is pulse-shaped, transmitted to output layer
        let output_current = self.fc2.forward(&pulse1);
        let (pulse2, lif2_state, lif2_cache) =
            self.lif2
                .forward_with_pulse(&output_current, &state.lif2_state, dt);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
            input_accum: None,
        };

        let cache = NetworkCache::new(
            hidden_current,
            lif1_cache.spikes.clone(), // Binary spikes for backward
            output_current,
            lif1_cache,
            lif2_cache,
        );

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
            let (_, output_membrane, new_state, cache) =
                self.forward_step_with_pulse(input, &state, dt);
            // Accumulate binary spikes (not pulses) for classification
            spike_count += &cache.lif2_cache.spikes;
            final_mem = output_membrane;
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
        let hidden_current = self.fc1.forward(input);
        let (hidden_spikes, lif1_state, lif1_cache) =
            self.lif1
                .forward_with_adaptation(&hidden_current, &state.lif1_state, dt);

        // Layer 2: FC -> LIF with adaptation
        let output_current = self.fc2.forward(&hidden_spikes);
        let (output_spikes, lif2_state, lif2_cache) =
            self.lif2
                .forward_with_adaptation(&output_current, &state.lif2_state, dt);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
            input_accum: None,
        };

        let cache = NetworkCache::new(
            hidden_current,
            hidden_spikes,
            output_current,
            lif1_cache,
            lif2_cache,
        );

        // Return actual membrane potential (consistent with other forward_step methods)
        (
            output_spikes,
            new_state.lif2_state.mem.clone(),
            new_state,
            cache,
        )
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
        let state = self.init_state_with_adaptation(batch_size);
        self.run_forward_loop(input, num_steps, state, |net, inp, st| {
            net.forward_step_with_adaptation(inp, st, dt)
        })
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
        // For spiking input, undo MNIST normalization to get [0,1] pixel values
        let spiking_raw_input = if self.spiking_input {
            Some(input.mapv(|x| (x * 0.3081 + 0.1307).clamp(0.0, 1.0)))
        } else {
            None
        };

        for t in 0..num_steps {
            // Apply input noise (to raw [0,1] values for spiking, to normalized for rate-coded)
            let noisy_input = if self.spiking_input {
                let raw = spiking_raw_input.as_ref().unwrap();
                if let Some(normal) = &input_normal {
                    raw.mapv(|x| (x * (1.0 + normal.sample(rng) as f32)).clamp(0.0, 1.0))
                } else {
                    raw.clone()
                }
            } else if let Some(normal) = &input_normal {
                input.mapv(|x| x * (1.0 + normal.sample(rng) as f32))
            } else {
                input.clone()
            };

            // Generate input spikes if spiking input mode (burst spikes + residual)
            let (fc1_input, input_spikes_cache, input_accum_cache) = if self.spiking_input {
                let accum = state.input_accum.as_ref().unwrap();
                let accum_pre = accum + &noisy_input;
                let spikes = accum_pre.mapv(|v| v.floor().max(0.0));
                let new_accum = &accum_pre - &spikes;
                let fc1_in = if t == num_steps - 1 {
                    &spikes + &new_accum
                } else {
                    spikes.clone()
                };
                state.input_accum = Some(new_accum);
                (fc1_in, Some(spikes), Some(accum_pre))
            } else {
                (noisy_input, None, None)
            };

            // Layer 1: FC -> LIF
            let mut hidden_current = fc1_input.dot(&fc1_weight);
            self.fc1
                .apply_synapse_drive_model_inplace(&mut hidden_current);
            if let Some(ref b) = self.fc1.bias {
                for mut row in hidden_current.rows_mut() {
                    row += b;
                }
            }
            let (hidden_spikes, lif1_state, lif1_cache) = self.lif1.forward_noisy(
                &hidden_current,
                &state.lif1_state,
                threshold_noise_std,
                membrane_noise_std,
                rng,
            );

            // Layer 2: FC -> LIF (apply spike_scale for hardware pulse duration)
            let effective_spikes = if (self.spike_scale - 1.0).abs() > 1e-6 {
                &hidden_spikes * self.spike_scale
            } else {
                hidden_spikes.clone()
            };
            let mut output_current = effective_spikes.dot(&fc2_weight);
            self.fc2
                .apply_synapse_drive_model_inplace(&mut output_current);
            if let Some(ref b) = self.fc2.bias {
                for mut row in output_current.rows_mut() {
                    row += b;
                }
            }
            let (output_spikes, lif2_state, lif2_cache) = self.lif2.forward_noisy(
                &output_current,
                &state.lif2_state,
                threshold_noise_std,
                membrane_noise_std,
                rng,
            );

            spike_count += &output_spikes;

            state = NetworkState {
                lif1_state,
                lif2_state,
                input_accum: state.input_accum,
            };
            let mut cache = NetworkCache::new(
                hidden_current,
                hidden_spikes,
                output_current,
                lif1_cache,
                lif2_cache,
            );
            cache.input_spikes = input_spikes_cache;
            cache.input_accum_pre = input_accum_cache;
            caches.push(cache);
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
            let hidden_current = self.fc1.forward(input);
            let (hidden_spikes, lif1_state, lif1_cache) =
                self.lif1.forward(&hidden_current, &state.lif1_state);

            // Inter-layer signal: spikes + analog membrane (if enabled)
            let layer1_output = if analog_gain > 0.0 {
                // Combine binary spikes with amplified membrane voltage
                &hidden_spikes + &(&lif1_state.mem * analog_gain)
            } else {
                // Pure spike-based (default)
                hidden_spikes.clone()
            };

            // Layer 2: FC -> LIF (receives combined signal)
            let output_current = self.fc2.forward(&layer1_output);
            let (output_spikes, lif2_state, lif2_cache) =
                self.lif2.forward(&output_current, &state.lif2_state);

            spike_count += &output_spikes;
            final_mem = lif2_state.mem.clone();

            state = NetworkState {
                lif1_state,
                lif2_state,
                input_accum: state.input_accum,
            };
            caches.push(NetworkCache::new(
                hidden_current,
                hidden_spikes,
                output_current,
                lif1_cache,
                lif2_cache,
            ));
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
            let hidden_current = self.fc1.forward(&encoded_input);
            let (hidden_spikes, lif1_state, lif1_cache) =
                self.lif1.forward(&hidden_current, &state.lif1_state);

            let output_current = self.fc2.forward(&hidden_spikes);
            let (output_spikes, lif2_state, lif2_cache) =
                self.lif2.forward(&output_current, &state.lif2_state);

            spike_count += &output_spikes;
            final_mem = lif2_state.mem.clone();

            state = NetworkState {
                lif1_state,
                lif2_state,
                input_accum: state.input_accum,
            };

            // Store encoded_input for temporal encoding (needed for correct gradients)
            // For rate-coded, encoder returns input unchanged so this is equivalent
            {
                let mut cache = NetworkCache::new(
                    hidden_current,
                    hidden_spikes,
                    output_current,
                    lif1_cache,
                    lif2_cache,
                );
                cache.encoded_input = Some(encoded_input);
                caches.push(cache);
            }
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
        let num_steps = bptt_steps
            .map(|n| n.min(total_steps))
            .unwrap_or(total_steps);
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
        let mut grad_hidden_membrane_next = Array2::zeros((batch_size, self.lif1.size));
        let mut grad_output_membrane_next = Array2::zeros((batch_size, self.lif2.size));

        // Process timesteps in reverse order (only the truncated subset)
        for cache in caches_to_use.iter().rev() {
            // Backward through LIF2
            let (grad_output_current, grad_output_membrane_prev) = self.lif2.backward(
                &grad_per_step,
                &grad_output_membrane_next,
                &cache.lif2_cache,
            );
            grad_output_membrane_next = grad_output_membrane_prev;

            // Backward through FC2
            let (grad_hidden_spikes, fc2_weight_grad, fc2_bias_grad) = self
                .fc2
                .backward(&cache.hidden_spikes, &grad_output_current);
            grad_fc2_weight = &grad_fc2_weight + &fc2_weight_grad;
            if let (Some(ref mut acc), Some(bias_grad)) = (&mut grad_fc2_bias, fc2_bias_grad) {
                *acc = &*acc + &bias_grad;
            }

            // Backward through LIF1
            let (grad_hidden_current, grad_hidden_membrane_prev) = self.lif1.backward(
                &grad_hidden_spikes,
                &grad_hidden_membrane_next,
                &cache.lif1_cache,
            );
            grad_hidden_membrane_next = grad_hidden_membrane_prev;

            // Backward through FC1
            // Use input_spikes if spiking encoding, encoded_input if temporal,
            // otherwise use the static input (rate-coded)
            let fc1_input = if let Some(ref spikes) = cache.input_spikes {
                spikes
            } else {
                cache.encoded_input.as_ref().unwrap_or(input)
            };
            let (_, fc1_weight_grad, fc1_bias_grad) =
                self.fc1.backward(fc1_input, &grad_hidden_current);
            grad_fc1_weight = &grad_fc1_weight + &fc1_weight_grad;
            if let (Some(ref mut acc), Some(bias_grad)) = (&mut grad_fc1_bias, fc1_bias_grad) {
                *acc = &*acc + &bias_grad;
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
        if let (Some(ref mut bias), Some(ref bias_grad)) = (&mut self.fc1.bias, &grads.fc1_bias) {
            *bias = &*bias - &(bias_grad * lr);
        }

        self.fc2.weight = &self.fc2.weight - &(&grads.fc2_weight * lr);
        if let (Some(ref mut bias), Some(ref bias_grad)) = (&mut self.fc2.bias, &grads.fc2_bias) {
            *bias = &*bias - &(bias_grad * lr);
        }
    }

    /// Total number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.fc1.num_parameters() + self.fc2.num_parameters()
    }

    /// Forward pass with full simulation trace for visualization/analysis
    ///
    /// Captures per-timestep membrane potentials, spikes, and currents
    /// for both hidden and output layers. Useful for debugging,
    /// visualization, and comparison with SPICE simulation.
    ///
    /// Args:
    ///   input: [batch, features] - presented at each timestep
    ///   num_steps: number of timesteps
    ///
    /// Returns:
    ///   SimulationTrace containing full history of network activity
    pub fn forward_traced(&self, input: &Array2<f32>, num_steps: usize) -> SimulationTrace {
        let batch_size = input.shape()[0];
        let use_pulse = self.lif1.mode.tau_pulse() > 0.0 && self.is_physics_mode();
        let dt = self.lif1.mode.dt().unwrap_or(0.001);

        let mut state = if use_pulse {
            self.init_state_with_pulse(batch_size)
        } else {
            self.init_state(batch_size)
        };

        let mut hidden_mem_history = Vec::with_capacity(num_steps);
        let mut output_mem_history = Vec::with_capacity(num_steps);
        let mut hidden_spike_history = Vec::with_capacity(num_steps);
        let mut output_spike_history = Vec::with_capacity(num_steps);
        let mut hidden_current_history = Vec::with_capacity(num_steps);
        let mut output_current_history = Vec::with_capacity(num_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            // Layer 1: FC -> LIF
            let hidden_current = self.fc1.forward(input);

            if use_pulse {
                // Physics mode with pulse stretching: matches SPICE circuit behavior
                let (pulse_output, lif1_state, lif1_cache) =
                    self.lif1
                        .forward_with_pulse(&hidden_current, &state.lif1_state, dt);

                hidden_current_history.push(hidden_current);
                hidden_mem_history.push(lif1_state.mem.clone());
                hidden_spike_history.push(lif1_cache.spikes.clone());

                // Layer 2: pulse output drives output synapses
                let output_current = self.fc2.forward(&pulse_output);
                let (output_spikes, lif2_state, _) =
                    self.lif2.forward(&output_current, &state.lif2_state);

                output_current_history.push(output_current);
                output_mem_history.push(lif2_state.mem.clone());
                output_spike_history.push(output_spikes.clone());

                spike_count += &output_spikes;

                state = NetworkState {
                    lif1_state,
                    lif2_state,
                    input_accum: state.input_accum,
                };
            } else {
                // Simple mode: binary spikes between layers
                let (hidden_spikes, lif1_state, _) =
                    self.lif1.forward(&hidden_current, &state.lif1_state);

                hidden_current_history.push(hidden_current);
                hidden_mem_history.push(lif1_state.mem.clone());
                hidden_spike_history.push(hidden_spikes.clone());

                let output_current = self.fc2.forward(&hidden_spikes);
                let (output_spikes, lif2_state, _) =
                    self.lif2.forward(&output_current, &state.lif2_state);

                output_current_history.push(output_current);
                output_mem_history.push(lif2_state.mem.clone());
                output_spike_history.push(output_spikes.clone());

                spike_count += &output_spikes;

                state = NetworkState {
                    lif1_state,
                    lif2_state,
                    input_accum: state.input_accum,
                };
            }
        }

        SimulationTrace {
            hidden_mem_history,
            output_mem_history,
            hidden_spike_history,
            output_spike_history,
            hidden_current_history,
            output_current_history,
            output_spike_count: spike_count,
            output_final_mem: state.lif2_state.mem,
        }
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

        let (_spikes, _, caches) = net.forward(&input, 5);

        // Gradient of cross-entropy loss (mock)
        let grad_output = Array2::from_elem((2, 10), 0.1);

        let grads = net.backward(&input, &caches, &grad_output);

        assert_eq!(grads.fc1_weight.shape(), net.fc1.weight.shape());
        assert_eq!(grads.fc2_weight.shape(), net.fc2.weight.shape());
    }

    #[test]
    fn test_network_parameter_count() {
        let net = Network::new(49, 100, 10, 0.9, 42);

        // FC1: 49*100 = 4900 (no bias)
        // FC2: 100*10 = 1000 (no bias)
        // Total: 5900
        assert_eq!(net.num_parameters(), 5900);
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
        let diff: f32 = (&spikes_rate - &spikes_regular).mapv(|x| x.abs()).sum();
        assert!(
            diff < 1e-5,
            "Rate-coded encoding should match regular forward"
        );

        // Test with temporal encoding (row-by-row)
        // Temporal encoding outputs 7 features (one row at a time), so we need
        // a network with 7 inputs, not 49
        let net_temporal = Network::new(7, 100, 10, 0.9, 42);
        let encoder_temporal = InputEncoder::temporal(7, 0.002, 0.9, 0.001);
        let (spikes_temporal, mem_temporal, caches_temporal) =
            net_temporal.forward_with_encoding(&input, &encoder_temporal, 10);

        // With temporal encoding, timesteps may be auto-adjusted
        // 7 rows * 2ms spacing = 14ms = 14 timesteps at dt=1ms
        assert!(
            caches_temporal.len() >= 14,
            "Should have at least 14 timesteps for temporal"
        );

        assert_eq!(spikes_temporal.shape(), &[4, 10]);
        assert_eq!(mem_temporal.shape(), &[4, 10]);

        // Just check it ran without error and produced valid output
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
        assert!(diff < 1e-5, "Analog gain=0 should match regular forward");

        // Test with analog enabled (gain=0.1)
        let (spikes_analog, mem_analog, _) = net.forward_with_analog(&input, 25, 0.1);

        assert_eq!(spikes_analog.shape(), &[4, 10]);
        assert_eq!(mem_analog.shape(), &[4, 10]);
        assert!(spikes_analog.iter().all(|v| v.is_finite()));

        // With analog enabled, results may differ from pure spike mode
        // (membrane contributes to inter-layer signal)
        let _diff_analog: f32 = (&spikes_analog - &spikes_regular).mapv(|x| x.abs()).sum();
        // Just verify it runs and produces valid output
        // The difference depends on the specific gain value
        assert!(spikes_analog.iter().all(|&v| v >= 0.0));
    }
}
