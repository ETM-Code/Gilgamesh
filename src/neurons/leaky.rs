//! Leaky Integrate-and-Fire (LIF) neuron
//!
//! Supports two modes:
//! - Simple: `mem[t+1] = beta * mem[t] + input` (snnTorch-compatible)
//! - Physics: `mem[t+1] = u_inf + (mem[t] - u_inf) * exp(-dt/tau)` (RC circuit)
//!
//! Where:
//!   - beta: membrane decay rate (0 to 1), equivalent to exp(-dt/tau)
//!   - threshold: spike threshold (default 1.0)
//!   - reset: 1 if spike occurred, else 0

use crate::surrogate::SurrogateGradient;
use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Reset mechanism for membrane potential after spike
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ResetMechanism {
    /// Subtract threshold from membrane: mem = mem - threshold
    Subtract,
    /// Reset membrane to zero: mem = 0
    Zero,
    /// No reset (pure integration)
    None,
}

impl Default for ResetMechanism {
    fn default() -> Self {
        ResetMechanism::Subtract
    }
}

/// Neuron computation mode
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NeuronMode {
    /// Simple discrete-time model (snnTorch-compatible)
    /// mem[t+1] = beta * mem[t] + input
    Simple,

    /// Physics-accurate RC circuit model
    /// mem[t+1] = u_inf + (mem[t] - u_inf) * exp(-dt/tau)
    /// where u_inf = input * tau (steady-state for constant input)
    Physics {
        /// Membrane time constant (seconds), computed from beta if not set
        tau_m: f32,
        /// Integration timestep (seconds)
        dt: f32,
        /// Pulse width time constant (seconds) for exponential decay output
        /// V_pulse(t) = V_peak * exp(-(t - t_spike) / tau_pulse)
        #[serde(default = "default_tau_pulse")]
        tau_pulse: f32,
        /// Peak pulse voltage (hardware: V_rail - drops)
        #[serde(default = "default_v_peak")]
        v_peak: f32,
        /// Threshold adaptation time constant (seconds)
        /// θ(t+dt) = θ_target + (θ(t) - θ_target) * exp(-dt/τ_θ)
        #[serde(default = "default_tau_theta")]
        tau_theta: f32,
        /// Threshold equilibrium when neuron is quiet (low activity)
        #[serde(default = "default_theta_low")]
        theta_low: f32,
        /// Threshold equilibrium when neuron is active (high activity)
        #[serde(default = "default_theta_high")]
        theta_high: f32,
        /// Minimum membrane voltage (hardware rail)
        #[serde(default = "default_v_min")]
        v_min: f32,
        /// Maximum membrane voltage (hardware rail)
        #[serde(default = "default_v_max")]
        v_max: f32,
        /// Comparator propagation delay (seconds)
        /// Models the time from threshold crossing to spike output
        /// Based on NCS2250: ~50ns typical
        #[serde(default = "default_comparator_delay")]
        comparator_delay_s: f32,
        /// Reset hold period (seconds)
        /// Time membrane is held at reset value after spike
        /// Models the pulse stretcher controlling the reset switch
        #[serde(default = "default_reset_hold")]
        reset_hold_s: f32,
    },
}

fn default_tau_pulse() -> f32 {
    0.00167 // 1.67ms from hardware
}

fn default_v_peak() -> f32 {
    4.42 // 5.0 - 0.21 - 0.37 from hardware
}

fn default_tau_theta() -> f32 {
    0.001 // 1ms threshold adaptation time constant
}

fn default_theta_low() -> f32 {
    1.0 // Base threshold when quiet
}

fn default_theta_high() -> f32 {
    1.2 // Elevated threshold after spiking (20% increase)
}

fn default_v_min() -> f32 {
    0.0 // Hardware ground rail
}

fn default_v_max() -> f32 {
    5.0 // Hardware supply rail
}

fn default_comparator_delay() -> f32 {
    0.0 // Default: no delay (instant, for backwards compatibility)
    // Set to 50e-9 (50ns) for realistic NCS2250 behavior
}

fn default_reset_hold() -> f32 {
    0.0 // Default: no hold (instant reset, for backwards compatibility)
    // Set to 0.35e-3 (0.35ms) for realistic pulse-stretcher controlled reset
}

impl Default for NeuronMode {
    fn default() -> Self {
        // Default to Physics mode for hardware-accurate simulation
        // Use sensible defaults matching typical passive RC neuron circuits
        NeuronMode::Physics {
            tau_m: 0.0012,       // 1.2ms (matches SPICE: 120kΩ * 10nF)
            dt: 1e-6,            // 1µs timestep
            tau_pulse: 0.5e-3,   // 0.5ms pulse stretch
            v_peak: 2.6,         // Peak with diode drop
            tau_theta: default_tau_theta(),
            theta_low: default_theta_low(),
            theta_high: default_theta_high(),
            v_min: default_v_min(),
            v_max: default_v_max(),
            comparator_delay_s: 50e-9,  // 50ns comparator delay
            reset_hold_s: 0.15e-3,      // 0.15ms reset hold
        }
    }
}

impl NeuronMode {
    /// Create physics mode from beta and dt
    /// Uses: beta = exp(-dt/tau), so tau = -dt/ln(beta)
    pub fn physics_from_beta(beta: f32, dt: f32) -> Self {
        let tau_m = if beta > 0.0 && beta < 1.0 {
            -dt / beta.ln()
        } else {
            dt * 10.0 // Fallback: 10x dt if beta is invalid
        };
        NeuronMode::Physics {
            tau_m,
            dt,
            tau_pulse: default_tau_pulse(),
            v_peak: default_v_peak(),
            tau_theta: default_tau_theta(),
            theta_low: default_theta_low(),
            theta_high: default_theta_high(),
            v_min: default_v_min(),
            v_max: default_v_max(),
            comparator_delay_s: default_comparator_delay(),
            reset_hold_s: default_reset_hold(),
        }
    }

    /// Create physics mode with pulse parameters (no threshold adaptation)
    pub fn physics(tau_m: f32, dt: f32, tau_pulse: f32, v_peak: f32) -> Self {
        NeuronMode::Physics {
            tau_m,
            dt,
            tau_pulse,
            v_peak,
            tau_theta: default_tau_theta(),
            theta_low: default_theta_low(),
            theta_high: default_theta_high(),
            v_min: default_v_min(),
            v_max: default_v_max(),
            comparator_delay_s: default_comparator_delay(),
            reset_hold_s: default_reset_hold(),
        }
    }

    /// Create physics mode with full parameters including threshold adaptation
    pub fn physics_full(
        tau_m: f32,
        dt: f32,
        tau_pulse: f32,
        v_peak: f32,
        tau_theta: f32,
        theta_low: f32,
        theta_high: f32,
    ) -> Self {
        NeuronMode::Physics {
            tau_m,
            dt,
            tau_pulse,
            v_peak,
            tau_theta,
            theta_low,
            theta_high,
            v_min: default_v_min(),
            v_max: default_v_max(),
            comparator_delay_s: default_comparator_delay(),
            reset_hold_s: default_reset_hold(),
        }
    }

    /// Create physics mode with hardware timing parameters
    pub fn physics_with_hardware_timing(
        tau_m: f32,
        dt: f32,
        tau_pulse: f32,
        v_peak: f32,
        comparator_delay_s: f32,
        reset_hold_s: f32,
    ) -> Self {
        NeuronMode::Physics {
            tau_m,
            dt,
            tau_pulse,
            v_peak,
            tau_theta: default_tau_theta(),
            theta_low: default_theta_low(),
            theta_high: default_theta_high(),
            v_min: default_v_min(),
            v_max: default_v_max(),
            comparator_delay_s,
            reset_hold_s,
        }
    }

    /// Get effective beta for this mode
    pub fn effective_beta(&self, default_beta: f32) -> f32 {
        match self {
            NeuronMode::Simple => default_beta,
            NeuronMode::Physics { tau_m, dt, .. } => (-dt / tau_m).exp(),
        }
    }

    /// Get tau_pulse (returns 0 for Simple mode)
    pub fn tau_pulse(&self) -> f32 {
        match self {
            NeuronMode::Simple => 0.0,
            NeuronMode::Physics { tau_pulse, .. } => *tau_pulse,
        }
    }

    /// Get v_peak (returns 1.0 for Simple mode)
    pub fn v_peak(&self) -> f32 {
        match self {
            NeuronMode::Simple => 1.0,
            NeuronMode::Physics { v_peak, .. } => *v_peak,
        }
    }

    /// Get tau_theta (returns 0 for Simple mode - no adaptation)
    pub fn tau_theta(&self) -> f32 {
        match self {
            NeuronMode::Simple => 0.0,
            NeuronMode::Physics { tau_theta, .. } => *tau_theta,
        }
    }

    /// Get theta_low (returns 1.0 for Simple mode)
    pub fn theta_low(&self) -> f32 {
        match self {
            NeuronMode::Simple => 1.0,
            NeuronMode::Physics { theta_low, .. } => *theta_low,
        }
    }

    /// Get theta_high (returns 1.0 for Simple mode - no adaptation)
    pub fn theta_high(&self) -> f32 {
        match self {
            NeuronMode::Simple => 1.0,
            NeuronMode::Physics { theta_high, .. } => *theta_high,
        }
    }

    /// Get v_min (returns 0.0 for Simple mode - no clamping)
    pub fn v_min(&self) -> f32 {
        match self {
            NeuronMode::Simple => f32::NEG_INFINITY, // No clamping in simple mode
            NeuronMode::Physics { v_min, .. } => *v_min,
        }
    }

    /// Get v_max (returns infinity for Simple mode - no clamping)
    pub fn v_max(&self) -> f32 {
        match self {
            NeuronMode::Simple => f32::INFINITY, // No clamping in simple mode
            NeuronMode::Physics { v_max, .. } => *v_max,
        }
    }

    /// Check if threshold adaptation is enabled
    pub fn has_threshold_adaptation(&self) -> bool {
        match self {
            NeuronMode::Simple => false,
            NeuronMode::Physics { theta_low, theta_high, .. } => {
                (theta_high - theta_low).abs() > 1e-6
            }
        }
    }

    /// Get tau_m (returns None for Simple mode)
    pub fn tau_m(&self) -> Option<f32> {
        match self {
            NeuronMode::Simple => None,
            NeuronMode::Physics { tau_m, .. } => Some(*tau_m),
        }
    }

    /// Get dt (returns None for Simple mode)
    pub fn dt(&self) -> Option<f32> {
        match self {
            NeuronMode::Simple => None,
            NeuronMode::Physics { dt, .. } => Some(*dt),
        }
    }

    /// Get comparator propagation delay (seconds)
    pub fn comparator_delay(&self) -> f32 {
        match self {
            NeuronMode::Simple => 0.0,
            NeuronMode::Physics { comparator_delay_s, .. } => *comparator_delay_s,
        }
    }

    /// Get reset hold period (seconds)
    pub fn reset_hold(&self) -> f32 {
        match self {
            NeuronMode::Simple => 0.0,
            NeuronMode::Physics { reset_hold_s, .. } => *reset_hold_s,
        }
    }

    /// Check if hardware timing is enabled (either delay or hold > 0)
    pub fn has_hardware_timing(&self) -> bool {
        self.comparator_delay() > 0.0 || self.reset_hold() > 0.0
    }
}

/// Leaky Integrate-and-Fire neuron layer
///
/// Supports two modes:
/// - Simple: mem = beta * mem + input (snnTorch-compatible)
/// - Physics: RC circuit with exp(-dt/tau) dynamics
///
/// Spike: S = Heaviside(mem - threshold)
/// Gradient: Uses surrogate gradient for dS/d(mem)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Leaky {
    /// Membrane decay rate (0 to 1). Higher = slower decay.
    pub beta: f32,
    /// Spike threshold
    pub threshold: f32,
    /// Surrogate gradient function for backprop
    pub spike_grad: SurrogateGradient,
    /// Reset mechanism
    pub reset_mechanism: ResetMechanism,
    /// Number of neurons in this layer
    pub size: usize,
    /// Computation mode (Simple or Physics)
    #[serde(default)]
    pub mode: NeuronMode,
}

impl Leaky {
    /// Create a new Leaky neuron layer (defaults to Physics mode)
    ///
    /// The beta parameter is used for backward compatibility but may be
    /// overridden by the Physics mode's tau_m/dt settings.
    pub fn new(size: usize, beta: f32) -> Self {
        Self {
            beta: beta.clamp(0.0, 1.0),
            threshold: 1.0,
            spike_grad: SurrogateGradient::fast_sigmoid(25.0),
            reset_mechanism: ResetMechanism::Subtract,
            size,
            mode: NeuronMode::default(), // Now defaults to Physics mode
        }
    }

    /// Create a new Leaky neuron layer with Simple mode (backward compatibility)
    ///
    /// Use this when you want the original snnTorch-style behavior without
    /// hardware-accurate physics simulation.
    pub fn new_simple(size: usize, beta: f32) -> Self {
        Self {
            beta: beta.clamp(0.0, 1.0),
            threshold: 1.0,
            spike_grad: SurrogateGradient::fast_sigmoid(25.0),
            reset_mechanism: ResetMechanism::Subtract,
            size,
            mode: NeuronMode::Simple,
        }
    }

    /// Create a new Leaky neuron layer in Physics mode
    pub fn new_physics(size: usize, tau_m: f32, dt: f32) -> Self {
        let beta = (-dt / tau_m).exp();
        Self {
            beta,
            threshold: 1.0,
            spike_grad: SurrogateGradient::fast_sigmoid(25.0),
            reset_mechanism: ResetMechanism::Subtract,
            size,
            mode: NeuronMode::Physics {
                tau_m,
                dt,
                tau_pulse: default_tau_pulse(),
                v_peak: default_v_peak(),
                tau_theta: default_tau_theta(),
                theta_low: default_theta_low(),
                theta_high: default_theta_high(),
                v_min: default_v_min(),
                v_max: default_v_max(),
                comparator_delay_s: default_comparator_delay(),
                reset_hold_s: default_reset_hold(),
            },
        }
    }

    /// Create a new Leaky neuron layer in Physics mode with pulse parameters
    pub fn new_physics_with_pulse(
        size: usize,
        tau_m: f32,
        dt: f32,
        tau_pulse: f32,
        v_peak: f32,
    ) -> Self {
        let beta = (-dt / tau_m).exp();
        Self {
            beta,
            threshold: 1.0,
            spike_grad: SurrogateGradient::fast_sigmoid(25.0),
            reset_mechanism: ResetMechanism::Subtract,
            size,
            mode: NeuronMode::Physics {
                tau_m,
                dt,
                tau_pulse,
                v_peak,
                tau_theta: default_tau_theta(),
                theta_low: default_theta_low(),
                theta_high: default_theta_high(),
                v_min: default_v_min(),
                v_max: default_v_max(),
                comparator_delay_s: default_comparator_delay(),
                reset_hold_s: default_reset_hold(),
            },
        }
    }

    /// Create physics-based leaky neuron with custom threshold adaptation parameters
    ///
    /// Uses continuous RC membrane dynamics with adaptive threshold.
    pub fn new_physics_with_threshold_adaptation(
        size: usize,
        tau_m: f32,
        dt: f32,
        tau_theta: f32,
        theta_low: f32,
        theta_high: f32,
    ) -> Self {
        let beta = (-dt / tau_m).exp();
        Self {
            beta,
            threshold: theta_low, // Base threshold is theta_low
            spike_grad: SurrogateGradient::fast_sigmoid(25.0),
            reset_mechanism: ResetMechanism::Subtract,
            size,
            mode: NeuronMode::Physics {
                tau_m,
                dt,
                tau_pulse: default_tau_pulse(),
                v_peak: default_v_peak(),
                tau_theta,
                theta_low,
                theta_high,
                v_min: default_v_min(),
                v_max: default_v_max(),
                comparator_delay_s: default_comparator_delay(),
                reset_hold_s: default_reset_hold(),
            },
        }
    }

    /// Set spike threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set surrogate gradient function
    pub fn with_spike_grad(mut self, spike_grad: SurrogateGradient) -> Self {
        self.spike_grad = spike_grad;
        self
    }

    /// Set reset mechanism
    pub fn with_reset_mechanism(mut self, reset_mechanism: ResetMechanism) -> Self {
        self.reset_mechanism = reset_mechanism;
        self
    }

    /// Set computation mode
    pub fn with_mode(mut self, mode: NeuronMode) -> Self {
        // Update beta to match mode if switching to physics
        if let NeuronMode::Physics { tau_m, dt, .. } = &mode {
            self.beta = (-dt / tau_m).exp();
        }
        self.mode = mode;
        self
    }

    /// Get tau_pulse for pulse stretching
    pub fn get_tau_pulse(&self) -> f32 {
        self.mode.tau_pulse()
    }

    /// Get v_peak for pulse stretching
    pub fn get_v_peak(&self) -> f32 {
        self.mode.v_peak()
    }

    /// Check if in physics mode
    pub fn is_physics_mode(&self) -> bool {
        matches!(self.mode, NeuronMode::Physics { .. })
    }

    /// Initialize membrane state for a batch (no pulse tracking or adaptation)
    pub fn init_state(&self, batch_size: usize) -> LeakyState {
        LeakyState {
            mem: Array2::zeros((batch_size, self.size)),
            time_since_spike: None,
            adaptive_threshold: None,
            pending_spike_steps: None,
            reset_hold_steps: None,
        }
    }

    /// Initialize membrane state with pulse tracking enabled
    pub fn init_state_with_pulse(&self, batch_size: usize) -> LeakyState {
        LeakyState {
            mem: Array2::zeros((batch_size, self.size)),
            time_since_spike: Some(Array2::from_elem((batch_size, self.size), f32::INFINITY)),
            adaptive_threshold: None,
            pending_spike_steps: None,
            reset_hold_steps: None,
        }
    }

    /// Initialize membrane state with threshold adaptation enabled
    pub fn init_state_with_adaptation(&self, batch_size: usize) -> LeakyState {
        let initial_threshold = self.mode.theta_low();
        LeakyState {
            mem: Array2::zeros((batch_size, self.size)),
            time_since_spike: None,
            adaptive_threshold: Some(Array2::from_elem(
                (batch_size, self.size),
                initial_threshold,
            )),
            pending_spike_steps: None,
            reset_hold_steps: None,
        }
    }

    /// Initialize membrane state with all physics features (pulse + adaptation)
    pub fn init_state_full(&self, batch_size: usize) -> LeakyState {
        let initial_threshold = self.mode.theta_low();
        LeakyState {
            mem: Array2::zeros((batch_size, self.size)),
            time_since_spike: Some(Array2::from_elem((batch_size, self.size), f32::INFINITY)),
            adaptive_threshold: Some(Array2::from_elem(
                (batch_size, self.size),
                initial_threshold,
            )),
            pending_spike_steps: None,
            reset_hold_steps: None,
        }
    }

    /// Initialize membrane state with hardware timing enabled
    /// Used for SPICE-accurate simulation with comparator delay and reset hold
    pub fn init_state_with_hardware_timing(&self, batch_size: usize) -> LeakyState {
        LeakyState {
            mem: Array2::zeros((batch_size, self.size)),
            time_since_spike: Some(Array2::from_elem((batch_size, self.size), f32::INFINITY)),
            adaptive_threshold: None,
            pending_spike_steps: Some(Array2::zeros((batch_size, self.size))),
            reset_hold_steps: Some(Array2::zeros((batch_size, self.size))),
        }
    }

    /// Compute membrane update based on mode (uses stored dt)
    #[inline]
    fn compute_membrane(&self, mem_prev: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        self.compute_membrane_with_dt(mem_prev, input, None)
    }

    /// Compute membrane update with optional dt override
    ///
    /// Args:
    ///   mem_prev: Previous membrane potential
    ///   input: Current injection
    ///   dt_override: Optional timestep override (for variable dt simulation)
    #[inline]
    fn compute_membrane_with_dt(
        &self,
        mem_prev: &Array2<f32>,
        input: &Array2<f32>,
        dt_override: Option<f32>,
    ) -> Array2<f32> {
        match &self.mode {
            NeuronMode::Simple => {
                // Simple: mem = beta * mem_prev + input
                // This treats input as direct injection (snnTorch style)
                // Note: dt_override is ignored in simple mode
                mem_prev * self.beta + input
            }
            NeuronMode::Physics { tau_m, dt, v_min, v_max, .. } => {
                // Physics: RC circuit with direct input injection
                // decay = exp(-dt/tau_m) (equivalent to beta)
                // mem = mem_prev * decay + input
                let effective_dt = dt_override.unwrap_or(*dt);
                let decay = (-effective_dt / tau_m).exp();
                let mem_new = mem_prev * decay + input;
                // Clamp to hardware voltage rails
                mem_new.mapv(|v| v.clamp(*v_min, *v_max))
            }
        }
    }

    /// Get the current dt for physics mode (or default 1ms for simple mode)
    pub fn get_dt(&self) -> f32 {
        match &self.mode {
            NeuronMode::Simple => 0.001, // Default 1ms
            NeuronMode::Physics { dt, .. } => *dt,
        }
    }

    /// Get tau_m for physics mode (or compute from beta for simple mode)
    pub fn get_tau_m(&self) -> f32 {
        match &self.mode {
            NeuronMode::Simple => {
                // Compute tau from beta assuming dt=1ms
                let dt = 0.001;
                if self.beta > 0.0 && self.beta < 1.0 {
                    -dt / self.beta.ln()
                } else {
                    0.01 // Fallback
                }
            }
            NeuronMode::Physics { tau_m, .. } => *tau_m,
        }
    }

    /// Forward pass for a single timestep
    ///
    /// Uses either Simple or Physics mode based on self.mode.
    ///
    /// Args:
    ///   input: Current injection [batch, size]
    ///   state: Previous membrane state
    ///
    /// Returns:
    ///   (spikes, new_state, cache for backward)
    pub fn forward(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        // Update membrane potential using mode-specific dynamics
        let mut mem_new = self.compute_membrane(&state.mem, input);

        // Generate spikes from current membrane: S = Heaviside(mem - threshold)
        let mem_shifted = &mem_new - self.threshold;
        let spikes = mem_shifted.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        // Apply reset based on CURRENT spike (immediate reset, not delayed)
        // This matches snnTorch with reset_delay=False behavior
        match self.reset_mechanism {
            ResetMechanism::Subtract => {
                // Subtract threshold when spike occurred
                mem_new = &mem_new - &(&spikes * self.threshold);
            }
            ResetMechanism::Zero => {
                // Zero out membrane where spike occurred
                mem_new = &mem_new * &(1.0 - &spikes);
            }
            ResetMechanism::None => {
                // No reset, pure integration
            }
        }

        // Cache only values needed for backward pass (trimmed for performance)
        let cache = LeakyCache {
            mem_shifted: mem_shifted.clone(),
            spikes: spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_new,
            time_since_spike: None,
            adaptive_threshold: None,
            pending_spike_steps: None,
            reset_hold_steps: None,
        };

        (spikes, new_state, cache)
    }

    /// Forward pass with variable timestep (for fine-grained physics simulation)
    ///
    /// Args:
    ///   input: Current injection [batch, size]
    ///   state: Previous membrane state
    ///   dt: Integration timestep (overrides stored dt in physics mode)
    ///
    /// Returns:
    ///   (spikes, new_state, cache for backward)
    pub fn forward_with_dt(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        // Update membrane potential with specified dt
        let mut mem_new = self.compute_membrane_with_dt(&state.mem, input, Some(dt));

        // Generate spikes from current membrane: S = Heaviside(mem - threshold)
        let mem_shifted = &mem_new - self.threshold;
        let spikes = mem_shifted.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        // Apply reset
        match self.reset_mechanism {
            ResetMechanism::Subtract => {
                mem_new = &mem_new - &(&spikes * self.threshold);
            }
            ResetMechanism::Zero => {
                mem_new = &mem_new * &(1.0 - &spikes);
            }
            ResetMechanism::None => {}
        }

        let cache = LeakyCache {
            mem_shifted: mem_shifted.clone(),
            spikes: spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_new,
            time_since_spike: None,
            adaptive_threshold: None,
            pending_spike_steps: None,
            reset_hold_steps: None,
        };

        (spikes, new_state, cache)
    }

    /// Forward pass with noise injection (for robustness training)
    ///
    /// Args:
    ///   input: Current injection [batch, size]
    ///   state: Previous membrane state
    ///   threshold_noise_std: Std dev for threshold variation (relative, e.g., 0.02 = 2%)
    ///   membrane_noise_std: Std dev for membrane noise (absolute)
    ///   rng: Random number generator
    ///
    /// Returns:
    ///   (spikes, new_state, cache for backward)
    pub fn forward_noisy<R: Rng>(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        threshold_noise_std: f32,
        membrane_noise_std: f32,
        rng: &mut R,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        // Update membrane potential using mode-specific dynamics
        let mut mem_new = self.compute_membrane(&state.mem, input);

        // Add membrane noise (thermal/shot noise)
        if membrane_noise_std.is_finite() && membrane_noise_std > 0.0 {
            if let Ok(normal) = Normal::new(0.0, membrane_noise_std as f64) {
                for v in mem_new.iter_mut() {
                    *v += normal.sample(rng) as f32;
                }
            }
        }

        let mem_shifted_for_grad = &mem_new - self.threshold;

        // Generate spikes with noisy threshold
        let mut effective_threshold = if threshold_noise_std.is_finite()
            && threshold_noise_std > 0.0
            && self.threshold.is_finite()
            && self.threshold != 0.0
        {
            let std = (self.threshold * threshold_noise_std).abs();
            if let Ok(normal) = Normal::new(0.0, std as f64) {
                self.threshold + normal.sample(rng) as f32
            } else {
                self.threshold
            }
        } else {
            self.threshold
        };
        if self.threshold > 0.0 {
            effective_threshold = effective_threshold.max(1e-6);
        }

        let mem_shifted = &mem_new - effective_threshold;
        let spikes = mem_shifted.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        // Apply reset based on CURRENT spike
        match self.reset_mechanism {
            ResetMechanism::Subtract => {
                mem_new = &mem_new - &(&spikes * effective_threshold);
            }
            ResetMechanism::Zero => {
                mem_new = &mem_new * &(1.0 - &spikes);
            }
            ResetMechanism::None => {}
        }

        // Cache for backward (use base threshold for gradient stability, and pre-reset membrane)
        let cache = LeakyCache {
            mem_shifted: mem_shifted_for_grad,
            spikes: spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_new,
            time_since_spike: None,
            adaptive_threshold: None,
            pending_spike_steps: None,
            reset_hold_steps: None,
        };

        (spikes, new_state, cache)
    }

    /// Forward pass with pulse stretching (physics mode)
    ///
    /// Instead of returning binary spikes, returns pulse-shaped output:
    /// V_pulse = V_peak * exp(-time_since_spike / tau_pulse)
    ///
    /// Binary spikes are still tracked for backward pass (gradient computation).
    ///
    /// Args:
    ///   input: Current injection [batch, size]
    ///   state: Previous membrane state (must have time_since_spike enabled)
    ///   dt: Integration timestep
    ///
    /// Returns:
    ///   (pulse_output, new_state, cache for backward)
    pub fn forward_with_pulse(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        // Get physics parameters
        let (tau_pulse, v_peak) = match &self.mode {
            NeuronMode::Physics {
                tau_pulse, v_peak, ..
            } => (*tau_pulse, *v_peak),
            NeuronMode::Simple => (default_tau_pulse(), 1.0), // Fallback for simple mode
        };

        // Update membrane potential
        let mut mem_new = self.compute_membrane_with_dt(&state.mem, input, Some(dt));

        // Generate spikes from current membrane
        let mem_shifted = &mem_new - self.threshold;
        let spikes = mem_shifted.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        // Apply reset
        match self.reset_mechanism {
            ResetMechanism::Subtract => {
                mem_new = &mem_new - &(&spikes * self.threshold);
            }
            ResetMechanism::Zero => {
                mem_new = &mem_new * &(1.0 - &spikes);
            }
            ResetMechanism::None => {}
        }

        // Update time_since_spike tracking
        let new_time_since_spike = if let Some(ref tss) = state.time_since_spike {
            // Increment time for all neurons
            let mut new_tss = tss + dt;
            // Reset to 0 for neurons that just spiked
            new_tss
                .iter_mut()
                .zip(spikes.iter())
                .for_each(|(t, &s)| {
                    if s > 0.0 {
                        *t = 0.0;
                    }
                });
            Some(new_tss)
        } else {
            // Initialize if not present
            let mut tss = Array2::from_elem(spikes.raw_dim(), f32::INFINITY);
            tss.iter_mut().zip(spikes.iter()).for_each(|(t, &s)| {
                if s > 0.0 {
                    *t = 0.0;
                }
            });
            Some(tss)
        };

        // Compute pulse output: V_peak * exp(-time_since_spike / tau_pulse)
        // Only output pulse if time_since_spike < 5*tau_pulse (cutoff for efficiency)
        let pulse_output = if let Some(ref tss) = new_time_since_spike {
            let cutoff = 5.0 * tau_pulse;
            tss.mapv(|t| {
                if t < cutoff {
                    v_peak * (-t / tau_pulse).exp()
                } else {
                    0.0
                }
            })
        } else {
            // Fallback: binary spikes scaled by v_peak
            &spikes * v_peak
        };

        let cache = LeakyCache {
            mem_shifted: mem_shifted.clone(),
            spikes: spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_new,
            time_since_spike: new_time_since_spike,
            adaptive_threshold: None,
            pending_spike_steps: None,
            reset_hold_steps: None,
        };

        (pulse_output, new_state, cache)
    }

    /// Forward pass with threshold adaptation (physics mode)
    ///
    /// Implements adaptive threshold dynamics:
    /// θ(t+dt) = θ_target + (θ(t) - θ_target) * exp(-dt/τ_θ)
    /// where θ_target = θ_high after spike, θ_low when quiet.
    ///
    /// Args:
    ///   input: Current injection [batch, size]
    ///   state: Previous membrane state (should have adaptive_threshold enabled)
    ///   dt: Integration timestep
    ///
    /// Returns:
    ///   (spikes, new_state, cache for backward)
    pub fn forward_with_adaptation(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        // Get threshold adaptation parameters
        let (tau_theta, theta_low, theta_high) = match &self.mode {
            NeuronMode::Physics {
                tau_theta,
                theta_low,
                theta_high,
                ..
            } => (*tau_theta, *theta_low, *theta_high),
            NeuronMode::Simple => (default_tau_theta(), 1.0, 1.0), // No adaptation
        };

        // Update membrane potential
        let mut mem_new = self.compute_membrane_with_dt(&state.mem, input, Some(dt));

        // Get current threshold for each neuron
        let current_threshold = if let Some(ref thresh) = state.adaptive_threshold {
            thresh.clone()
        } else {
            Array2::from_elem(mem_new.raw_dim(), self.threshold)
        };

        // Generate spikes using current adaptive threshold
        let mem_shifted = &mem_new - &current_threshold;
        let spikes = mem_shifted.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        // Apply reset using current threshold
        match self.reset_mechanism {
            ResetMechanism::Subtract => {
                mem_new = &mem_new - &(&spikes * &current_threshold);
            }
            ResetMechanism::Zero => {
                mem_new = &mem_new * &(1.0 - &spikes);
            }
            ResetMechanism::None => {}
        }

        // Update adaptive threshold:
        // θ(t+dt) = θ_target + (θ(t) - θ_target) * exp(-dt/τ_θ)
        // θ_target = θ_high if just spiked, θ_low otherwise
        let decay = (-dt / tau_theta).exp();
        let new_threshold = if state.adaptive_threshold.is_some() {
            let mut new_thresh = current_threshold.clone();
            new_thresh
                .iter_mut()
                .zip(spikes.iter())
                .for_each(|(theta, &spike)| {
                    let target = if spike > 0.0 { theta_high } else { theta_low };
                    *theta = target + (*theta - target) * decay;
                });
            Some(new_thresh)
        } else {
            None
        };

        // Cache uses mem_shifted relative to base threshold for gradient stability
        let cache = LeakyCache {
            mem_shifted: &mem_new + &current_threshold - self.threshold, // Normalize to base threshold
            spikes: spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_new,
            time_since_spike: None,
            adaptive_threshold: new_threshold,
            pending_spike_steps: None,
            reset_hold_steps: None,
        };

        (spikes, new_state, cache)
    }

    /// Forward pass with full physics features (pulse + adaptation)
    ///
    /// Combines pulse stretching and threshold adaptation.
    ///
    /// Args:
    ///   input: Current injection [batch, size]
    ///   state: Previous membrane state (should have both pulse and adaptation enabled)
    ///   dt: Integration timestep
    ///
    /// Returns:
    ///   (pulse_output, new_state, cache for backward)
    pub fn forward_full_physics(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        // Get all physics parameters
        let (tau_pulse, v_peak, tau_theta, theta_low, theta_high) = match &self.mode {
            NeuronMode::Physics {
                tau_pulse,
                v_peak,
                tau_theta,
                theta_low,
                theta_high,
                ..
            } => (*tau_pulse, *v_peak, *tau_theta, *theta_low, *theta_high),
            NeuronMode::Simple => (
                default_tau_pulse(),
                1.0,
                default_tau_theta(),
                1.0,
                1.0,
            ),
        };

        // Update membrane potential
        let mut mem_new = self.compute_membrane_with_dt(&state.mem, input, Some(dt));

        // Get current threshold
        let current_threshold = if let Some(ref thresh) = state.adaptive_threshold {
            thresh.clone()
        } else {
            Array2::from_elem(mem_new.raw_dim(), self.threshold)
        };

        // Generate spikes
        let mem_shifted = &mem_new - &current_threshold;
        let spikes = mem_shifted.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        // Apply reset
        match self.reset_mechanism {
            ResetMechanism::Subtract => {
                mem_new = &mem_new - &(&spikes * &current_threshold);
            }
            ResetMechanism::Zero => {
                mem_new = &mem_new * &(1.0 - &spikes);
            }
            ResetMechanism::None => {}
        }

        // Update time_since_spike
        let new_time_since_spike = if let Some(ref tss) = state.time_since_spike {
            let mut new_tss = tss + dt;
            new_tss
                .iter_mut()
                .zip(spikes.iter())
                .for_each(|(t, &s)| {
                    if s > 0.0 {
                        *t = 0.0;
                    }
                });
            Some(new_tss)
        } else {
            let mut tss = Array2::from_elem(spikes.raw_dim(), f32::INFINITY);
            tss.iter_mut().zip(spikes.iter()).for_each(|(t, &s)| {
                if s > 0.0 {
                    *t = 0.0;
                }
            });
            Some(tss)
        };

        // Update adaptive threshold
        let decay_theta = (-dt / tau_theta).exp();
        let new_threshold = {
            let mut new_thresh = current_threshold.clone();
            new_thresh
                .iter_mut()
                .zip(spikes.iter())
                .for_each(|(theta, &spike)| {
                    let target = if spike > 0.0 { theta_high } else { theta_low };
                    *theta = target + (*theta - target) * decay_theta;
                });
            Some(new_thresh)
        };

        // Compute pulse output
        let pulse_output = if let Some(ref tss) = new_time_since_spike {
            let cutoff = 5.0 * tau_pulse;
            tss.mapv(|t| {
                if t < cutoff {
                    v_peak * (-t / tau_pulse).exp()
                } else {
                    0.0
                }
            })
        } else {
            &spikes * v_peak
        };

        let cache = LeakyCache {
            mem_shifted: &mem_new + &current_threshold - self.threshold,
            spikes: spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_new,
            time_since_spike: new_time_since_spike,
            adaptive_threshold: new_threshold,
            pending_spike_steps: None,
            reset_hold_steps: None,
        };

        (pulse_output, new_state, cache)
    }

    /// Forward pass with hardware timing simulation
    ///
    /// Models realistic circuit behavior:
    /// - Comparator propagation delay: spike output is delayed after threshold crossing
    /// - Reset hold period: membrane is held at reset value for a configurable duration
    ///
    /// This matches SPICE circuit behavior more closely than instant reset.
    ///
    /// Args:
    ///   input: Current injection [batch, size]
    ///   state: Previous membrane state (should have hardware timing enabled)
    ///   dt: Integration timestep
    ///
    /// Returns:
    ///   (spikes, new_state, cache for backward)
    pub fn forward_with_hardware_timing(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        use ndarray::Zip;

        let shape = state.mem.raw_dim();
        let comparator_delay = self.mode.comparator_delay();
        let reset_hold = self.mode.reset_hold();

        // Convert delays to step counts
        let delay_steps = if comparator_delay > 0.0 {
            (comparator_delay / dt).ceil() as u16
        } else {
            0
        };
        let hold_steps = if reset_hold > 0.0 {
            (reset_hold / dt).ceil() as u16
        } else {
            0
        };

        // Get or initialize timing state
        let mut pending = state.pending_spike_steps.clone()
            .unwrap_or_else(|| Array2::zeros(shape));
        let mut hold = state.reset_hold_steps.clone()
            .unwrap_or_else(|| Array2::zeros(shape));

        // Initialize output spikes array (will be filled by pending spike completion)
        let mut emitted_spikes = Array2::zeros(shape);

        // Step 1: Decrement pending spike counters and emit spikes when ready
        Zip::from(&mut pending)
            .and(&mut emitted_spikes)
            .for_each(|p, e| {
                if *p > 0 {
                    *p -= 1;
                    if *p == 0 {
                        *e = 1.0; // Emit spike
                    }
                }
            });

        // Step 2: Compute membrane update
        // During reset hold: apply RC decay toward vref (0) with fast time constant
        // This models the reset switch pulling membrane to vref through R_reset (~200 ohm)
        // tau_reset = R_reset * C_mem = 200 * 10nF = 2µs (very fast)
        let mut mem_new = state.mem.clone();
        let mem_integrated = self.compute_membrane_with_dt(&state.mem, input, Some(dt));

        // Reset decay factor: tau_reset = 2µs, very fast decay toward vref
        let tau_reset = 2e-6_f32;
        let reset_decay = (-dt / tau_reset).exp();

        Zip::from(&mut mem_new)
            .and(&mem_integrated)
            .and(&hold)
            .for_each(|m, &integrated, &h| {
                if h == 0 {
                    // Normal integration
                    *m = integrated;
                } else {
                    // During reset hold: RC decay toward vref (0)
                    // membrane = membrane * exp(-dt/tau_reset)
                    *m *= reset_decay;
                }
            });

        // Step 3: Detect new threshold crossings (only for neurons not in hold, no pending spike,
        // and didn't just emit a spike this timestep)
        let mem_shifted = &mem_new - self.threshold;
        let mut new_crossings = Array2::zeros(shape);
        Zip::from(&mut new_crossings)
            .and(&mem_shifted)
            .and(&pending)
            .and(&hold)
            .and(&emitted_spikes)
            .for_each(|cross, &shifted, &pend, &h, &just_emitted| {
                // Only detect crossing if:
                // - Membrane is above threshold
                // - No pending spike already
                // - Not in reset hold
                // - Didn't just emit a spike (prevents immediate re-trigger)
                if shifted > 0.0 && pend == 0 && h == 0 && just_emitted == 0.0 {
                    *cross = 1.0;
                }
            });

        // Step 4: Schedule new spikes (apply delay or emit immediately)
        if delay_steps > 0 {
            Zip::from(&mut pending)
                .and(&new_crossings)
                .for_each(|p, &cross| {
                    if cross > 0.0 {
                        *p = delay_steps;
                    }
                });
        } else {
            // No delay - emit immediately
            emitted_spikes = &emitted_spikes + &new_crossings;
        }

        // Step 5: Start reset hold for neurons that just emitted spikes
        // (Reset is now handled by RC decay in Step 2 during hold period)
        // No instant reset - membrane decays naturally toward vref

        // Step 6: Start reset hold for neurons that emitted spikes
        if hold_steps > 0 {
            Zip::from(&mut hold)
                .and(&emitted_spikes)
                .for_each(|h, &spike| {
                    if spike > 0.0 {
                        *h = hold_steps;
                    }
                });
        }

        // Step 7: Decrement hold counters
        hold.mapv_inplace(|h| if h > 0 { h - 1 } else { 0 });

        // Update time_since_spike for pulse output calculation
        let new_time_since_spike = if let Some(ref tss) = state.time_since_spike {
            let mut new_tss = tss + dt;
            Zip::from(&mut new_tss)
                .and(&emitted_spikes)
                .for_each(|t, &spike| {
                    if spike > 0.0 {
                        *t = 0.0;
                    }
                });
            Some(new_tss)
        } else {
            None
        };

        // Compute pulse output if pulse mode is enabled
        let pulse_output = if let Some(ref tss) = new_time_since_spike {
            let tau_pulse = self.mode.tau_pulse();
            let v_peak = self.mode.v_peak();
            if tau_pulse > 0.0 {
                tss.mapv(|t| {
                    if t < 5.0 * tau_pulse {
                        v_peak * (-t / tau_pulse).exp()
                    } else {
                        0.0
                    }
                })
            } else {
                &emitted_spikes * v_peak
            }
        } else {
            emitted_spikes.clone()
        };

        let cache = LeakyCache {
            mem_shifted: mem_shifted.clone(),
            spikes: emitted_spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_new,
            time_since_spike: new_time_since_spike,
            adaptive_threshold: state.adaptive_threshold.clone(),
            pending_spike_steps: Some(pending),
            reset_hold_steps: Some(hold),
        };

        (pulse_output, new_state, cache)
    }

    /// Backward pass for a single timestep
    ///
    /// Args:
    ///   grad_spikes: Gradient w.r.t. spikes [batch, size]
    ///   grad_mem_next: Gradient from next timestep's membrane [batch, size]
    ///   cache: Cached values from forward pass
    ///
    /// Returns:
    ///   (grad_input, grad_mem_prev)
    pub fn backward(
        &self,
        grad_spikes: &Array2<f32>,
        grad_mem_next: &Array2<f32>,
        cache: &LeakyCache,
    ) -> (Array2<f32>, Array2<f32>) {
        // Surrogate gradient: dS/d(mem_shifted)
        let surrogate_grad = cache.mem_shifted.mapv(|x| self.spike_grad.backward(x));

        // Gradient through spike function
        // grad_mem_from_spikes = grad_spikes * surrogate_grad
        let grad_mem_from_spikes = grad_spikes * &surrogate_grad;

        // Total gradient w.r.t. membrane (from spikes and from next timestep)
        let grad_mem = &grad_mem_from_spikes + grad_mem_next;

        // Gradient w.r.t. input: d(mem_new)/d(input) = 1
        let grad_input = grad_mem.clone();

        // Gradient w.r.t. previous membrane: d(mem_new)/d(mem_prev) = beta
        // Reset gradient is detached (standard practice in SNN training)
        let grad_mem_prev = &grad_mem * self.beta;

        (grad_input, grad_mem_prev)
    }
}

/// State for Leaky neuron layer
#[derive(Clone, Debug)]
pub struct LeakyState {
    /// Membrane potential [batch, size]
    pub mem: Array2<f32>,
    /// Time since last spike for each neuron [batch, size]
    /// Used for pulse stretching: V_pulse(t) = V_peak * exp(-time_since_spike / tau_pulse)
    /// A value of f32::INFINITY means the neuron hasn't spiked yet
    pub time_since_spike: Option<Array2<f32>>,
    /// Adaptive threshold for each neuron [batch, size]
    /// θ(t+dt) = θ_target + (θ(t) - θ_target) * exp(-dt/τ_θ)
    /// None means using fixed threshold from Leaky::threshold
    pub adaptive_threshold: Option<Array2<f32>>,
    /// Pending spike delay counter (steps remaining until spike is emitted)
    /// 0 = no pending spike, >0 = steps until spike output
    /// Models comparator propagation delay
    pub pending_spike_steps: Option<Array2<u16>>,
    /// Reset hold counter (steps remaining in reset hold period)
    /// 0 = not in hold, >0 = steps until membrane can integrate again
    /// Models pulse-stretcher controlled reset switch
    pub reset_hold_steps: Option<Array2<u16>>,
}

impl LeakyState {
    /// Create new state with membrane only (no pulse tracking or adaptation)
    pub fn new(mem: Array2<f32>) -> Self {
        Self {
            mem,
            time_since_spike: None,
            adaptive_threshold: None,
            pending_spike_steps: None,
            reset_hold_steps: None,
        }
    }

    /// Create new state with pulse tracking enabled
    pub fn new_with_pulse_tracking(mem: Array2<f32>) -> Self {
        let shape = mem.raw_dim();
        Self {
            mem,
            time_since_spike: Some(Array2::from_elem(shape, f32::INFINITY)),
            adaptive_threshold: None,
            pending_spike_steps: None,
            reset_hold_steps: None,
        }
    }

    /// Create new state with threshold adaptation enabled
    pub fn new_with_threshold_adaptation(mem: Array2<f32>, initial_threshold: f32) -> Self {
        let shape = mem.raw_dim();
        Self {
            mem,
            time_since_spike: None,
            adaptive_threshold: Some(Array2::from_elem(shape, initial_threshold)),
            pending_spike_steps: None,
            reset_hold_steps: None,
        }
    }

    /// Create new state with both pulse tracking and threshold adaptation
    pub fn new_full(mem: Array2<f32>, initial_threshold: f32) -> Self {
        let shape = mem.raw_dim();
        Self {
            mem,
            time_since_spike: Some(Array2::from_elem(shape, f32::INFINITY)),
            adaptive_threshold: Some(Array2::from_elem(shape, initial_threshold)),
            pending_spike_steps: None,
            reset_hold_steps: None,
        }
    }

    /// Create new state with hardware timing enabled
    pub fn new_with_hardware_timing(mem: Array2<f32>) -> Self {
        let shape = mem.raw_dim();
        Self {
            mem,
            time_since_spike: Some(Array2::from_elem(shape, f32::INFINITY)),
            adaptive_threshold: None,
            pending_spike_steps: Some(Array2::zeros(shape)),
            reset_hold_steps: Some(Array2::zeros(shape)),
        }
    }

    /// Create new state with all features (pulse, adaptation, hardware timing)
    pub fn new_full_physics(mem: Array2<f32>, initial_threshold: f32) -> Self {
        let shape = mem.raw_dim();
        Self {
            mem,
            time_since_spike: Some(Array2::from_elem(shape, f32::INFINITY)),
            adaptive_threshold: Some(Array2::from_elem(shape, initial_threshold)),
            pending_spike_steps: Some(Array2::zeros(shape)),
            reset_hold_steps: Some(Array2::zeros(shape)),
        }
    }

    /// Reset membrane to zeros and thresholds to initial value
    pub fn reset(&mut self) {
        self.mem.fill(0.0);
        if let Some(ref mut tss) = self.time_since_spike {
            tss.fill(f32::INFINITY);
        }
        if let Some(ref mut pending) = self.pending_spike_steps {
            pending.fill(0);
        }
        if let Some(ref mut hold) = self.reset_hold_steps {
            hold.fill(0);
        }
        // Note: adaptive_threshold is not reset here - call reset_threshold() if needed
    }

    /// Reset adaptive threshold to a specific value
    pub fn reset_threshold(&mut self, initial_threshold: f32) {
        if let Some(ref mut thresh) = self.adaptive_threshold {
            thresh.fill(initial_threshold);
        }
    }

    /// Detach from computation graph (for TBPTT)
    pub fn detach(&mut self) {
        // In Rust, we don't have autograd, so this is a no-op
        // Kept for API compatibility with snnTorch
    }
}

/// Cached values from forward pass for backward computation (trimmed for performance)
#[derive(Clone, Debug)]
pub struct LeakyCache {
    pub mem_shifted: Array2<f32>,
    pub spikes: Array2<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_leaky_forward_no_spike() {
        let lif = Leaky::new(3, 0.9);
        let state = lif.init_state(1);

        // Small input, should not spike
        let input = array![[0.1, 0.2, 0.3]];
        let (spikes, new_state, _) = lif.forward(&input, &state);

        // No spikes expected
        assert!(spikes.iter().all(|&s| s == 0.0));

        // Membrane should increase
        assert!(new_state.mem[[0, 0]] > 0.0);
    }

    #[test]
    fn test_leaky_forward_with_spike() {
        let lif = Leaky::new(2, 0.9);
        let state = lif.init_state(1);

        // Large input, should spike
        let input = array![[2.0, 0.5]];
        let (spikes, _new_state, _) = lif.forward(&input, &state);

        // First neuron should spike
        assert_eq!(spikes[[0, 0]], 1.0);
        // Second should not
        assert_eq!(spikes[[0, 1]], 0.0);
    }

    #[test]
    fn test_leaky_reset_subtract() {
        // Use Simple mode for testing basic snnTorch-style behavior
        let lif = Leaky::new_simple(1, 0.9).with_reset_mechanism(ResetMechanism::Subtract);

        // Build up membrane over time
        let mut state = lif.init_state(1);
        let input = array![[0.6]];

        // First step: mem = 0.6
        let (_, new_state, _) = lif.forward(&input, &state);
        state = new_state;

        // Second step: mem = 0.9 * 0.6 + 0.6 = 1.14 -> spike
        let (spikes, new_state, _) = lif.forward(&input, &state);
        assert_eq!(spikes[[0, 0]], 1.0);

        // After spike, membrane should be reduced by threshold
        // Expected: 1.14 - 1.0 = 0.14
        assert!((new_state.mem[[0, 0]] - 0.14).abs() < 0.01);
    }

    #[test]
    fn test_leaky_backward() {
        let lif = Leaky::new(2, 0.9);
        let state = lif.init_state(1);

        let input = array![[0.5, 1.5]];
        let (_, _, cache) = lif.forward(&input, &state);

        let grad_spikes = array![[1.0, 1.0]];
        let grad_mem_next = array![[0.0, 0.0]];

        let (grad_input, grad_mem_prev) = lif.backward(&grad_spikes, &grad_mem_next, &cache);

        // Gradients should have correct shape
        assert_eq!(grad_input.shape(), input.shape());
        assert_eq!(grad_mem_prev.shape(), state.mem.shape());

        // Gradient should flow through surrogate function
        assert!(grad_input.iter().all(|&g| g.is_finite()));
    }

    #[test]
    fn test_beta_clamping() {
        // Beta should be clamped to [0, 1]
        let lif = Leaky::new(1, 1.5);
        assert_eq!(lif.beta, 1.0);

        let lif = Leaky::new(1, -0.5);
        assert_eq!(lif.beta, 0.0);
    }

    #[test]
    fn test_neuron_mode_simple() {
        // new() now defaults to Physics mode
        let lif = Leaky::new(2, 0.9);
        assert!(lif.is_physics_mode());

        // new_simple() for backward compatibility with Simple mode
        let lif_simple = Leaky::new_simple(2, 0.9);
        assert!(!lif_simple.is_physics_mode());
        assert!(matches!(lif_simple.mode, NeuronMode::Simple));
    }

    #[test]
    fn test_neuron_mode_physics() {
        // tau_m = 0.01s, dt = 0.001s -> beta = exp(-0.001/0.01) ≈ 0.9048
        let lif = Leaky::new_physics(2, 0.01, 0.001);
        assert!(lif.is_physics_mode());

        // Check beta is computed correctly
        let expected_beta = (-0.001f32 / 0.01).exp();
        assert!((lif.beta - expected_beta).abs() < 1e-5);
    }

    #[test]
    fn test_physics_mode_from_beta() {
        let mode = NeuronMode::physics_from_beta(0.9, 0.001);
        if let NeuronMode::Physics { tau_m, dt, .. } = mode {
            // tau = -dt / ln(beta) = -0.001 / ln(0.9) ≈ 0.00949
            let expected_tau = -0.001f32 / 0.9f32.ln();
            assert!((tau_m - expected_tau).abs() < 1e-5);
            assert_eq!(dt, 0.001);
        } else {
            panic!("Expected Physics mode");
        }
    }

    #[test]
    fn test_physics_mode_dynamics() {
        // Physics mode should produce similar results to simple mode
        // when tau is computed from beta
        let beta = 0.9f32;
        let dt = 0.001f32;
        let tau_m = -dt / beta.ln();

        let lif_simple = Leaky::new(1, beta);
        let lif_physics = Leaky::new_physics(1, tau_m, dt);

        let state = lif_simple.init_state(1);
        let input = array![[0.5]];

        // Forward pass should produce very similar membrane dynamics
        let (_, state_simple, _) = lif_simple.forward(&input, &state);
        let (_, state_physics, _) = lif_physics.forward(&input, &state);

        // Membrane potentials should be very close (RC formula equivalent to beta*mem + input)
        let diff = (state_simple.mem[[0, 0]] - state_physics.mem[[0, 0]]).abs();
        assert!(diff < 1e-3, "Membrane diff too large: {}", diff);
    }

    #[test]
    fn test_with_mode_builder() {
        let lif = Leaky::new(2, 0.9)
            .with_mode(NeuronMode::physics(0.01, 0.001, default_tau_pulse(), default_v_peak()));

        assert!(lif.is_physics_mode());
        // Beta should be updated to match physics parameters
        let expected_beta = (-0.001f32 / 0.01).exp();
        assert!((lif.beta - expected_beta).abs() < 1e-5);
    }

    #[test]
    fn test_forward_with_dt() {
        // Test variable dt support
        let tau_m = 0.01f32;
        let dt_stored = 0.001f32;
        let lif = Leaky::new_physics(1, tau_m, dt_stored);

        let state = lif.init_state(1);
        let input = array![[0.5]];

        // Forward with stored dt
        let (_, state1, _) = lif.forward(&input, &state);

        // Forward with 2x dt (should decay more)
        let (_, state2, _) = lif.forward_with_dt(&input, &state, dt_stored * 2.0);

        // Forward with 0.5x dt (should decay less)
        let (_, state3, _) = lif.forward_with_dt(&input, &state, dt_stored * 0.5);

        // With larger dt, membrane should be lower (more decay)
        // state2 < state1 < state3 for the membrane after decay
        // Actually: mem = mem_prev * exp(-dt/tau) + input
        // From state.mem = 0: mem = 0 + input = 0.5 (no decay effect yet)
        // So first step all should be the same!

        // Let's do a second step to see the difference
        let (_, state1b, _) = lif.forward(&input, &state1);
        let (_, state2b, _) = lif.forward_with_dt(&input, &state2, dt_stored * 2.0);
        let (_, state3b, _) = lif.forward_with_dt(&input, &state3, dt_stored * 0.5);

        // After 2 steps, larger dt should result in less membrane buildup
        // (more decay between steps)
        assert!(state2b.mem[[0, 0]] < state1b.mem[[0, 0]]);
        assert!(state1b.mem[[0, 0]] < state3b.mem[[0, 0]]);
    }

    #[test]
    fn test_forward_with_pulse() {
        // Test pulse stretching
        let tau_m = 0.01f32;
        let dt = 0.001f32;
        let tau_pulse = 0.00167f32;
        let v_peak = 4.42f32;

        let lif = Leaky::new_physics_with_pulse(1, tau_m, dt, tau_pulse, v_peak);
        let mut state = lif.init_state_with_pulse(1);
        let input = array![[1.5]]; // Large enough to spike (threshold = 1.0)

        // First timestep: should spike and output v_peak
        let (pulse1, new_state, cache1) = lif.forward_with_pulse(&input, &state, dt);
        state = new_state;
        assert_eq!(
            cache1.spikes[[0, 0]], 1.0,
            "Expected spike in first timestep"
        );

        // At t=0 (just spiked), pulse should be v_peak
        assert!(
            (pulse1[[0, 0]] - v_peak).abs() < 0.01,
            "Expected pulse {} at spike, got {}",
            v_peak,
            pulse1[[0, 0]]
        );

        // Second timestep with zero input (no new spike, let membrane decay)
        let zero_input = array![[0.0]];
        let (pulse2, new_state, cache2) = lif.forward_with_pulse(&zero_input, &state, dt);
        state = new_state;
        assert_eq!(
            cache2.spikes[[0, 0]], 0.0,
            "Expected no spike in second timestep, mem = {}",
            state.mem[[0, 0]]
        );

        // Pulse should decay: v_peak * exp(-dt/tau_pulse)
        let expected_decay = v_peak * (-dt / tau_pulse).exp();
        assert!(
            (pulse2[[0, 0]] - expected_decay).abs() < 0.1,
            "Expected decayed pulse ~{}, got {}",
            expected_decay,
            pulse2[[0, 0]]
        );

        // After more timesteps without spiking, pulse should continue decaying
        let (pulse3, _, _) = lif.forward_with_pulse(&zero_input, &state, dt);
        assert!(
            pulse3[[0, 0]] < pulse2[[0, 0]],
            "Pulse should decay: {} should be < {}",
            pulse3[[0, 0]],
            pulse2[[0, 0]]
        );
    }

    #[test]
    fn test_get_dt_and_tau() {
        // Simple mode: dt derived from beta assuming 1ms timestep
        let lif_simple = Leaky::new_simple(2, 0.9);
        assert!((lif_simple.get_dt() - 0.001).abs() < 1e-6);

        let tau_expected = -0.001f32 / 0.9f32.ln();
        assert!((lif_simple.get_tau_m() - tau_expected).abs() < 1e-5);

        // Physics mode: dt and tau_m explicitly set
        let lif_physics = Leaky::new_physics(2, 0.005, 0.0001);
        assert!((lif_physics.get_dt() - 0.0001).abs() < 1e-8);
        assert!((lif_physics.get_tau_m() - 0.005).abs() < 1e-6);
    }

    #[test]
    fn test_forward_with_adaptation() {
        // Create a physics neuron with threshold adaptation
        let tau_m = 0.00949; // ~beta=0.9 at dt=1ms
        let dt = 0.001;
        let tau_theta = 0.001; // Fast adaptation for testing
        let theta_low = 1.0;
        let theta_high = 1.5; // 50% increase after spike

        let lif = Leaky::new_physics_with_threshold_adaptation(
            2, tau_m, dt, tau_theta, theta_low, theta_high,
        );

        // Initialize state with adaptive threshold
        let mut state = lif.init_state_with_adaptation(1);

        // Verify initial threshold is theta_low
        let initial_thresh = state.adaptive_threshold.as_ref().unwrap()[[0, 0]];
        assert!(
            (initial_thresh - theta_low).abs() < 1e-6,
            "Initial threshold should be theta_low={}, got {}",
            theta_low,
            initial_thresh
        );

        // Apply strong input to trigger spike
        let strong_input = Array2::from_elem((1, 2), 3.0);
        let (spikes, new_state, _) = lif.forward_with_adaptation(&strong_input, &state, dt);

        // Should have spiked
        assert!(
            spikes[[0, 0]] > 0.0,
            "Expected spike with strong input, mem was {}",
            new_state.mem[[0, 0]]
        );

        // Threshold should have moved toward theta_high
        let thresh_after_spike = new_state.adaptive_threshold.as_ref().unwrap()[[0, 0]];
        assert!(
            thresh_after_spike > theta_low,
            "Threshold should increase after spike: {} should be > {}",
            thresh_after_spike,
            theta_low
        );

        // Expected: theta_high + (theta_low - theta_high) * exp(-dt/tau_theta)
        // = theta_high + (theta_low - theta_high) * exp(-1) ≈ theta_high + (theta_low - theta_high) * 0.368
        let decay = (-dt / tau_theta).exp();
        let expected_after_spike = theta_high + (theta_low - theta_high) * decay;
        assert!(
            (thresh_after_spike - expected_after_spike).abs() < 0.01,
            "Threshold after spike should be ~{}, got {}",
            expected_after_spike,
            thresh_after_spike
        );

        // Now apply zero input for several timesteps until no more spikes
        // Threshold should decay toward theta_low
        state = new_state;
        let zero_input = Array2::zeros((1, 2));

        // Run until membrane drains and no more spikes
        let mut no_spike_count = 0;
        for _ in 0..50 {
            let (spikes, next_state, _) = lif.forward_with_adaptation(&zero_input, &state, dt);
            if spikes[[0, 0]] == 0.0 {
                no_spike_count += 1;
            }
            state = next_state;
        }

        // Should have at least some timesteps without spiking
        assert!(
            no_spike_count > 40,
            "Expected at least 40 no-spike timesteps, got {}",
            no_spike_count
        );

        // Threshold should have decayed toward theta_low after many timesteps
        let thresh_after_decay = state.adaptive_threshold.as_ref().unwrap()[[0, 0]];
        assert!(
            (thresh_after_decay - theta_low).abs() < 0.05,
            "Threshold should be near theta_low after decay: {} should be near {}",
            thresh_after_decay,
            theta_low
        );
    }

    #[test]
    fn test_physics_voltage_clamping() {
        // Create physics neuron with hardware constraints
        let tau_m = 0.00949; // ~beta=0.9 at dt=1ms
        let dt = 0.001;
        let lif = Leaky::new_physics(1, tau_m, dt);

        // Verify v_min and v_max are set correctly
        assert_eq!(lif.mode.v_min(), 0.0, "v_min should be 0.0V (ground)");
        assert_eq!(lif.mode.v_max(), 5.0, "v_max should be 5.0V (supply rail)");

        let mut state = lif.init_state(1);

        // Test clamping at upper limit: inject huge current that would exceed v_max
        let huge_input = array![[100.0]]; // Would drive membrane way above 5V
        let (_, new_state, _) = lif.forward(&huge_input, &state);
        state = new_state;

        assert!(
            state.mem[[0, 0]] <= 5.0,
            "Membrane should be clamped to v_max=5.0V, got {}",
            state.mem[[0, 0]]
        );

        // Test clamping at lower limit: after reset, apply negative current
        // First spike to reset membrane
        state = lif.init_state(1);
        let spike_input = array![[2.0]]; // Enough to spike
        let (spikes, new_state, _) = lif.forward(&spike_input, &state);
        state = new_state;
        assert_eq!(spikes[[0, 0]], 1.0, "Should have spiked");

        // Now apply large negative input (would drive membrane negative without clamping)
        let negative_input = array![[-100.0]];
        let (_, new_state, _) = lif.forward(&negative_input, &state);

        assert!(
            new_state.mem[[0, 0]] >= 0.0,
            "Membrane should be clamped to v_min=0.0V, got {}",
            new_state.mem[[0, 0]]
        );
    }

    #[test]
    fn test_simple_mode_no_clamping() {
        // Simple mode should NOT clamp (backward compatibility)
        let lif = Leaky::new_simple(1, 0.9);

        // Check that simple mode returns infinite bounds
        assert!(lif.mode.v_min().is_infinite(), "Simple mode should have -inf v_min");
        assert!(lif.mode.v_max().is_infinite(), "Simple mode should have +inf v_max");

        let state = lif.init_state(1);

        // Large input should NOT be clamped in simple mode
        let huge_input = array![[100.0]];
        let (_, new_state, _) = lif.forward(&huge_input, &state);

        // Note: membrane won't actually be 100 due to reset after spike,
        // but the point is it's not clamped to 5.0V
        // After spike with subtract reset: 100 - 1.0 = 99.0 (way above 5V)
        assert!(
            new_state.mem[[0, 0]] > 5.0,
            "Simple mode should NOT clamp membrane: got {}",
            new_state.mem[[0, 0]]
        );
    }
}
