//! Neuron state types for LIF neurons
//!
//! Contains the runtime state (membrane potential, timing, thresholds)
//! that persists across timesteps during simulation.

use ndarray::Array2;

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
