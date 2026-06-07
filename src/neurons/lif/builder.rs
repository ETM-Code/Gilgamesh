use crate::surrogate::SurrogateGradient;
use ndarray::Array2;

use super::leaky::Leaky;
use super::mode::{
    default_comparator_delay, default_reset_hold, default_tau_pulse, default_tau_theta,
    default_theta_high, default_theta_low, default_v_max, default_v_min, default_v_peak,
    NeuronMode, ResetMechanism,
};
use super::state::LeakyState;
use crate::neurons::{DEFAULT_DT, DEFAULT_SPIKE_GRAD_SLOPE, DEFAULT_THRESHOLD};

impl Leaky {
    /// Create a new Leaky neuron layer (defaults to Physics mode)
    ///
    /// The beta parameter is used for backward compatibility but may be
    /// overridden by the Physics mode's tau_m/dt settings.
    pub fn new(size: usize, beta: f32) -> Self {
        Self {
            beta: beta.clamp(0.0, 1.0),
            threshold: DEFAULT_THRESHOLD,
            spike_grad: SurrogateGradient::fast_sigmoid(DEFAULT_SPIKE_GRAD_SLOPE),
            reset_mechanism: ResetMechanism::Subtract,
            size,
            mode: NeuronMode::default(),
        }
    }

    /// Create a new Leaky neuron layer with Simple mode (backward compatibility)
    pub fn new_simple(size: usize, beta: f32) -> Self {
        Self {
            beta: beta.clamp(0.0, 1.0),
            threshold: DEFAULT_THRESHOLD,
            spike_grad: SurrogateGradient::fast_sigmoid(DEFAULT_SPIKE_GRAD_SLOPE),
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
            threshold: DEFAULT_THRESHOLD,
            spike_grad: SurrogateGradient::fast_sigmoid(DEFAULT_SPIKE_GRAD_SLOPE),
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
            threshold: DEFAULT_THRESHOLD,
            spike_grad: SurrogateGradient::fast_sigmoid(DEFAULT_SPIKE_GRAD_SLOPE),
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
            threshold: theta_low,
            spike_grad: SurrogateGradient::fast_sigmoid(DEFAULT_SPIKE_GRAD_SLOPE),
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
        if let NeuronMode::Physics { tau_m, dt, .. } = &mode {
            self.beta = (-dt / tau_m).exp();
        }
        self.mode = mode;
        self
    }

    /// Check if in physics mode
    pub fn is_physics_mode(&self) -> bool {
        matches!(self.mode, NeuronMode::Physics { .. })
    }

    /// Allocate a zeroed membrane matrix for `batch_size` samples of this layer.
    fn zero_mem(&self, batch_size: usize) -> Array2<f32> {
        Array2::zeros((batch_size, self.size))
    }

    /// Initialize membrane state for a batch (no pulse tracking or adaptation)
    pub fn init_state(&self, batch_size: usize) -> LeakyState {
        LeakyState::new(self.zero_mem(batch_size))
    }

    /// Initialize membrane state with pulse tracking enabled
    pub fn init_state_with_pulse(&self, batch_size: usize) -> LeakyState {
        LeakyState::new_with_pulse_tracking(self.zero_mem(batch_size))
    }

    /// Initialize membrane state with threshold adaptation enabled
    pub fn init_state_with_adaptation(&self, batch_size: usize) -> LeakyState {
        let initial_threshold = self.mode.theta_low();
        LeakyState::new_with_threshold_adaptation(self.zero_mem(batch_size), initial_threshold)
    }

    /// Initialize membrane state with all physics features (pulse + adaptation)
    pub fn init_state_full(&self, batch_size: usize) -> LeakyState {
        let initial_threshold = self.mode.theta_low();
        LeakyState::new_full(self.zero_mem(batch_size), initial_threshold)
    }

    /// Initialize membrane state with hardware timing enabled
    pub fn init_state_with_hardware_timing(&self, batch_size: usize) -> LeakyState {
        LeakyState::new_with_hardware_timing(self.zero_mem(batch_size))
    }

    /// Get the current dt for physics mode (or default 1ms for simple mode)
    pub fn get_dt(&self) -> f32 {
        match &self.mode {
            NeuronMode::Simple => DEFAULT_DT,
            NeuronMode::Physics { dt, .. } => *dt,
        }
    }

    /// Get tau_m for physics mode (or compute from beta for simple mode)
    pub fn get_tau_m(&self) -> f32 {
        match &self.mode {
            NeuronMode::Simple => {
                let dt = DEFAULT_DT;
                if self.beta > 0.0 && self.beta < 1.0 {
                    -dt / self.beta.ln()
                } else {
                    0.01
                }
            }
            NeuronMode::Physics { tau_m, .. } => *tau_m,
        }
    }
}
