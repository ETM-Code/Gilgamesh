//! Neuron mode and reset mechanism types
//!
//! Defines the computation modes for LIF neurons:
//! - Simple: discrete-time model (snnTorch-compatible)
//! - Physics: RC circuit model with hardware-accurate timing

use serde::{Deserialize, Serialize};

/// Reset mechanism for membrane potential after spike
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

pub fn default_tau_pulse() -> f32 {
    1.5e-6 // 1.5us pulse stretch (~0.8us above 2.5V with diode drop)
}

pub fn default_v_peak() -> f32 {
    4.44 // ~5.0 - 0.56 diode drop on pulse stretcher
}

pub fn default_tau_theta() -> f32 {
    0.001 // 1ms threshold adaptation time constant
}

pub fn default_theta_low() -> f32 {
    1.0 // Base threshold when quiet
}

pub fn default_theta_high() -> f32 {
    1.2 // Elevated threshold after spiking (20% increase)
}

pub fn default_v_min() -> f32 {
    0.0 // Hardware ground rail
}

pub fn default_v_max() -> f32 {
    5.0 // Hardware supply rail
}

pub fn default_comparator_delay() -> f32 {
    0.0 // Default: no delay (instant, for backwards compatibility)
    // Set to 50e-9 (50ns) for realistic NCS2250 behavior
}

pub fn default_reset_hold() -> f32 {
    0.24 * default_tau_pulse() // Hold until comp_pulse falls below ~3.5V (0.7 * VDD)
}

impl Default for NeuronMode {
    fn default() -> Self {
        // Default to Physics mode for hardware-accurate simulation
        // Use sensible defaults matching typical passive RC neuron circuits
        NeuronMode::Physics {
            tau_m: 0.00396,      // 3.96ms (matches SPICE: 120kΩ * 33nF)
            dt: 1e-6,            // 1µs timestep
            tau_pulse: default_tau_pulse(),
            v_peak: 2.6,         // Peak with diode drop
            tau_theta: default_tau_theta(),
            theta_low: default_theta_low(),
            theta_high: default_theta_high(),
            v_min: default_v_min(),
            v_max: default_v_max(),
            comparator_delay_s: 50e-9,  // 50ns comparator delay
            reset_hold_s: default_reset_hold(),
        }
    }
}

/// Parameters for Physics mode construction.
/// Use struct update syntax to override specific fields:
/// ```ignore
/// PhysicsParams { tau_m: 0.005, dt: 1e-6, ..Default::default() }.into()
/// ```
#[derive(Clone, Debug)]
pub struct PhysicsParams {
    pub tau_m: f32,
    pub dt: f32,
    pub tau_pulse: f32,
    pub v_peak: f32,
    pub tau_theta: f32,
    pub theta_low: f32,
    pub theta_high: f32,
    pub v_min: f32,
    pub v_max: f32,
    pub comparator_delay_s: f32,
    pub reset_hold_s: f32,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            tau_m: 0.00396,
            dt: 0.001,
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
}

impl From<PhysicsParams> for NeuronMode {
    fn from(p: PhysicsParams) -> Self {
        NeuronMode::Physics {
            tau_m: p.tau_m,
            dt: p.dt,
            tau_pulse: p.tau_pulse,
            v_peak: p.v_peak,
            tau_theta: p.tau_theta,
            theta_low: p.theta_low,
            theta_high: p.theta_high,
            v_min: p.v_min,
            v_max: p.v_max,
            comparator_delay_s: p.comparator_delay_s,
            reset_hold_s: p.reset_hold_s,
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
        PhysicsParams { tau_m, dt, ..Default::default() }.into()
    }

    /// Create physics mode with pulse parameters (no threshold adaptation)
    pub fn physics(tau_m: f32, dt: f32, tau_pulse: f32, v_peak: f32) -> Self {
        PhysicsParams { tau_m, dt, tau_pulse, v_peak, ..Default::default() }.into()
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
        PhysicsParams { tau_m, dt, tau_pulse, v_peak, tau_theta, theta_low, theta_high, ..Default::default() }.into()
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
        PhysicsParams { tau_m, dt, tau_pulse, v_peak, comparator_delay_s, reset_hold_s, ..Default::default() }.into()
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

    /// Get v_min (returns -infinity for Simple mode - no clamping)
    pub fn v_min(&self) -> f32 {
        match self {
            NeuronMode::Simple => f32::NEG_INFINITY,
            NeuronMode::Physics { v_min, .. } => *v_min,
        }
    }

    /// Get v_max (returns infinity for Simple mode - no clamping)
    pub fn v_max(&self) -> f32 {
        match self {
            NeuronMode::Simple => f32::INFINITY,
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
