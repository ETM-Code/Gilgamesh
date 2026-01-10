//! Leaky Integrate-and-Fire (LIF) neuron implementation
//!
//! Core neuron struct.

use crate::surrogate::SurrogateGradient;
use serde::{Deserialize, Serialize};

use super::mode::{NeuronMode, ResetMechanism};

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
