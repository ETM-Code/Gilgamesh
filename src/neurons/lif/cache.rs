//! Cache types for LIF neuron forward pass
//!
//! Stores intermediate values needed for backpropagation.

use ndarray::Array2;

/// Cached values from forward pass for backward computation (trimmed for performance)
#[derive(Clone, Debug)]
pub struct LeakyCache {
    pub mem_shifted: Array2<f32>,
    pub spikes: Array2<f32>,
}
