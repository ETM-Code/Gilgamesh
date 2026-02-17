//! Network cache types for backward pass
//!
//! Stores intermediate values needed for backpropagation through time.

use crate::neurons::LeakyCache;
use ndarray::Array2;

/// Cached values for backward pass
#[derive(Clone, Debug)]
pub struct NetworkCache {
    // For rate-coded: input is same for all timesteps, passed separately to backward()
    // For temporal: encoded_input stores the per-timestep input for correct gradients
    pub encoded_input: Option<Array2<f32>>,
    pub hidden_current: Array2<f32>,
    pub hidden_spikes: Array2<f32>,
    pub output_current: Array2<f32>,
    pub lif1_cache: LeakyCache,
    pub lif2_cache: LeakyCache,
    /// For spiking input: the input spikes fed to fc1 (binary {0,1})
    pub input_spikes: Option<Array2<f32>>,
    /// For spiking input: accumulator before thresholding (for surrogate gradient)
    pub input_accum_pre: Option<Array2<f32>>,
}

impl NetworkCache {
    pub fn new(
        hidden_current: Array2<f32>,
        hidden_spikes: Array2<f32>,
        output_current: Array2<f32>,
        lif1_cache: LeakyCache,
        lif2_cache: LeakyCache,
    ) -> Self {
        Self {
            encoded_input: None,
            hidden_current,
            hidden_spikes,
            output_current,
            lif1_cache,
            lif2_cache,
            input_spikes: None,
            input_accum_pre: None,
        }
    }
}
