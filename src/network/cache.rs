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
    pub cur1: Array2<f32>,
    pub spk1: Array2<f32>,
    pub cur2: Array2<f32>,
    pub lif1_cache: LeakyCache,
    pub lif2_cache: LeakyCache,
}
