//! Spiking neuron models
//!
//! This module provides spiking neuron implementations.
//! Supports both Simple (snnTorch-compatible) and Physics (RC circuit) modes.

pub mod lif;

pub use lif::{Leaky, LeakyCache, LeakyState, NeuronMode, ResetMechanism};

/// Default integration timestep in seconds (1ms) used when a mode has no
/// explicit `dt` (e.g. Simple mode) or as the generic trainer/network step.
pub const DEFAULT_DT: f32 = 0.001;

/// Default spike threshold for a freshly built [`Leaky`] layer.
pub const DEFAULT_THRESHOLD: f32 = 1.0;

/// Default fast-sigmoid surrogate-gradient slope for spike backprop.
pub const DEFAULT_SPIKE_GRAD_SLOPE: f32 = 25.0;
