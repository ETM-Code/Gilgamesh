//! Spiking neuron models
//!
//! This module provides spiking neuron implementations.
//! Supports both Simple (snnTorch-compatible) and Physics (RC circuit) modes.

pub mod lif;

pub use lif::{Leaky, LeakyCache, LeakyState, NeuronMode, ResetMechanism};
