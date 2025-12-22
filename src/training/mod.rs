//! Training algorithms for spiking neural networks.
//!
//! This module provides:
//! - Surrogate gradient functions for backpropagation through spikes
//! - Quantization-aware training (QAT) for hardware deployment

pub mod surrogate;
pub mod qat;

pub use surrogate::{SurrogateGradient, FastSigmoid};
pub use qat::QATWeight;
