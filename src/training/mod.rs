//! Training algorithms for spiking neural networks.
//!
//! This module provides:
//! - Surrogate gradient functions for backpropagation through spikes
//! - Quantization-aware training (QAT) for hardware deployment
//! - Network weight quantization utilities

pub mod qat;
pub mod quantize;
pub mod surrogate;

pub use qat::QATWeight;
pub use quantize::{quantize_network_weights, apply_quantized_weights, QuantizeConfig, QuantizationResult};
pub use surrogate::{FastSigmoid, SurrogateGradient};
