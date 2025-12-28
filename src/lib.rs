//! gilgamesh - Hardware-accurate spiking neural network
//!
//! A Rust implementation of spiking neural networks with physics-accurate
//! membrane dynamics for chip deployment. Supports both simple beta-decay
//! (snnTorch-compatible) and physics-accurate RC membrane models.

// Link BLAS library when enabled via features
#[cfg(any(feature = "blas-accelerate", feature = "blas-openblas"))]
extern crate blas_src;

pub mod config;
pub mod surrogate;
pub mod neurons;
pub mod layers;
pub mod network;
pub mod training;
pub mod data;
pub mod tensor;

pub use config::Config;
pub use data::{InputEncoder, InputEncodingType};
pub use neurons::leaky::Leaky;
pub use layers::linear::Linear;
pub use network::Network;
pub use surrogate::SurrogateGradient;
