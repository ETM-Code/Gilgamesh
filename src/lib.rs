//! gilgamesh - Hardware-accurate spiking neural network
//!
//! A Rust implementation of spiking neural networks with physics-accurate
//! membrane dynamics for chip deployment. Supports both simple beta-decay
//! (snnTorch-compatible) and physics-accurate RC membrane models.

// Link BLAS library when enabled via features
#[cfg(any(feature = "blas-accelerate", feature = "blas-openblas"))]
extern crate blas_src;

pub mod animation;
pub mod checkpoint;
pub mod config;
pub mod dashboard;
pub mod data;
pub mod hardware;
pub mod inspector;
pub mod layers;
pub mod network;
pub mod neurons;
pub mod spice;
pub mod surrogate;
pub mod tensor;
pub mod training;
pub mod visualization;
pub mod web;

pub use config::Config;
pub use data::{InputEncoder, InputEncodingType};
pub use hardware::{HardwareConfig, HardwareMapping};
pub use layers::linear::Linear;
pub use network::Network;
pub use neurons::Leaky;
pub use surrogate::SurrogateGradient;
pub use visualization::TrainingRecorder;
