//! Network composition for spiking neural networks
//!
//! This module provides a complete SNN architecture matching the snnTorch example:
//! Input -> Linear -> LIF -> Linear -> LIF -> Output

mod cache;
mod constructors;
mod gradients;
mod network;
mod state;
mod trace;

pub use cache::NetworkCache;
pub use gradients::NetworkGradients;
pub use network::Network;
pub use state::NetworkState;
pub use trace::SimulationTrace;
