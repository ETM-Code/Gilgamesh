//! Leaky Integrate-and-Fire (LIF) neuron module
//!
//! Supports two modes:
//! - Simple: `mem[t+1] = beta * mem[t] + input` (snnTorch-compatible)
//! - Physics: `mem[t+1] = u_inf + (mem[t] - u_inf) * exp(-dt/tau)` (RC circuit)

mod backward;
mod builder;
mod cache;
mod forward;
mod leaky;
mod mode;
mod state;

pub use cache::LeakyCache;
pub use leaky::Leaky;
pub use mode::{NeuronMode, PhysicsParams, ResetMechanism};
pub use state::LeakyState;

#[cfg(test)]
mod leaky_tests;
