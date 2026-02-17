//! Network state types
//!
//! Contains the runtime state (membrane potentials) that persists across timesteps.

use crate::neurons::LeakyState;
use ndarray::Array2;

/// Network state (membrane potentials)
#[derive(Clone, Debug)]
pub struct NetworkState {
    pub lif1_state: LeakyState,
    pub lif2_state: LeakyState,
    /// Accumulator for spiking input encoding (None when not using spiking inputs)
    pub input_accum: Option<Array2<f32>>,
}

impl NetworkState {
    pub fn reset(&mut self) {
        self.lif1_state.reset();
        self.lif2_state.reset();
        if let Some(ref mut accum) = self.input_accum {
            accum.fill(0.0);
        }
    }
}
