//! Network state types
//!
//! Contains the runtime state (membrane potentials) that persists across timesteps.

use crate::neurons::LeakyState;

/// Network state (membrane potentials)
#[derive(Clone, Debug)]
pub struct NetworkState {
    pub lif1_state: LeakyState,
    pub lif2_state: LeakyState,
}

impl NetworkState {
    pub fn reset(&mut self) {
        self.lif1_state.reset();
        self.lif2_state.reset();
    }
}
