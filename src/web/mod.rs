//! Web UI server for gilgamesh
//!
//! Provides a browser-based interface for:
//! - Real-time network visualization during inference
//! - Training progress monitoring
//! - Interactive controls (pause, navigate samples, adjust speed)
//!
//! Enable with the `web` feature:
//! ```bash
//! cargo run --features web -- web --port 3000
//! ```

#[cfg(feature = "web")]
pub mod protocol;
#[cfg(feature = "web")]
pub mod server;
#[cfg(feature = "web")]
pub mod simulation;

#[cfg(feature = "web")]
pub use protocol::{ClientMessage, ServerMessage, SimulationMode};
#[cfg(feature = "web")]
pub use server::{run_server, AppState};
