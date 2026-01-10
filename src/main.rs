//! gilgamesh - Hardware-accurate spiking neural network
//!
//! A Rust implementation of spiking neural networks with physics-accurate
//! membrane dynamics for chip deployment.
//!
//! Usage:
//!   gilgamesh train --epochs 15 --lr 0.001
//!   gilgamesh train --config config.json
//!   gilgamesh evaluate --checkpoint model.json

use anyhow::Result;
use clap::Parser;

mod cli;

fn main() -> Result<()> {
    cli::Cli::parse().run()
}
