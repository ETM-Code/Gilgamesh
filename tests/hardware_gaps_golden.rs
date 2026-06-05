//! Characterization tests for hardware.rs methods not covered by hardware_golden.rs:
//! HardwareMapping::with_gain, HardwareConfig::current_for_dv, input_to_dv.

mod common;
use common::*;

use gilgamesh::hardware::{HardwareConfig, HardwareMapping};

#[test]
fn with_gain_apply_matches_input_to_dv_golden() {
    let cfg = HardwareConfig::default();
    let custom_gain = 2e-5;
    let map = HardwareMapping::with_gain(cfg.clone(), custom_gain);
    // with_gain bypasses compute_current_gain and stores the gain verbatim.
    close32(map.current_gain, custom_gain, 1e-12);
    // apply(x) == input_to_dv(x, gain).
    close32(map.apply(1.0), cfg.input_to_dv(1.0, custom_gain), 1e-12);
    // exact golden: input_to_dv(1.0, 2e-5) = dv_per_step(2e-5) = (2e-5 * 1e-6) / 10e-9 = 0.002
    close32(map.apply(1.0), 0.002, 1e-9);
}

#[test]
fn current_for_dv_round_trips_dv_per_step() {
    let cfg = HardwareConfig::default();
    let current = 7e-6f32;
    let dv = cfg.dv_per_step(current);
    // dv = (I * dt) / C = (7e-6 * 1e-6) / 10e-9 = 7e-4
    close32(dv, 0.0007, 1e-9);
    // current_for_dv is the inverse: (dv * C) / dt
    close32(cfg.current_for_dv(dv), current, 1e-11);
}

#[test]
fn input_to_dv_golden_for_fixed_input_and_gain() {
    let cfg = HardwareConfig::default();
    // input_to_dv(0.5, 3e-5) = dv_per_step(0.5 * 3e-5) = (1.5e-5 * 1e-6) / 10e-9 = 0.0015
    close32(cfg.input_to_dv(0.5, 3e-5), 0.0015, 1e-9);
}
