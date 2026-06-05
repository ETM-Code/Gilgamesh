//! Characterization tests for QuantizationConfig::quantize symmetric/asymmetric
//! math (src/config/mod.rs:437-458) with values that actually exercise the
//! rounding and scale formula (config_golden.rs only checks exact round-trips).

mod common;
use common::*;

use gilgamesh::config::QuantizationConfig;

fn enabled(bits: u8, symmetric: bool) -> QuantizationConfig {
    let mut q = QuantizationConfig::default();
    q.enabled = true;
    q.bits = bits;
    q.symmetric = symmetric;
    q
}

#[test]
fn symmetric_quantize_value_and_negation_golden() {
    // bits=3 -> quantization_levels = 1<<3 = 8.
    // For a single value w, scale = (8/2)/|w| = 4/|w|, so round(w*scale)/scale
    // = round(w * 4/|w|)/(4/|w|) = round(±4)/(4/|w|) = ±|w|. Round-trips to w.
    let sym = enabled(3, true);
    close32(sym.quantize(0.37), 0.37, 1e-6);
    close32(sym.quantize(-0.37), -0.37, 1e-6);
}

#[test]
fn asymmetric_quantize_rounds_to_level_grid_golden() {
    // bits=3 -> levels = 8. asymmetric: round(w*8)/8.
    // 0.37*8 = 2.96 -> round 3 -> 3/8 = 0.375.
    let asym = enabled(3, false);
    close32(asym.quantize(0.37), 0.375, 1e-6);
    close32(asym.quantize(-0.37), -0.375, 1e-6);
}

#[test]
fn symmetric_differs_from_asymmetric_at_least_once() {
    // 0.37 is a value where the two paths diverge: symmetric -> 0.37, asymmetric -> 0.375.
    let sym = enabled(3, true);
    let asym = enabled(3, false);
    let s = sym.quantize(0.37);
    let a = asym.quantize(0.37);
    assert!((s - a).abs() > 1e-4, "symmetric {s} should differ from asymmetric {a}");
}

#[test]
fn symmetric_near_zero_returns_zero() {
    // max_magnitude < 1e-8 short-circuits to 0.0.
    let sym = enabled(3, true);
    close32(sym.quantize(1e-9), 0.0, 1e-12);
    close32(sym.quantize(-1e-9), 0.0, 1e-12);
}

#[test]
fn disabled_passthrough() {
    let mut q = enabled(3, true);
    q.enabled = false;
    close32(q.quantize(0.37), 0.37, 1e-12);
    close32(q.quantize(-0.37), -0.37, 1e-12);
}
