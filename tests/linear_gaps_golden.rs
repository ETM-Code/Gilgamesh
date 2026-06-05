//! Characterization tests for Linear synapse-drive helpers and quantize_weights
//! branches that linear_golden.rs does not cover.

mod common;
use common::*;

use gilgamesh::layers::linear::{quantize_weights, Linear};
use ndarray::{array, Array2};

fn flat(a: &Array2<f32>) -> Vec<f32> {
    a.iter().cloned().collect()
}

// ----- synapse_gain_for_weight sign dispatch -----

#[test]
fn synapse_gain_for_weight_sign_dispatch() {
    let layer = Linear::with_seed(3, 2, false, 42);
    // defaults: pos=1.07, neg=1.06
    close32(layer.synapse_gain_for_weight(1.0), 1.07, 1e-9);
    close32(layer.synapse_gain_for_weight(0.0), 1.07, 1e-9); // w>=0 -> pos
    close32(layer.synapse_gain_for_weight(-1.0), 1.06, 1e-9);
}

// ----- with_synapse_efficiency sets both gains -----

#[test]
fn with_synapse_efficiency_sets_both_gains() {
    let eff = Linear::with_seed(3, 2, false, 42).with_synapse_efficiency(0.8);
    close32(eff.synapse_pos_gain, 0.8, 1e-9);
    close32(eff.synapse_neg_gain, 0.8, 1e-9);
    // efficiency is floored at 0.0
    let clamped = Linear::with_seed(3, 2, false, 42).with_synapse_efficiency(-0.5);
    close32(clamped.synapse_pos_gain, 0.0, 1e-9);
    close32(clamped.synapse_neg_gain, 0.0, 1e-9);
}

// ----- apply_synapse_drive_model_inplace no-op fast path (gains == 1.0) -----

#[test]
fn synapse_drive_noop_when_both_gains_one() {
    let mut noop = Linear::with_seed(3, 2, false, 42);
    noop.synapse_pos_gain = 1.0;
    noop.synapse_neg_gain = 1.0;
    let input = array![[1.0, 2.0, 3.0]];
    // forward output must equal the raw dot product (no bias, no gain).
    let raw = input.dot(&noop.weight);
    let out = noop.forward(&input);
    assert_eq!(flat(&raw), flat(&out));
    close_arr(&out, &[2.4971628, 0.5604768], 1e-5);

    // Directly: apply_synapse_drive_model_inplace leaves the array untouched.
    let mut buf = raw.clone();
    noop.apply_synapse_drive_model_inplace(&mut buf);
    assert_eq!(flat(&buf), flat(&raw));
}

// ----- quantized_weight_matrix with fixed_quant_scale > 0 -----

#[test]
fn quantized_weight_matrix_fixed_scale_overrides_adaptive() {
    // fixed_quant_scale > 0 uses that scale for both signs, bypassing the
    // adaptive max-weight scaling.
    let mut fixed = Linear::with_seed(3, 2, false, 42);
    fixed.fixed_quant_scale = 0.1;
    let qm_fixed = fixed.quantized_weight_matrix(3);
    close_arr(&qm_fixed, &[0.4, -0.2, 0.6, 0.2, 0.3, 0.1], 1e-6);

    // adaptive (fixed_quant_scale = 0) produces a different matrix.
    let qm_adapt = Linear::with_seed(3, 2, false, 42).quantized_weight_matrix(3);
    close_arr(
        &qm_adapt,
        &[0.39910913, -0.20920753, 0.5587528, 0.23946548, 0.3192873, 0.079821825],
        1e-6,
    );
    assert_ne!(flat(&qm_fixed), flat(&qm_adapt));
}

// ----- quantize_weights adaptive fallback (scale=1.0) + 8-bit golden -----

#[test]
fn quantize_weights_all_positive_neg_scale_fallback_golden() {
    // No negative weights: max_neg = 0 -> neg_scale fallback to 1.0.
    // pos_scale = max_pos/7 = 0.8/7. round(0.1/(0.8/7)) = round(0.875) = 1 -> 0.114285...
    let q = quantize_weights(&array![[0.1, 0.5, 0.8]], 3);
    close_arr(&q, &[0.114285715, 0.45714286, 0.8], 1e-6);
}

#[test]
fn quantize_weights_all_zero_both_scales_fallback_golden() {
    // All-zero -> both max_pos and max_neg are 0 -> both scales fall back to 1.0.
    let q = quantize_weights(&Array2::<f32>::zeros((2, 2)), 3);
    close_arr(&q, &[0.0, 0.0, 0.0, 0.0], 1e-9);
}

#[test]
fn quantize_weights_8bit_adaptive_golden() {
    // 8-bit (255 magnitude levels), split-sign adaptive scaling.
    let m = array![[0.1, 0.5, -0.3], [0.8, -0.2, 0.4]];
    let q = quantize_weights(&m, 8);
    close_arr(
        &q,
        &[0.100392155, 0.49882352, -0.3, 0.8, -0.20000002, 0.40156862],
        1e-6,
    );
}
