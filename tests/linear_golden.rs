//! Characterization tests for src/layers/linear.rs

mod common;
use common::*;

use gilgamesh::layers::linear::{
    quantize_input, quantize_weights, quantize_weights_with_fixed_scale_and_defect, Linear,
    DEFAULT_SYNAPSE_NEG_GAIN, DEFAULT_SYNAPSE_POS_GAIN,
};
use ndarray::array;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn with_seed_kaiming_init_golden() {
    // Reproducibility anchor: Xoshiro seed 42, Uniform[-sqrt(1/3), +sqrt(1/3)].
    let layer = Linear::with_seed(3, 2, true, 42);
    close_arr(
        &layer.weight,
        &[0.3629282, -0.20920753, 0.5587528, 0.23225129, 0.33890975, 0.10172725],
        1e-6,
    );
    let bias: Vec<f32> = layer.bias.as_ref().unwrap().iter().cloned().collect();
    close_vec(&bias, &[-0.4326058, 0.12138492], 1e-6);
    // Default synapse gains.
    close32(layer.synapse_pos_gain, DEFAULT_SYNAPSE_POS_GAIN, 1e-9);
    close32(layer.synapse_neg_gain, DEFAULT_SYNAPSE_NEG_GAIN, 1e-9);
    close32(DEFAULT_SYNAPSE_POS_GAIN, 1.07, 1e-9);
    close32(DEFAULT_SYNAPSE_NEG_GAIN, 1.06, 1e-9);
}

#[test]
fn forward_applies_synapse_gain_then_bias_golden() {
    let layer = Linear::with_seed(3, 2, true, 42);
    let input = array![[1.0, 2.0, 3.0]];
    let out = layer.forward(&input);
    close_arr(&out, &[2.2393587, 0.7210951], 1e-5);
}

#[test]
fn forward_with_current_gain_and_cap() {
    let layer = Linear::with_seed(3, 2, true, 42)
        .with_current_gain(2.0)
        .with_total_current_cap(Some(1.0));
    let input = array![[1.0, 2.0, 3.0]];
    let out = layer.forward(&input);
    // Both outputs scaled by 2.0 then clamped to +/-1.0.
    for &v in out.iter() {
        assert!(v.abs() <= 1.0 + 1e-6);
    }
    // First output (2.239 * 2 = 4.48) clamps to 1.0; second (0.721*2=1.44) clamps to 1.0.
    close_arr(&out, &[1.0, 1.0], 1e-6);
}

#[test]
fn quantize_weights_split_sign_full_matrix_golden() {
    // pos_scale = max_pos/7 = 0.8/7; neg_scale = max_neg/7 = 0.3/7.
    let q = quantize_weights(&array![[0.1, 0.5, -0.3], [0.8, -0.2, 0.4]], 3);
    close_arr(
        &q,
        &[0.114285715, 0.45714286, -0.3, 0.8, -0.21428572, 0.45714286],
        1e-6,
    );
}

#[test]
fn quantize_input_dac_rounding_golden() {
    // levels = 7; round(x*7)/7.
    let q = quantize_input(&array![[0.0, 0.33, 0.5, 1.0]], 3);
    close_arr(&q, &[0.0, 0.2857143, 0.5714286, 1.0], 1e-6);
}

#[test]
fn quantize_broken_inhibitory_lsb_golden() {
    // Negative magnitudes forced even (&!1); positive path untouched; fixed scale 0.10.
    let q = quantize_weights_with_fixed_scale_and_defect(
        &array![[-0.10, -0.20, -0.30, -0.70, 0.35]],
        3,
        Some(0.10),
        Some(0.10),
        true,
    );
    close_arr(&q, &[0.0, -0.2, -0.2, -0.6, 0.4], 1e-6);

    // disable=false keeps the odd magnitude (-0.1 -> mag 1).
    let q2 = quantize_weights_with_fixed_scale_and_defect(
        &array![[-0.10, -0.20, -0.30, -0.70, 0.35]],
        3,
        Some(0.10),
        Some(0.10),
        false,
    );
    close_arr(&q2, &[-0.1, -0.2, -0.3, -0.7, 0.4], 1e-6);
}

#[test]
fn backward_with_synapse_gains_golden() {
    let layer = Linear::with_seed(3, 2, false, 42)
        .with_synapse_gains(1.1, 0.9)
        .with_current_gain(2.0);
    let input = array![[0.5, -0.3, 0.8]];
    let grad_output = array![[1.0, -0.5]];
    let (gi, gw, gb) = layer.backward(&input, &grad_output);
    assert!(gb.is_none());
    close_arr(&gi, &[0.98672885, 1.02023, 0.65404695], 1e-5);
    close_arr(&gw, &[1.1, -0.45, -0.66, 0.27, 1.7600001, -0.71999997], 1e-5);
}

#[test]
fn forward_noisy_seeded_golden_and_repeatable() {
    let layer = Linear::with_seed(3, 2, true, 42);
    let input = array![[1.0, 2.0, 3.0]];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let out = layer.forward_noisy(&input, 0.05, &mut rng);
    close_arr(&out, &[2.3272364, 0.7311407], 1e-5);

    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(7);
    let out2 = layer.forward_noisy(&input, 0.05, &mut rng2);
    assert_eq!(
        out.iter().cloned().collect::<Vec<f32>>(),
        out2.iter().cloned().collect::<Vec<f32>>()
    );
}

#[test]
fn num_parameters_with_and_without_bias() {
    assert_eq!(Linear::new(10, 5, true).num_parameters(), 55);
    assert_eq!(Linear::new(10, 5, false).num_parameters(), 50);
}
