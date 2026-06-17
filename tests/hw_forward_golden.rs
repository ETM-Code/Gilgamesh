//! Characterization tests for src/hw_forward.rs
//!
//! Locks the just-ported GND-referenced threshold divider, the DAC scaling,
//! and the two-phase pulse membrane integration end-to-end.

mod common;
use common::*;

use gilgamesh::hw_forward::{
    hw_forward_batch, hw_forward_batch_cfg, hw_forward_batch_with_thresholds, hw_forward_single,
    r_bottom_for_theta, theta_from_r_bottom, HwForwardConfig, R_BOTTOM_NOMINAL,
    R_BOTTOM_OUTPUT_FAITHFUL,
};
use ndarray::{array, Array2};

// ---- GND-referenced threshold: theta = V_DD * R_bot / (R_top + R_bot) ----

#[test]
fn theta_from_r_bottom_nominal_220k() {
    // V_DD=5.0, R_top=820k, R_bot=220k -> 5*220/(820+220) = 1.0576923...
    close64(theta_from_r_bottom(220e3), 1.0576923076923077, 1e-9);
}

#[test]
fn theta_from_r_bottom_150k() {
    close64(theta_from_r_bottom(150e3), 0.7731958762886598, 1e-9);
}

#[test]
fn theta_from_r_bottom_zero_is_zero() {
    // R_bottom = 0 -> divider collapses to GND -> 0 V.
    close64(theta_from_r_bottom(0.0), 0.0, 1e-12);
}

#[test]
fn r_bottom_nominal_constant_is_220k() {
    close64(R_BOTTOM_NOMINAL, 220e3, 1e-6);
}

// ---- Inverse round-trip ----

#[test]
fn r_bottom_for_theta_inverts_theta() {
    let r = r_bottom_for_theta(theta_from_r_bottom(220e3));
    assert!((r - 220e3).abs() / 220e3 < 1e-6, "got {r}");
}

#[test]
fn r_bottom_for_theta_absolute_golden() {
    close64(r_bottom_for_theta(0.387), 68792.54281378713, 1e-3);
}

// ---- duty_cycle clamp (verified analytically; duty_cycle is private) ----

#[test]
fn duty_cycle_clamps_to_full_timestep() {
    // CHARACTERIZATION: current behavior, possibly surprising (full-timestep pulse).
    // v_peak = V_DD - 0.56 = 4.44; tau_pulse = R_STRETCH*C_STRETCH = 150e3*5.8e-9 = 8.7e-4.
    // t_on = tau_pulse * ln(v_peak / V_BE) = 8.7e-4 * ln(4.44/0.65).
    // (t_on / DT) with DT=0.001 is far above 1, so duty clamps to 1.0 => t_on=DT, t_off=0.
    let v_peak = 5.0_f64 - 0.56;
    let tau_pulse = 150e3_f64 * 5.8e-9;
    let t_on = tau_pulse * (v_peak / 0.65).ln();
    let dt = 0.001_f64;
    let duty = (t_on / dt).min(1.0);
    assert_eq!(duty, 1.0, "duty must clamp to 1.0, raw ratio {}", t_on / dt);
}

// ---- hw_forward_batch: zero input -> zero spikes ----

#[test]
fn hw_forward_batch_zero_input_zero_spikes() {
    let fc1 = Array2::<f32>::zeros((1, 9));
    let fc2: Vec<Vec<i8>> = (0..9).map(|_| vec![0i8; 10]).collect();
    let out = hw_forward_batch(&fc1, &fc2, 1.16, 25);
    assert_eq!(out.shape(), &[1, 10]);
    eq_arr(&out, &[0.0; 10]);
}

// Distinct fc2 columns so per-output spike counts differ; locks GND membrane + DAC end-to-end.
fn golden_fc2() -> Vec<Vec<i8>> {
    vec![
        vec![7, 6, 5, 4, 3, 2, 1, 0, -1, -2],
        vec![6, 5, 4, 3, 2, 1, 0, -1, -2, -3],
        vec![5, 4, 3, 2, 1, 0, 7, 6, 5, 4],
        vec![4, 3, 2, 1, 0, 7, 6, 5, 4, 3],
        vec![3, 2, 1, 0, 7, 6, 5, 4, 3, 2],
        vec![2, 1, 0, 7, 6, 5, 4, 3, 2, 1],
        vec![1, 0, 7, 6, 5, 4, 3, 2, 1, 0],
        vec![0, 7, 6, 5, 4, 3, 2, 1, 0, 7],
        vec![7, 0, 6, 1, 5, 2, 4, 3, 7, 0],
    ]
}

#[test]
fn hw_forward_batch_nonzero_golden() {
    let fc1: Array2<f32> = array![[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]];
    let out = hw_forward_batch(&fc1, &golden_fc2(), 1.16, 25);
    eq_arr(
        &out,
        &[18.0, 12.0, 16.0, 12.0, 16.0, 12.0, 16.0, 8.0, 0.0, 0.0],
    );
}

#[test]
fn hw_forward_batch_two_rows_golden() {
    let fc1: Array2<f32> = array![
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [3.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    ];
    let out = hw_forward_batch(&fc1, &golden_fc2(), 1.16, 25);
    eq_arr(
        &out,
        &[
            18.0, 12.0, 16.0, 12.0, 16.0, 12.0, 16.0, 8.0, 0.0, 0.0, // row 0
            16.0, 8.0, 16.0, 12.0, 16.0, 12.0, 16.0, 8.0, 4.0, 0.0, // row 1
        ],
    );
}

#[test]
fn hw_forward_batch_with_per_layer_thresholds_golden() {
    let fc1: Array2<f32> = array![[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]];
    let hidden_theta = theta_from_r_bottom(220e3);
    let output_theta = theta_from_r_bottom(150e3);
    let out = hw_forward_batch_with_thresholds(
        &fc1,
        &golden_fc2(),
        1.16,
        25,
        hidden_theta,
        output_theta,
    );
    // Lower output threshold => more output spikes than the shared-threshold path.
    eq_arr(
        &out,
        &[25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 15.0, 12.0, 0.0],
    );
}

#[test]
fn hw_forward_single_matches_batch_row0() {
    let fc1_vec = vec![2.0f64; 9];
    let single = hw_forward_single(&fc1_vec, &golden_fc2(), 1.16, 25);
    let expected = [18.0, 12.0, 16.0, 12.0, 16.0, 12.0, 16.0, 8.0, 0.0, 0.0];
    assert_eq!(single.len(), 10);
    for (s, e) in single.iter().zip(expected.iter()) {
        close64(*s, *e, 1e-6);
    }
}

#[test]
fn hw_forward_dac_clamp_high_input_saturates() {
    // Very large fc1 clamps v_dac to DAC_VREF=5.0; spike counts saturate.
    let fc2: Vec<Vec<i8>> = vec![vec![7i8; 10]; 9];
    let fc1: Array2<f32> = Array2::from_elem((1, 9), 1000.0);
    let out = hw_forward_batch(&fc1, &fc2, 1.16, 25);
    eq_arr(&out, &[37.0; 10]);
}

#[test]
fn hw_forward_negative_input_yields_no_current() {
    // Negative fc1 -> v_dac <= V_BE -> zero hidden current -> zero output spikes.
    let fc1: Array2<f32> = Array2::from_elem((1, 9), -5.0);
    let out = hw_forward_batch(&fc1, &golden_fc2(), 1.16, 25);
    eq_arr(&out, &[0.0; 10]);
}

// ---- HwForwardConfig (faithful as-built board) ----

#[test]
fn config_default_is_faithful_as_built() {
    let cfg = HwForwardConfig::default();
    // Hidden at nominal 1.058 V, output at the real 0.543 V (150k‖300k = 100k).
    close64(cfg.hidden_theta, theta_from_r_bottom(R_BOTTOM_NOMINAL), 1e-12);
    close64(cfg.output_theta, theta_from_r_bottom(R_BOTTOM_OUTPUT_FAITHFUL), 1e-12);
    close64(cfg.output_theta, 0.5434782608695652, 1e-9);
    assert_eq!(cfg.masked_outputs, vec![1, 9]); // O2, O10
    assert_eq!(cfg.mirror_mismatch_cv, 0.0); // matched / deterministic by default
    assert_eq!(cfg.dac_scale, 1.16);
    assert_eq!(cfg.num_steps, 25);
}

#[test]
fn config_no_mask_matched_equals_legacy_with_thresholds() {
    // A config with empty mask + cv=0 must reproduce the legacy entry point exactly.
    let fc1: Array2<f32> = array![[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]];
    let cfg = HwForwardConfig {
        hidden_theta: theta_from_r_bottom(220e3),
        output_theta: theta_from_r_bottom(150e3),
        dac_scale: 1.16,
        num_steps: 25,
        masked_outputs: Vec::new(),
        mirror_mismatch_cv: 0.0,
        mismatch_seed: 0,
    };
    let via_cfg = hw_forward_batch_cfg(&fc1, &golden_fc2(), &cfg);
    let via_legacy = hw_forward_batch_with_thresholds(
        &fc1,
        &golden_fc2(),
        1.16,
        25,
        theta_from_r_bottom(220e3),
        theta_from_r_bottom(150e3),
    );
    eq_arr(&via_cfg, via_legacy.as_slice().unwrap());
}

#[test]
fn config_masks_o2_and_o10() {
    // Same as the per-layer golden, but with the default O2/O10 mask applied:
    // indices 1 and 9 must read zero, the rest unchanged.
    let fc1: Array2<f32> = array![[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]];
    let cfg = HwForwardConfig {
        hidden_theta: theta_from_r_bottom(220e3),
        output_theta: theta_from_r_bottom(150e3),
        masked_outputs: vec![1, 9],
        ..HwForwardConfig::default()
    };
    let out = hw_forward_batch_cfg(&fc1, &golden_fc2(), &cfg);
    // cf. hw_forward_batch_with_per_layer_thresholds_golden: [25,25,25,25,25,25,25,15,12,0]
    // masking idx1 (O2)->0; idx9 (O10) was already 0; idx8 (O9)=12 untouched.
    eq_arr(
        &out,
        &[25.0, 0.0, 25.0, 25.0, 25.0, 25.0, 25.0, 15.0, 12.0, 0.0],
    );
}

#[test]
fn config_mismatch_is_deterministic_and_perturbs() {
    let fc1: Array2<f32> = array![[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]];
    let base = HwForwardConfig {
        masked_outputs: Vec::new(),
        ..HwForwardConfig::default()
    };
    let mut mm = base.clone();
    mm.mirror_mismatch_cv = 0.05;
    mm.mismatch_seed = 42;
    let a = hw_forward_batch_cfg(&fc1, &golden_fc2(), &mm);
    let b = hw_forward_batch_cfg(&fc1, &golden_fc2(), &mm);
    // Same seed => identical (deterministic).
    eq_arr(&a, b.as_slice().unwrap());
    // cv=0 baseline differs from cv=0.05 (mismatch actually does something).
    let matched = hw_forward_batch_cfg(&fc1, &golden_fc2(), &base);
    assert!(
        a != matched,
        "cv=0.05 should perturb counts vs matched mirrors"
    );
}
