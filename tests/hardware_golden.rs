//! Characterization tests for src/hardware.rs

mod common;
use common::*;

use gilgamesh::hardware::{HardwareConfig, HardwareMapping};
use ndarray::array;

#[test]
fn default_physics_quantities_golden() {
    let hw = HardwareConfig::default();
    // ported value
    close32(hw.v_threshold, 0.387, 1e-6);
    // tau_m = R*C = 120k * 10nF = 1.2ms
    close32(hw.tau_m(), 1.2e-3, 1e-6);
    // alpha = exp(-dt/tau)
    close32(hw.alpha(), (-1e-6f32 / 1.2e-3).exp(), 1e-9);
    close32(hw.alpha(), 0.999167, 1e-6);
    // i_threshold = v_threshold / r_leak
    close32(hw.i_threshold(), 0.387 / 120e3, 1e-12);
    close32(hw.i_threshold(), 3.225e-6, 1e-9);
    // dv per step for 1uA
    close32(hw.dv_per_step(1e-6), 0.0001, 1e-9);
    // current gain = C * Vth / (tau/4)
    close32(hw.compute_current_gain(), hw.c_mem * hw.v_threshold / (hw.tau_m() / 4.0), 1e-12);
    close32(hw.compute_current_gain(), 1.29e-5, 1e-7);
    // weight_to_current clamps to +/- i_syn_max (25uA)
    close32(hw.weight_to_current(100.0, hw.compute_current_gain()), 2.5e-5, 1e-9);
    close32(hw.weight_to_current(-100.0, hw.compute_current_gain()), -2.5e-5, 1e-9);
}

#[test]
fn hardware_mapping_apply_and_batch_golden() {
    let hw = HardwareConfig::default();
    let map = HardwareMapping::new(hw);
    close32(map.current_gain, 1.29e-5, 1e-7);
    close32(map.apply(1.0), 0.0012899999, 1e-7);

    let batch = map.apply_batch(&array![[1.0, 0.5], [0.0, -1.0]]);
    close_arr(&batch, &[0.0012899999, 0.00064499996, 0.0, -0.0012899999], 1e-7);
}
