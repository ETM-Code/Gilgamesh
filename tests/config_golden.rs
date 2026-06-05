//! Characterization tests for src/config/mod.rs

mod common;
use common::*;

use gilgamesh::config::{
    Config, OperationMode, OutputConfig, OutputMode, OutputModeType, PhysicsConfig,
    QuantizationConfig,
};
use gilgamesh::config::HardwareConfig as CfgHardwareConfig;

#[test]
fn default_config_constants() {
    let c = Config::default();
    assert_eq!(c.mode, OperationMode::Simple);
    assert_eq!(c.network.hidden_size, 12);
    assert_eq!(c.network.input_size, 36);
    assert_eq!(c.network.output_size, 10);
    assert_eq!(c.neuron.beta, 0.9);
    close32(c.physics.tau_m, 0.0012, 1e-9);
    close32(c.physics.tau_pulse, 1.5e-6, 1e-12);
    close32(c.physics.tau_theta, 0.000596, 1e-9);
    assert!(c.noise.enabled);
    close32(c.noise.weight_std, 0.05, 1e-9);
    close32(c.noise.threshold_std, 0.02, 1e-9);
    close32(c.noise.membrane_std, 0.01, 1e-9);
    close32(c.noise.input_std, 0.1, 1e-9);
    assert_eq!(c.quantization.bits, 3);
}

#[test]
fn config_save_load_round_trip() {
    let c = Config::default();
    let dir = std::env::temp_dir();
    let path = dir.join(format!("gilgamesh_cfg_{}.json", std::process::id()));
    c.save(&path).unwrap();
    let loaded = Config::load(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.mode, c.mode);
    assert_eq!(loaded.network.hidden_size, c.network.hidden_size);
    close32(loaded.neuron.beta, c.neuron.beta, 1e-9);
    close32(loaded.physics.tau_m, c.physics.tau_m, 1e-9);
    assert_eq!(loaded.quantization.bits, c.quantization.bits);
    assert!(loaded.is_physics_mode() == c.is_physics_mode());
}

#[test]
fn quantization_symmetric_and_asymmetric() {
    let mut q = QuantizationConfig::default();
    q.enabled = true;

    let mut sym = q.clone();
    sym.bits = 8;
    sym.symmetric = true;
    close32(sym.quantize(0.5), 0.5, 1e-6);
    // near-zero returns 0.0
    close32(sym.quantize(1e-9), 0.0, 1e-12);

    let mut asym = q.clone();
    asym.bits = 8;
    asym.symmetric = false;
    close32(asym.quantize(0.5), 0.5, 1e-6);

    // disabled returns input unchanged
    let mut dis = q.clone();
    dis.enabled = false;
    close32(dis.quantize(0.5), 0.5, 1e-12);
}

#[test]
fn physics_tau_beta_inverse() {
    let tau = PhysicsConfig::tau_from_beta(0.9, 0.001);
    close32(tau, 0.009491219, 1e-6);
    close32(PhysicsConfig::beta_from_tau(tau, 0.001), 0.9, 1e-6);
}

#[test]
fn hardware_pulse_peak_and_caps() {
    let hw = CfgHardwareConfig::default();
    close32(hw.pulse_peak(), 4.42, 1e-6); // 5.0 - 0.21 - 0.37
    close32(hw.synapse_scale_from_baseline(), 1.0, 1e-6); // 3.0 / 3.0
    assert_eq!(hw.total_cap_units(), None); // caps disabled

    let mut hw2 = CfgHardwareConfig::default();
    hw2.enable_current_caps = true;
    // total_current_max_ua(50)/synapse_current_max_ua(3) = 16.666...
    let units = hw2.total_cap_units().unwrap();
    close32(units, 16.666666, 1e-4);
}

#[test]
fn output_config_to_mode_mapping() {
    let mut oc = OutputConfig::default();
    assert_eq!(oc.to_mode(), OutputMode::SpikeCount);

    oc.mode = OutputModeType::AnalogFinal;
    assert_eq!(oc.to_mode(), OutputMode::AnalogFinal);

    oc.mode = OutputModeType::AnalogMax;
    assert_eq!(oc.to_mode(), OutputMode::AnalogMax);

    oc.mode = OutputModeType::AnalogFiltered;
    oc.filter_tau = 0.005;
    match oc.to_mode() {
        OutputMode::AnalogFiltered { tau_filter } => close32(tau_filter, 0.005, 1e-6),
        other => panic!("expected AnalogFiltered, got {other:?}"),
    }
}
