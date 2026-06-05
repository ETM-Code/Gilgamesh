//! Characterization tests for src/checkpoint.rs

mod common;
use common::*;

use gilgamesh::checkpoint::Checkpoint;
use gilgamesh::layers::linear::{DEFAULT_SYNAPSE_NEG_GAIN, DEFAULT_SYNAPSE_POS_GAIN};
use gilgamesh::network::Network;
use ndarray::Array2;
use std::path::Path;

#[test]
fn save_load_round_trip_exact_weights() {
    let net = Network::new(36, 100, 10, 0.9, 42);
    let cp = Checkpoint::from_network(&net, None);

    let path = std::env::temp_dir().join(format!("gilgamesh_ckpt_{}.json", std::process::id()));
    cp.save(&path).unwrap();
    let loaded = Checkpoint::load(&path).unwrap();
    std::fs::remove_file(&path).ok();

    let net2 = loaded.to_network().unwrap();

    assert_eq!(net.fc1.weight.shape(), net2.fc1.weight.shape());
    assert_eq!(net.fc2.weight.shape(), net2.fc2.weight.shape());
    let d1: f32 = (&net.fc1.weight - &net2.fc1.weight).mapv(|x| x.abs()).sum();
    let d2: f32 = (&net.fc2.weight - &net2.fc2.weight).mapv(|x| x.abs()).sum();
    assert!(d1 < 1e-6, "fc1 diff {d1}");
    assert!(d2 < 1e-6, "fc2 diff {d2}");

    assert_eq!(loaded.architecture.input_size, 36);
    assert_eq!(loaded.architecture.hidden_size, 100);
    assert_eq!(loaded.architecture.output_size, 10);
    assert_eq!(loaded.architecture.mode, "physics");
    close32(net2.fc1.synapse_pos_gain, DEFAULT_SYNAPSE_POS_GAIN, 1e-9);
    close32(net2.fc2.synapse_neg_gain, DEFAULT_SYNAPSE_NEG_GAIN, 1e-9);
}

#[test]
fn from_network_quantized_exported_integers_golden() {
    let net = Network::new(36, 100, 10, 0.9, 42);
    let cp = Checkpoint::from_network_quantized(&net, None, Some(3), Some((6, 6)));
    let q = cp.quantized.as_ref().unwrap();
    assert_eq!(q.magnitude_bits, 3);
    assert_eq!(q.max_magnitude, 7);
    close32(q.fc1_pos_scale, 0.023807501, 1e-7);
    close32(q.fc1_neg_scale, 0.023800386, 1e-7);
    assert_eq!(&q.fc1_weight[0][0..5], &[4, -3, 7, 3, 4]);
    assert_eq!(cp.architecture.image_width, Some(6));
    assert_eq!(cp.architecture.image_height, Some(6));
}

#[test]
fn load_version_too_new_rejected() {
    let json = r#"{
        "version":2,
        "architecture":{"input_size":2,"hidden_size":2,"output_size":2,"mode":"physics"},
        "weights":{"fc1_weight":[[0.1,0.2],[0.3,0.4]],"fc2_weight":[[0.5,0.6],[0.7,0.8]]}
    }"#;
    let path = std::env::temp_dir().join(format!("gilgamesh_ckpt_v2_{}.json", std::process::id()));
    std::fs::write(&path, json).unwrap();
    let res = Checkpoint::load(&path);
    std::fs::remove_file(&path).ok();
    assert!(res.is_err(), "version 2 must be rejected");
}

#[test]
fn to_network_defaults_for_missing_optional_fields() {
    let minimal = r#"{
        "version":1,
        "architecture":{"input_size":2,"hidden_size":2,"output_size":2,"mode":"physics"},
        "weights":{"fc1_weight":[[0.1,0.2],[0.3,0.4]],"fc2_weight":[[0.5,0.6],[0.7,0.8]]}
    }"#;
    let cp: Checkpoint = serde_json::from_str(minimal).unwrap();
    assert_eq!(cp.architecture.threshold, 1.0);
    assert_eq!(cp.architecture.slope, 25.0);

    let net = cp.to_network().unwrap();
    close32(net.lif1.threshold, 1.0, 1e-9);
    close32(net.lif1.spike_grad.slope(), 25.0, 1e-9);
    close32(net.fc1.synapse_pos_gain, DEFAULT_SYNAPSE_POS_GAIN, 1e-9);
    close32(net.fc1.synapse_neg_gain, DEFAULT_SYNAPSE_NEG_GAIN, 1e-9);
    close32(net.spike_scale, 1.0, 1e-9);
    assert!(net.dac_max.is_infinite());
    // physics + missing tau_m -> fallback 0.0026.
    close32(net.lif1.mode.tau_m().unwrap(), 0.0026, 1e-9);
    assert!(net.is_physics_mode());
}

#[test]
fn emulator_fixture_golden_path_guarded() {
    let cp_path = "models/tarski_36_9_10_emulator.json";
    if !Path::new(cp_path).exists() {
        eprintln!("Skipping emulator fixture test: {cp_path} not found");
        return;
    }
    let cp = Checkpoint::load(cp_path).unwrap();
    let net = cp.to_network().unwrap();
    assert_eq!(net.fc1.weight.shape(), &[36, 9]);
    assert_eq!(net.fc2.weight.shape(), &[9, 10]);
    close32(net.lif1.threshold, 1.0, 1e-9);
    assert_eq!(cp.architecture.mode, "physics");
    close32(net.lif1.mode.tau_pulse(), 1.5e-6, 1e-12);

    // Real ported-model characterization input (digit 7, 6x6 normalized).
    let input_vec: Vec<f32> = vec![
        -0.424, -0.424, -0.424, -0.424, -0.424, -0.424, -0.424, -0.424, 0.246, 0.912, 0.415,
        -0.424, -0.424, -0.250, 0.744, 0.580, 0.912, -0.424, -0.424, -0.424, -0.424, 0.415, 0.580,
        -0.424, -0.424, -0.424, 0.080, 0.746, 0.246, -0.424, -0.424, -0.424, 0.415, 0.580, -0.250,
        -0.424,
    ];
    let input = Array2::from_shape_vec((1, 36), input_vec).unwrap();
    let (sc, _, _) = net.forward(&input, 25);
    // CHARACTERIZATION: current ported model produces zero output spikes on this
    // input under the default Physics (1us dt) integration. Locked.
    eq_arr(&sc, &[0.0; 10]);
}

#[test]
fn vec_to_array2_errors_on_empty_and_ragged() {
    // Empty rows.
    let empty = r#"{
        "version":1,
        "architecture":{"input_size":0,"hidden_size":2,"output_size":2,"mode":"physics"},
        "weights":{"fc1_weight":[],"fc2_weight":[[0.5,0.6],[0.7,0.8]]}
    }"#;
    let cp: Checkpoint = serde_json::from_str(empty).unwrap();
    assert!(cp.to_network().is_err(), "empty fc1 must error");

    // Ragged rows.
    let ragged = r#"{
        "version":1,
        "architecture":{"input_size":2,"hidden_size":2,"output_size":2,"mode":"physics"},
        "weights":{"fc1_weight":[[0.1,0.2],[0.3]],"fc2_weight":[[0.5,0.6],[0.7,0.8]]}
    }"#;
    let cp2: Checkpoint = serde_json::from_str(ragged).unwrap();
    assert!(cp2.to_network().is_err(), "ragged fc1 must error");
}
