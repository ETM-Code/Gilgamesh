//! Characterization tests for src/network/*.

mod common;
use common::*;

use gilgamesh::data::InputEncoder;
use gilgamesh::network::{Network, NetworkGradients};
use gilgamesh::neurons::NeuronMode;
use ndarray::Array2;

const ZEROS10: [f32; 40] = [0.0; 40];

#[test]
fn forward_zeros_input_no_spikes_no_mem() {
    let net = Network::new(49, 100, 10, 0.9, 42);
    let input = Array2::<f32>::zeros((4, 49));
    let (sc, mem, caches) = net.forward(&input, 25);
    assert_eq!(sc.shape(), &[4, 10]);
    assert_eq!(caches.len(), 25);
    eq_arr(&sc, &ZEROS10);
    close32(mem.sum(), 0.0, 1e-9);
}

#[test]
fn forward_constant_input_default_physics_no_spikes() {
    // CHARACTERIZATION: default Physics mode (tau_m from beta, dt=1us) barely charges
    // the membrane, so a constant 0.1 input over 25 steps produces NO output spikes.
    let net = Network::new(49, 100, 10, 0.9, 42);
    let input = Array2::from_elem((4, 49), 0.1);
    let (sc, mem, _) = net.forward(&input, 25);
    eq_arr(&sc, &ZEROS10);
    // final membrane is also all-zero (locked).
    close_arr(&mem, &ZEROS10, 1e-4);
}

#[test]
fn forward_quantized_8_and_3_bit_golden() {
    let net = Network::new(49, 100, 10, 0.9, 42);
    let input = Array2::from_elem((4, 49), 0.1);
    let (q8, _, _) = net.forward_quantized(&input, 10, Some(8));
    let (q3, _, _) = net.forward_quantized(&input, 10, Some(3));
    eq_arr(&q8, &ZEROS10);
    eq_arr(&q3, &ZEROS10);
}

#[test]
fn forward_quantized_full_split_sign_golden() {
    let net = Network::new(49, 100, 10, 0.9, 42);
    let input = Array2::from_elem((4, 49), 0.1);
    let (qf, _, _) = net.forward_quantized_full(&input, 10, Some(3), true, 4);
    eq_arr(&qf, &ZEROS10);
}

#[test]
fn backward_bptt_gradient_sums_golden() {
    let net = Network::new(49, 100, 10, 0.9, 42);
    let input = Array2::from_elem((4, 49), 0.1);
    let (_, _, caches) = net.forward(&input, 5);
    let grad_output = Array2::from_elem((4, 10), 0.1);
    let grads = net.backward(&input, &caches, &grad_output);
    assert_eq!(grads.fc1_weight.shape(), net.fc1.weight.shape());
    assert_eq!(grads.fc2_weight.shape(), net.fc2.weight.shape());
    close32(grads.fc1_weight.sum(), 1.276706e-5, 1e-9);
    close32(grads.fc2_weight.sum(), 0.0, 1e-9);
    close32(grads.fc1_weight[[0, 0]], -1.6895282e-8, 1e-12);

    // Truncated BPTT through last 2 steps.
    let grads_t = net.backward_truncated(&input, &caches, &grad_output, Some(2));
    close32(grads_t.fc1_weight.sum(), 4.2404613e-6, 1e-9);
    close32(grads_t.fc2_weight.sum(), 0.0, 1e-9);
}

#[test]
fn forward_with_pulse_golden() {
    let net = Network::new_physics_with_pulse(49, 100, 10, 0.00949, 0.001, 0.00167, 4.42, 42);
    let input = Array2::from_elem((4, 49), 0.1);
    let (sc, mem, caches) = net.forward_with_pulse(&input, 25, 0.001);
    assert_eq!(caches.len(), 25);
    assert_eq!(mem.shape(), &[4, 10]);
    eq_arr(&sc, &ZEROS10);
}

#[test]
fn two_phase_pulse_step_golden() {
    let mut net = Network::new(49, 100, 10, 0.9, 42);
    net.spike_scale = 0.5; // triggers the two-phase t_on/t_off branch
    let input = Array2::from_elem((4, 49), 0.1);
    let state = net.init_state(4);
    let (combined, _, _, _) = net.forward_step(&input, &state);
    eq_arr(&combined, &ZEROS10);
}

#[test]
fn analog_gain_zero_matches_plain_forward() {
    let net = Network::new(49, 100, 10, 0.9, 42);
    let input = Array2::from_elem((4, 49), 0.1);
    let (analog0, _, _) = net.forward_with_analog(&input, 25, 0.0);
    let (plain, _, _) = net.forward(&input, 25);
    let diff: f32 = (&analog0 - &plain).mapv(|x| x.abs()).sum();
    close32(diff, 0.0, 1e-6);
}

#[test]
fn rate_coded_encoding_matches_plain_forward() {
    let net = Network::new(49, 100, 10, 0.9, 42);
    let input = Array2::from_elem((4, 49), 0.1);
    let enc = InputEncoder::rate_coded(7);
    let (enc_sc, _, _) = net.forward_with_encoding(&input, &enc, 25);
    let (plain, _, _) = net.forward(&input, 25);
    let diff: f32 = (&enc_sc - &plain).mapv(|x| x.abs()).sum();
    close32(diff, 0.0, 1e-6);
}

#[test]
fn forward_with_adaptation_and_analog_golden() {
    let anet =
        Network::new_physics_with_adaptation(49, 100, 10, 0.00949, 0.001, 0.001, 1.0, 1.2, 42);
    let input = Array2::from_elem((4, 49), 0.1);
    let (asc, _, _) = anet.forward_with_adaptation(&input, 25, 0.001);
    eq_arr(&asc, &ZEROS10);

    let net = Network::new(49, 100, 10, 0.9, 42);
    let (an1, _, _) = net.forward_with_analog(&input, 25, 0.1);
    eq_arr(&an1, &ZEROS10);
}

#[test]
fn spiking_input_accumulator_path_golden() {
    let mut net = Network::new(49, 100, 10, 0.9, 42);
    net.spiking_input = true; // init_input_accum seeds Xoshiro(0) deterministically
    let input = Array2::from_elem((4, 49), 0.1);
    let (sc, _, _) = net.forward_quantized(&input, 5, None);
    eq_arr(&sc, &ZEROS10);
}

#[test]
fn num_parameters_and_mode_flags() {
    let net = Network::new(49, 100, 10, 0.9, 42);
    assert_eq!(net.num_parameters(), 5900); // 49*100 + 100*10, no bias
    assert!(net.is_physics_mode());

    let mut net2 = net.clone();
    net2.set_mode(NeuronMode::Simple);
    assert!(!net2.is_physics_mode());
}

#[test]
fn gradients_total_norm_and_clip() {
    let net = Network::new(2, 2, 2, 0.9, 42);
    let mut g = NetworkGradients::zeros_like(&net);
    g.fc1_weight.fill(0.3); // 4 entries
    g.fc2_weight.fill(0.4); // 4 entries
    // total_norm = sqrt(4*0.09 + 4*0.16) = sqrt(1.0) = 1.0
    close32(g.total_norm(), 1.0, 1e-6);

    // clip below norm scales; returns original norm.
    let mut g_clip = g.clone();
    let returned = g_clip.clip_norm(0.5);
    close32(returned, 1.0, 1e-6);
    // scaled by 0.5/(1.0+1e-6)
    let scale = 0.5 / (1.0 + 1e-6);
    close32(g_clip.fc1_weight[[0, 0]], 0.3 * scale, 1e-6);

    // No-op when norm <= max.
    let mut g_noop = g.clone();
    let returned2 = g_noop.clip_norm(2.0);
    close32(returned2, 1.0, 1e-6);
    close32(g_noop.fc1_weight[[0, 0]], 0.3, 1e-9);
}
