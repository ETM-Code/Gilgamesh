//! Characterization tests closing coverage gaps in src/network/* that the
//! existing network_golden.rs leaves at all-zeros (default Physics mode).
//!
//! Every golden constant here was produced by RUNNING the current code and
//! pasting the observed value. The purpose is to lock current behavior so any
//! future change is detected, NOT to assert correctness.

mod common;
use common::*;

use gilgamesh::network::{Network, NetworkGradients};
use gilgamesh::neurons::NeuronMode;
use ndarray::Array2;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// A network forced into Simple mode (so the membrane actually charges and
/// spikes occur), with low LIF thresholds.
fn simple_net(threshold: f32) -> Network {
    let mut net = Network::new(49, 100, 10, 0.9, 42);
    net.set_mode(NeuronMode::Simple);
    net.lif1.threshold = threshold;
    net.lif2.threshold = threshold;
    net
}

fn flat(a: &Array2<f32>) -> Vec<f32> {
    a.iter().cloned().collect()
}

// ============================================================================
// forward_step_quantized two-phase pulse branch (network.rs:190-215)
// ============================================================================

#[test]
fn two_phase_pulse_step_nonzero_spikes_golden() {
    // Simple mode + low threshold + strong input -> the two-phase t_on/t_off
    // branch (spike_scale != 1.0) actually produces non-zero output spikes.
    let mut net = simple_net(0.05);
    net.spike_scale = 0.5;
    let input = Array2::from_elem((1, 49), 1.0);
    let state = net.init_state(1);
    let (spikes, mem, _, _) = net.forward_step(&input, &state);
    // OR-combine of phase1/phase2 spikes, clamped to {0,1}.
    eq_arr(&spikes, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    close_arr(
        &mem,
        &[
            0.041529644, -0.08159326, -0.14729768, 0.2622801, 0.056067754, -0.3433152, 0.32774195,
            -0.40142238, 0.1859488, -0.29397586,
        ],
        1e-6,
    );
}

#[test]
fn two_phase_vs_single_phase_membrane_differs() {
    // Same input/threshold, spike_scale=0.5 (two-phase) vs 1.0 (single-phase).
    // The OR-combine path yields the SAME spike set here but a DIFFERENT final
    // membrane, pinning that the two-phase split actually changes the result.
    let input = Array2::from_elem((1, 49), 1.0);

    let mut two = simple_net(0.05);
    two.spike_scale = 0.5;
    let (sp_two, mem_two, _, _) = two.forward_step(&input, &two.init_state(1));

    let mut one = simple_net(0.05);
    one.spike_scale = 1.0;
    let (sp_one, mem_one, _, _) = one.forward_step(&input, &one.init_state(1));

    // Spikes coincide for this input ...
    assert_eq!(flat(&sp_two), flat(&sp_one));
    // ... but the membranes differ (single-phase keeps more charge: no phase-2 leak).
    close_arr(
        &mem_one,
        &[
            0.04614405, -0.09065918, -0.16366409, 0.34697792, 0.11785306, -0.38146135, 0.41971332,
            -0.4460249, 0.26216534, -0.32663986,
        ],
        1e-6,
    );
    let diff: f32 = (&mem_two - &mem_one).mapv(|x| x.abs()).sum();
    assert!(diff > 1e-3, "two-phase and single-phase membranes must differ, diff={diff}");
}

#[test]
fn spike_scale_split_no_membrane_effect_in_simple_mode() {
    // CHARACTERIZATION: current behavior, possibly surprising, locked to detect change.
    // In Simple mode, compute_membrane_with_dt IGNORES dt (mem = beta*mem + input),
    // so the t_on/t_off split (spike_scale 0.25 vs 0.75) produces an IDENTICAL final
    // membrane. Only the two-phase-vs-single-phase distinction matters here.
    let input = Array2::from_elem((1, 49), 1.0);
    let golden_mem = [
        0.041529644, -0.08159326, -0.14729768, 0.2622801, 0.056067754, -0.3433152, 0.32774195,
        -0.40142238, 0.1859488, -0.29397586,
    ];
    for sc in [0.25f32, 0.5, 0.75] {
        let mut net = simple_net(0.05);
        net.spike_scale = sc;
        let (_, mem, _, _) = net.forward_step(&input, &net.init_state(1));
        close_arr(&mem, &golden_mem, 1e-6);
    }
}

#[test]
fn two_phase_multistep_forward_step_golden() {
    // Multi-step accumulation through repeated forward_step calls in the two-phase path.
    let mut net = simple_net(0.05);
    net.spike_scale = 0.5;
    let input = Array2::from_elem((1, 49), 1.0);
    let mut state = net.init_state(1);
    let mut total = Array2::<f32>::zeros((1, 10));
    let mut last_mem = Array2::<f32>::zeros((1, 10));
    for _ in 0..10 {
        let (sp, mem, ns, _) = net.forward_step(&input, &state);
        total = &total + &sp;
        last_mem = mem;
        state = ns;
    }
    eq_arr(&total, &[9.0, 0.0, 0.0, 10.0, 7.0, 0.0, 10.0, 0.0, 10.0, 0.0]);
    close_arr(
        &last_mem,
        &[
            0.4467374, -1.1165925, -0.2413457, 1.248587, 0.033265363, -1.8156611, 1.5438948,
            -1.3995532, 0.77106136, -1.0422359,
        ],
        1e-4,
    );
}

// ============================================================================
// dac_max clamp in forward_step_quantized (network.rs:167-171)
// ============================================================================

#[test]
fn dac_max_upper_clip_changes_step_output() {
    // forward_step_quantized clamps fc1 output to [0, dac_max]. With a hidden
    // threshold of 0.1, a dac_max of 0.08 clamps every positive hidden current
    // below threshold -> NO hidden spikes -> all-zero output. dac_max=INFINITY
    // leaves them large -> the hidden layer fires and the output spikes.
    let mut net = Network::new(49, 100, 10, 0.9, 42);
    net.set_mode(NeuronMode::Simple);
    net.lif1.threshold = 0.1;
    net.lif2.threshold = 0.05;
    let mut input = Array2::zeros((1, 49));
    for j in 0..49 {
        input[[0, j]] = if j % 2 == 0 { 5.0 } else { -5.0 };
    }

    let mut clipped = net.clone();
    clipped.dac_max = 0.08;
    let (sp_clip, _, _, _) = clipped.forward_step(&input, &clipped.init_state(1));
    eq_arr(&sp_clip, &[0.0; 10]);

    let mut unclamped = net.clone();
    unclamped.dac_max = f32::INFINITY;
    let (sp_inf, mem_inf, _, _) = unclamped.forward_step(&input, &unclamped.init_state(1));
    eq_arr(&sp_inf, &[1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    close32(mem_inf.sum(), 0.49054605, 1e-5);

    assert_ne!(flat(&sp_clip), flat(&sp_inf));
}

#[test]
fn dac_max_negative_zeroing_membrane_golden() {
    // Negative fc1 outputs are zeroed by clamp(0.0, dac_max). With a high hidden
    // threshold (1.0) and finite dac_max, zeroed-and-clipped currents leave the
    // hidden layer silent so the output membrane is exactly zero; the unclamped
    // path retains a nonzero membrane.
    let mut net = Network::new(49, 100, 10, 0.9, 42);
    net.set_mode(NeuronMode::Simple);
    net.lif1.threshold = 1.0;
    net.lif2.threshold = 1.0;
    let mut input = Array2::zeros((1, 49));
    for j in 0..49 {
        input[[0, j]] = if j % 2 == 0 { 5.0 } else { -5.0 };
    }

    let mut clamped = net.clone();
    clamped.dac_max = 0.5;
    let (_, mem_clamp, _, _) = clamped.forward_step(&input, &clamped.init_state(1));
    close32(mem_clamp.sum(), 0.0, 1e-9);

    let mut unclamped = net.clone();
    unclamped.dac_max = f32::INFINITY;
    let (_, mem_inf, _, _) = unclamped.forward_step(&input, &unclamped.init_state(1));
    close32(mem_inf.sum(), 0.6229335, 1e-5);
}

// ============================================================================
// forward_traced + SimulationTrace (network.rs:600, trace.rs)
// ============================================================================

#[test]
fn forward_traced_shapes_and_values_golden() {
    let net = simple_net(0.05);
    let input = Array2::from_elem((1, 49), 1.0);
    let trace = net.forward_traced(&input, 6);

    // Shape / length of the trace.
    assert_eq!(trace.num_timesteps(), 6);
    assert_eq!(trace.hidden_mem_history.len(), 6);
    assert_eq!(trace.output_mem_history.len(), 6);
    assert_eq!(trace.hidden_spike_history.len(), 6);
    assert_eq!(trace.output_spike_history.len(), 6);
    assert_eq!(trace.hidden_current_history.len(), 6);
    assert_eq!(trace.output_current_history.len(), 6);
    assert_eq!(trace.hidden_mem_history[0].shape(), &[1, 100]);
    assert_eq!(trace.output_mem_history[0].shape(), &[1, 10]);

    // Accumulated spike count + final membrane.
    eq_arr(
        &trace.output_spike_count,
        &[5.0, 0.0, 0.0, 6.0, 6.0, 0.0, 6.0, 0.0, 6.0, 0.0],
    );
    close_arr(
        &trace.output_final_mem,
        &[
            0.77309185, -1.2688524, -0.28653586, 1.706843, 0.0036779828, -2.058001, 1.9067723,
            -1.5831409, 1.0961466, -1.214547,
        ],
        1e-5,
    );

    // A few exact (step, neuron) probes.
    close32(trace.output_mem_history[0][[0, 0]], 0.04614405, 1e-6);
    close32(trace.output_mem_history[5][[0, 0]], 0.77309185, 1e-6);
    close32(trace.output_spike_history[5][[0, 0]], 1.0, 1e-9);
    close32(trace.hidden_spike_history[0].sum(), 52.0, 1e-6);
    close32(trace.hidden_mem_history[0][[0, 0]], 0.43908033, 1e-6);
    // In Simple mode the first-step output current equals the first-step output membrane.
    close32(trace.output_current_history[0][[0, 0]], 0.04614405, 1e-6);
}

// ============================================================================
// forward_noisy full noisy network path (forward_noisy.rs)
// ============================================================================

#[test]
fn forward_noisy_seeded_golden_and_repeatable() {
    let net = simple_net(0.05);
    let input = Array2::from_elem((1, 49), 1.0);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let (sc, fm, caches) = net.forward_noisy(&input, 8, 0.05, 0.02, 0.01, 0.1, &mut rng);
    assert_eq!(caches.len(), 8);
    eq_arr(&sc, &[6.0, 0.0, 0.0, 8.0, 7.0, 0.0, 8.0, 0.0, 8.0, 0.0]);
    close_arr(
        &fm,
        &[
            0.50321037, -1.2137272, -0.43309796, 1.9837952, 0.49420944, -2.3163338, 2.5649707,
            -2.329537, 1.3010632, -2.1514463,
        ],
        1e-4,
    );

    // Same seed -> identical (pins the fc1_weight -> fc2_weight -> per-step draw order).
    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(7);
    let (sc2, fm2, _) = net.forward_noisy(&input, 8, 0.05, 0.02, 0.01, 0.1, &mut rng2);
    assert_eq!(flat(&sc), flat(&sc2));
    assert_eq!(flat(&fm), flat(&fm2));
}

#[test]
fn forward_noisy_spiking_input_draw_order_golden() {
    // spiking_input=true exercises the normalize+clamp accumulator path with its
    // own RNG draw ordering inside forward_noisy.
    let mut net = simple_net(0.05);
    net.spiking_input = true;
    let input = Array2::from_elem((1, 49), 1.0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let (sc, fm, _) = net.forward_noisy(&input, 8, 0.05, 0.02, 0.01, 0.1, &mut rng);
    eq_arr(&sc, &[5.0, 0.0, 1.0, 8.0, 6.0, 0.0, 8.0, 0.0, 8.0, 0.0]);
    close_arr(
        &fm,
        &[
            0.1918156, -0.9798884, -0.98759604, 1.6103461, 1.1874689, -2.101699, 1.9300312,
            -2.5154316, 1.679041, -2.0845912,
        ],
        1e-4,
    );
}

// ============================================================================
// apply_gradients plain SGD update (network.rs:571)
// ============================================================================

#[test]
fn apply_gradients_sgd_update_golden() {
    let mut net = Network::new(4, 3, 2, 0.9, 42);
    let fc1_orig = net.fc1.weight[[0, 0]];
    let fc2_orig = net.fc2.weight[[0, 0]];
    close32(fc1_orig, 0.31430507, 1e-6);
    close32(fc2_orig, -0.38329852, 1e-6);

    let mut g = NetworkGradients::zeros_like(&net);
    g.fc1_weight.fill(0.5);
    g.fc2_weight.fill(0.25);
    net.apply_gradients(&g, 0.1);

    // weight -= grad * lr
    close32(net.fc1.weight[[0, 0]], fc1_orig - 0.5 * 0.1, 1e-6);
    close32(net.fc2.weight[[0, 0]], fc2_orig - 0.25 * 0.1, 1e-6);
    close32(net.fc1.weight[[0, 0]], 0.26430506, 1e-6);
    close32(net.fc2.weight[[0, 0]], -0.40829852, 1e-6);
}

// ============================================================================
// forward_quantized_full split-sign + broken-inhibitory + spiking output
// ============================================================================

#[test]
fn forward_quantized_full_broken_inhibitory_changes_spiking_output() {
    // Simple mode + low threshold so the full multi-step forward actually spikes,
    // with fixed_quant_scale set so the broken-inhibitory-LSB defect model is the
    // only difference between the two runs.
    let mut net = simple_net(0.05);
    let input = Array2::from_elem((1, 49), 1.0);

    let mut intact = net.clone();
    intact.fc1.fixed_quant_scale = 0.1;
    intact.fc2.fixed_quant_scale = 0.1;
    intact.fc1.disable_inhibitory_lsb = false;
    intact.fc2.disable_inhibitory_lsb = false;
    let (c_intact, _, _) = intact.forward_quantized_full(&input, 5, Some(3), true, 0);
    eq_arr(&c_intact, &[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 5.0, 0.0]);

    let mut broken = net.clone();
    broken.fc1.fixed_quant_scale = 0.1;
    broken.fc2.fixed_quant_scale = 0.1;
    broken.fc1.disable_inhibitory_lsb = true;
    broken.fc2.disable_inhibitory_lsb = true;
    let (c_broken, _, _) = broken.forward_quantized_full(&input, 5, Some(3), true, 0);
    eq_arr(&c_broken, &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);

    // The defect model demonstrably changes the spiking output.
    assert_ne!(flat(&c_intact), flat(&c_broken));
    close32(c_intact.sum(), 15.0, 1e-6);
    close32(c_broken.sum(), 50.0, 1e-6);
}
