//! Characterization tests for Leaky::forward_with_hardware_timing covering the
//! reset_hold trajectory and adaptive-threshold interaction (forward.rs:464-638),
//! which lif_golden.rs only exercises with the comparator-delay path and no adaptation.

mod common;
use common::*;

use gilgamesh::neurons::lif::{LeakyState, PhysicsParams};
use gilgamesh::neurons::{Leaky, NeuronMode};
use ndarray::{array, Array2};

const DT: f32 = 0.001;

#[test]
fn hardware_timing_reset_hold_and_adaptive_threshold_trajectory() {
    // Hardware-timing neuron WITH adaptive threshold, no comparator delay,
    // reset_hold = 3ms (=> 3 hold steps), sustained strong input.
    let mode: NeuronMode = PhysicsParams {
        tau_m: 0.01,
        dt: 0.001,
        comparator_delay_s: 0.0,
        reset_hold_s: 0.003,
        theta_low: 1.0,
        theta_high: 1.2,
        tau_theta: 0.001,
        ..Default::default()
    }
    .into();
    let lif = Leaky::new(1, 0.9).with_mode(mode).with_threshold(1.0);

    // full-physics state: pulse tracking + adaptive threshold + hardware timing.
    let mut st = LeakyState::new_full_physics(Array2::<f32>::zeros((1, 1)), 1.0);
    let input = array![[100.0]];

    let mut emitted = Vec::new();
    let mut mem = Vec::new();
    let mut thr = Vec::new();
    for _ in 0..20 {
        let (out, ns, _) = lif.forward_with_hardware_timing(&input, &st, DT);
        emitted.push(out[[0, 0]]);
        mem.push(ns.mem[[0, 0]]);
        thr.push(ns.adaptive_threshold.as_ref().unwrap()[[0, 0]]);
        st = ns;
    }

    // Emitted pulse train: fires, then the reset_hold gap suppresses re-firing for
    // 3 steps (membrane pinned to v_min), then fires again -> period of 4 steps.
    close_vec(
        &emitted,
        &[
            0.00666, 0.0, 0.0, 0.0, 0.00666, 0.0, 0.0, 0.0, 0.00666, 0.0, 0.0, 0.0, 0.00666, 0.0,
            0.0, 0.0, 0.00666, 0.0, 0.0, 0.0,
        ],
        1e-4,
    );

    // Membrane: charges to v_max (5.0) on the firing step, then pinned to v_min (0.0)
    // during the 3-step hold.
    close_vec(
        &mem,
        &[
            5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0,
            0.0, 0.0, 0.0,
        ],
        1e-4,
    );

    // Adaptive threshold trajectory (pulse_scale=1.0 since comparator_delay=0).
    close_vec(
        &thr,
        &[
            1.1264241, 1.0465088, 1.0171096, 1.0062943, 1.1287396, 1.0473607, 1.017423, 1.0064096,
            1.1287822, 1.0473763, 1.0174288, 1.0064117, 1.1287829, 1.0473766, 1.0174289, 1.0064118,
            1.1287829, 1.0473766, 1.0174289, 1.0064118,
        ],
        1e-4,
    );

    // After the last firing the hold counter has run back down to 0.
    assert_eq!(
        st.reset_hold_steps
            .as_ref()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<u16>>(),
        vec![0u16]
    );
}

#[test]
fn hardware_timing_edge_triggers_once_per_crossing() {
    // No comparator delay, NO reset hold, fixed threshold (theta_low==theta_high),
    // sustained above-threshold input. The membrane crosses threshold on step 0,
    // fires, and (because reset subtract leaves it above threshold but the
    // edge-trigger requires prev_shifted <= 0) does NOT keep firing every step.
    let mode: NeuronMode = PhysicsParams {
        tau_m: 0.01,
        dt: 0.001,
        comparator_delay_s: 0.0,
        reset_hold_s: 0.0,
        theta_low: 1.0,
        theta_high: 1.0,
        tau_theta: 0.001,
        ..Default::default()
    }
    .into();
    let lif = Leaky::new(1, 0.9).with_mode(mode).with_threshold(1.0);
    let mut st = LeakyState::new_with_hardware_timing(Array2::<f32>::zeros((1, 1)));
    let input = array![[100.0]];

    let mut emitted = Vec::new();
    for _ in 0..6 {
        let (out, ns, _) = lif.forward_with_hardware_timing(&input, &st, DT);
        emitted.push(out[[0, 0]]);
        st = ns;
    }
    // Fires exactly once on the initial upward crossing (edge behavior),
    // then stays high without re-firing.
    close_vec(&emitted, &[0.00666, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-4);
}
