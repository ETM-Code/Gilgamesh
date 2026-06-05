//! Characterization tests for src/neurons/lif/* (forward, backward, mode, builder, state).

mod common;
use common::*;

use gilgamesh::neurons::lif::PhysicsParams;
use gilgamesh::neurons::{Leaky, NeuronMode, ResetMechanism};
use gilgamesh::surrogate::SurrogateGradient;
use ndarray::{array, Array2};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const DT: f32 = 0.001;

// ---------------- forward.rs ----------------

#[test]
fn simple_mode_spike_and_reset_subtract() {
    let lif = Leaky::new_simple(2, 0.9);
    let st = lif.init_state(1);
    let input = array![[2.0, 0.5]];
    let (sp1, st1, _) = lif.forward(&input, &st);
    eq_arr(&sp1, &[1.0, 0.0]);
    // mem_new = beta*0 + input = [2.0, 0.5]; neuron0 spikes -> 2.0-1.0=1.0, neuron1 stays.
    close_arr(&st1.mem, &[1.0, 0.5], 1e-6);

    let (sp2, st2, _) = lif.forward(&input, &st1);
    eq_arr(&sp2, &[1.0, 0.0]);
    // mem_new = 0.9*[1.0,0.5] + [2.0,0.5] = [2.9,0.95]; n0 spikes -> 1.9; n1 0.95.
    close_arr(&st2.mem, &[1.9000001, 0.95], 1e-6);
}

#[test]
fn physics_default_no_spike_tiny_rc_step() {
    // CHARACTERIZATION: Leaky::new defaults to Physics (tau_m=0.00396, dt=1e-6),
    // so a single 1us step barely charges the membrane and produces NO spikes.
    // This mirrors the CLI "Test 2" surprising behavior. Locked.
    let lif = Leaky::new(3, 0.9);
    let st = lif.init_state(1);
    let input = array![[2.0, 0.5, 0.3]];
    let (sp, st1, _) = lif.forward(&input, &st);
    eq_arr(&sp, &[0.0, 0.0, 0.0]);
    close_arr(&st1.mem, &[0.00050497055, 0.00012624264, 7.5745585e-5], 1e-9);
}

#[test]
fn forward_with_dt_scaling_golden() {
    let lif = Leaky::new_physics(1, 0.01, 0.001);
    let zs = lif.init_state(1);
    let input = array![[3.0]];
    let (_, m_dt, _) = lif.forward_with_dt(&input, &zs, DT);
    let (_, m_2dt, _) = lif.forward_with_dt(&input, &zs, 2.0 * DT);
    let (_, m_hdt, _) = lif.forward_with_dt(&input, &zs, 0.5 * DT);
    // Larger dt => more charge accumulated from zero state.
    assert!(m_hdt.mem[[0, 0]] < m_dt.mem[[0, 0]]);
    assert!(m_dt.mem[[0, 0]] < m_2dt.mem[[0, 0]]);
    close_arr(&m_dt.mem, &[0.2854877], 1e-6);
    close_arr(&m_2dt.mem, &[0.54380786], 1e-6);
    close_arr(&m_hdt.mem, &[0.14631182], 1e-6);
}

#[test]
fn forward_with_pulse_charge_scale_and_decay() {
    let lif = Leaky::new_physics_with_pulse(1, 0.01, 0.001, 0.00167, 4.42);
    let st = lif.init_state_with_pulse(1);
    let strong = array![[100.0]];
    let (pulse1, st1, _) = lif.forward_with_pulse(&strong, &st, DT);
    close_arr(&pulse1, &[3.3255475], 1e-5);
    // Decaying pulse over subsequent zero-input steps.
    let zero = array![[0.0]];
    let (pulse2, st2, _) = lif.forward_with_pulse(&zero, &st1, DT);
    let (pulse3, _, _) = lif.forward_with_pulse(&zero, &st2, DT);
    close_arr(&pulse2, &[3.3255475], 1e-5);
    close_arr(&pulse3, &[3.3255475], 1e-5);
}

#[test]
fn forward_with_adaptation_threshold_trajectory() {
    let lif = Leaky::new_physics_with_threshold_adaptation(2, 0.01, 0.001, 0.001, 1.0, 1.5);
    let st = lif.init_state_with_adaptation(1);
    let strong = array![[100.0, 100.0]];
    let (sp1, st1, _) = lif.forward_with_adaptation(&strong, &st, DT);
    eq_arr(&sp1, &[1.0, 1.0]);
    // After spike, threshold moves toward theta_high (1.5) from theta_low (1.0).
    close_arr(st1.adaptive_threshold.as_ref().unwrap(), &[1.3160603, 1.3160603], 1e-5);

    // 50 quiet steps -> threshold relaxes back toward theta_low.
    let zero = array![[0.0, 0.0]];
    let mut state = st1;
    let mut spike_total = sp1.sum();
    for _ in 0..50 {
        let (s, ns, _) = lif.forward_with_adaptation(&zero, &state, DT);
        spike_total += s.sum();
        state = ns;
    }
    close_arr(state.adaptive_threshold.as_ref().unwrap(), &[1.0, 1.0], 1e-4);
    assert_eq!(spike_total, 6.0); // locked spike count over the whole run
}

#[test]
fn forward_with_hardware_timing_emitted_sequence() {
    // CHARACTERIZATION: complex, previously-untested edge-triggered + delayed path.
    let hw_mode: NeuronMode = PhysicsParams {
        tau_m: 0.01,
        dt: 0.001,
        comparator_delay_s: 0.0025,
        reset_hold_s: 0.003,
        ..Default::default()
    }
    .into();
    let lif = Leaky::new(1, 0.9).with_mode(hw_mode).with_threshold(1.0);
    let mut st = lif.init_state_with_hardware_timing(1);
    let input = array![[100.0]];
    let mut emitted = Vec::new();
    for _ in 0..10 {
        let (out, ns, _) = lif.forward_with_hardware_timing(&input, &st, DT);
        emitted.push(out[[0, 0]]);
        st = ns;
    }
    // delay_steps=ceil(0.0025/0.001)=3, so the first emission lands at step index 3.
    close_vec(
        &emitted,
        &[0.0, 0.0, 0.0, 0.00666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1e-4,
    );
    assert_eq!(
        st.pending_spike_steps.as_ref().unwrap().iter().cloned().collect::<Vec<u16>>(),
        vec![1u16]
    );
    assert_eq!(
        st.reset_hold_steps.as_ref().unwrap().iter().cloned().collect::<Vec<u16>>(),
        vec![0u16]
    );
}

#[test]
fn forward_full_physics_combined_trajectory() {
    let lif = Leaky::new_physics_with_pulse(1, 0.01, 0.001, 0.00167, 4.42);
    let mut st = lif.init_state_full(1);
    let input = array![[100.0]];
    let mut pulses = Vec::new();
    let mut mem = Vec::new();
    let mut thresh = Vec::new();
    for _ in 0..5 {
        let (p, ns, _) = lif.forward_full_physics(&input, &st, DT);
        pulses.push(p[[0, 0]]);
        mem.push(ns.mem[[0, 0]]);
        thresh.push(ns.adaptive_threshold.as_ref().unwrap()[[0, 0]]);
        st = ns;
    }
    close_vec(&pulses, &[3.3255475, 3.3255475, 3.3255475, 3.3255475, 3.3255475], 1e-4);
    close_vec(&mem, &[4.0, 3.8735757, 3.827067, 3.8099575, 3.8036633], 1e-4);
    close_vec(&thresh, &[1.1264242, 1.172933, 1.1900426, 1.1963369, 1.1986524], 1e-4);
}

#[test]
fn forward_noisy_with_dt_deterministic_under_seed() {
    let lif = Leaky::new_simple(2, 0.9);
    let st = lif.init_state(1);
    let input = array![[1.0, 0.8]];

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
    let (sp, state, _) = lif.forward_noisy_with_dt(&input, &st, 0.02, 0.01, &mut rng, DT);
    eq_arr(&sp, &[0.0, 0.0]);
    close_arr(&state.mem, &[1.0045979, 0.81495833], 1e-5);

    // Same seed -> identical result.
    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(123);
    let (sp2, state2, _) = lif.forward_noisy_with_dt(&input, &st, 0.02, 0.01, &mut rng2, DT);
    eq_arr(&sp2, &[0.0, 0.0]);
    close_arr(&state2.mem, &state.mem.iter().cloned().collect::<Vec<f32>>(), 0.0);
    assert_eq!(
        state.mem.iter().cloned().collect::<Vec<f32>>(),
        state2.mem.iter().cloned().collect::<Vec<f32>>()
    );
}

// ---------------- backward.rs ----------------

#[test]
fn backward_surrogate_scaled_gradients() {
    let lif = Leaky::new_simple(2, 0.9);
    let st = lif.init_state(1);
    let input = array![[2.0, 0.5]];
    let (_, _, cache) = lif.forward(&input, &st);
    let grad_spikes = array![[1.0, 1.0]];
    let grad_mem_next = Array2::<f32>::zeros((1, 2));
    let (gi, gmp) = lif.backward(&grad_spikes, &grad_mem_next, &cache);
    // grad_input = grad_spikes * surrogate(mem_shifted); grad_mem_prev = grad_input * beta.
    close_arr(&gi, &[0.00147929, 0.0054869684], 1e-7);
    close_arr(&gmp, &[0.0013313609, 0.0049382714], 1e-7);
    // grad_mem_prev == grad_input * beta (0.9)
    close32(gmp[[0, 0]], gi[[0, 0]] * 0.9, 1e-9);
}

// ---------------- mode.rs ----------------

#[test]
fn mode_accessors_simple_vs_physics() {
    let simple = NeuronMode::Simple;
    assert_eq!(simple.tau_pulse(), 0.0);
    assert_eq!(simple.v_peak(), 1.0);
    assert_eq!(simple.tau_theta(), 0.0);
    assert_eq!(simple.theta_low(), 1.0);
    assert_eq!(simple.theta_high(), 1.0);
    assert_eq!(simple.v_min(), f32::NEG_INFINITY);
    assert_eq!(simple.v_max(), f32::INFINITY);
    assert!(!simple.has_threshold_adaptation());
    assert!(!simple.has_hardware_timing());
    assert_eq!(simple.comparator_delay(), 0.0);
    assert_eq!(simple.reset_hold(), 0.0);
    assert_eq!(simple.tau_m(), None);
    assert_eq!(simple.dt(), None);
    assert_eq!(simple.effective_beta(0.7), 0.7);

    let phys = NeuronMode::default();
    assert_eq!(phys.tau_pulse(), 1.5e-6);
    assert_eq!(phys.v_peak(), 4.44);
    assert_eq!(phys.theta_low(), 1.0);
    assert_eq!(phys.theta_high(), 1.2);
    assert_eq!(phys.v_min(), 0.0);
    assert_eq!(phys.v_max(), 5.0);
    assert!(phys.has_threshold_adaptation()); // theta_high != theta_low
}

#[test]
fn physics_from_beta_tau_golden() {
    // tau_m = -dt/ln(beta)
    let m = NeuronMode::physics_from_beta(0.9, 0.001);
    close32(m.tau_m().unwrap(), 0.009491219, 1e-6);
    // Invalid beta (>=1) -> fallback dt*10.
    let inv = NeuronMode::physics_from_beta(1.5, 0.001);
    close32(inv.tau_m().unwrap(), 0.010000001, 1e-6);
}

#[test]
fn mode_default_constants() {
    // CHARACTERIZATION: NeuronMode::default()'s comparator_delay=50ns differs
    // from default_comparator_delay()=0. Locked.
    match NeuronMode::default() {
        NeuronMode::Physics {
            tau_m,
            dt,
            tau_pulse,
            v_peak,
            tau_theta,
            theta_low,
            theta_high,
            v_min,
            v_max,
            comparator_delay_s,
            reset_hold_s,
        } => {
            close32(tau_m, 0.00396, 1e-9);
            close32(dt, 1e-6, 1e-12);
            close32(tau_pulse, 1.5e-6, 1e-12);
            close32(v_peak, 4.44, 1e-6);
            close32(tau_theta, 0.001, 1e-9);
            close32(theta_low, 1.0, 1e-9);
            close32(theta_high, 1.2, 1e-9);
            close32(v_min, 0.0, 1e-9);
            close32(v_max, 5.0, 1e-9);
            close32(comparator_delay_s, 50e-9, 1e-12);
            close32(reset_hold_s, 0.24 * 1.5e-6, 1e-12);
        }
        _ => panic!("default must be Physics"),
    }
}

// ---------------- builder.rs ----------------

#[test]
fn builder_beta_clamp_and_setters() {
    assert_eq!(Leaky::new(1, 1.5).beta, 1.0);
    assert_eq!(Leaky::new(1, -0.5).beta, 0.0);

    let lif = Leaky::new_simple(1, 0.5)
        .with_threshold(0.7)
        .with_spike_grad(SurrogateGradient::atan(3.0))
        .with_reset_mechanism(ResetMechanism::Zero);
    assert_eq!(lif.threshold, 0.7);
    assert_eq!(lif.spike_grad.slope(), 3.0);
    assert_eq!(lif.reset_mechanism, ResetMechanism::Zero);

    // with_mode recomputes beta from tau_m/dt.
    let lif2 = Leaky::new_simple(1, 0.5).with_mode(NeuronMode::physics(0.01, 0.001, 1.5e-6, 4.44));
    close32(lif2.beta, (-0.001f32 / 0.01).exp(), 1e-6);

    // get_dt / get_tau_m
    assert_eq!(Leaky::new_simple(1, 0.9).get_dt(), 0.001);
    assert_eq!(Leaky::new_physics(1, 0.01, 0.002).get_dt(), 0.002);
    assert_eq!(Leaky::new_physics(1, 0.01, 0.002).get_tau_m(), 0.01);

    close32(Leaky::new_physics(2, 0.01, 0.001).beta, (-0.001f32 / 0.01).exp(), 1e-6);
}

// ---------------- state.rs ----------------

#[test]
fn state_constructors_option_fields() {
    use gilgamesh::neurons::LeakyState;
    let m = Array2::<f32>::zeros((1, 2));

    let plain = LeakyState::new(m.clone());
    assert!(plain.time_since_spike.is_none());
    assert!(plain.adaptive_threshold.is_none());
    assert!(plain.pending_spike_steps.is_none());
    assert!(plain.reset_hold_steps.is_none());

    let pulse = LeakyState::new_with_pulse_tracking(m.clone());
    assert!(pulse.time_since_spike.as_ref().unwrap().iter().all(|v| v.is_infinite()));
    assert!(pulse.adaptive_threshold.is_none());

    let adapt = LeakyState::new_with_threshold_adaptation(m.clone(), 1.3);
    assert!(adapt.adaptive_threshold.as_ref().unwrap().iter().all(|&v| v == 1.3));

    let full = LeakyState::new_full(m.clone(), 1.1);
    assert!(full.time_since_spike.is_some());
    assert!(full.adaptive_threshold.is_some());
    assert!(full.pending_spike_steps.is_none());

    let hw = LeakyState::new_with_hardware_timing(m.clone());
    assert!(hw.pending_spike_steps.is_some());
    assert!(hw.reset_hold_steps.is_some());
    assert!(hw.adaptive_threshold.is_none());

    let fp = LeakyState::new_full_physics(m.clone(), 1.2);
    assert!(fp.pending_spike_steps.is_some());
    assert!(fp.adaptive_threshold.is_some());
}

#[test]
fn state_reset_quirk_leaves_adaptive_threshold() {
    use gilgamesh::neurons::LeakyState;
    let mut s = LeakyState::new_full_physics(Array2::from_elem((1, 2), 3.0), 1.4);
    s.pending_spike_steps.as_mut().unwrap().fill(5);
    s.reset_hold_steps.as_mut().unwrap().fill(7);
    s.reset();
    // mem zeroed, tss=INF, counters zeroed.
    assert!(s.mem.iter().all(|&v| v == 0.0));
    assert!(s.time_since_spike.as_ref().unwrap().iter().all(|v| v.is_infinite()));
    assert!(s.pending_spike_steps.as_ref().unwrap().iter().all(|&v| v == 0));
    assert!(s.reset_hold_steps.as_ref().unwrap().iter().all(|&v| v == 0));
    // CHARACTERIZATION: adaptive_threshold is intentionally NOT reset by reset().
    assert!(s.adaptive_threshold.as_ref().unwrap().iter().all(|&v| v == 1.4));
    // reset_threshold fills it.
    s.reset_threshold(0.9);
    assert!(s.adaptive_threshold.as_ref().unwrap().iter().all(|&v| v == 0.9));
}
