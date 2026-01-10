#[cfg(test)]
mod tests {
    use super::super::leaky::Leaky;
    use super::super::mode::{default_tau_pulse, default_v_peak, NeuronMode};
    use super::super::ResetMechanism;
    use ndarray::Array2;
    use ndarray::array;

    #[test]
    fn test_leaky_forward_no_spike() {
        let lif = Leaky::new(3, 0.9);
        let state = lif.init_state(1);
        let input = array![[0.1, 0.2, 0.3]];
        let (spikes, new_state, _) = lif.forward(&input, &state);
        assert!(spikes.iter().all(|&s| s == 0.0));
        assert!(new_state.mem[[0, 0]] > 0.0);
    }

    #[test]
    fn test_leaky_forward_with_spike() {
        let lif = Leaky::new(2, 0.9);
        let state = lif.init_state(1);
        let input = array![[2.0, 0.5]];
        let (spikes, _new_state, _) = lif.forward(&input, &state);
        assert_eq!(spikes[[0, 0]], 1.0);
        assert_eq!(spikes[[0, 1]], 0.0);
    }

    #[test]
    fn test_leaky_reset_subtract() {
        let lif = Leaky::new_simple(1, 0.9).with_reset_mechanism(ResetMechanism::Subtract);
        let mut state = lif.init_state(1);
        let input = array![[0.6]];
        let (_, new_state, _) = lif.forward(&input, &state);
        state = new_state;
        let (spikes, new_state, _) = lif.forward(&input, &state);
        assert_eq!(spikes[[0, 0]], 1.0);
        assert!((new_state.mem[[0, 0]] - 0.14).abs() < 0.01);
    }

    #[test]
    fn test_leaky_backward() {
        let lif = Leaky::new(2, 0.9);
        let state = lif.init_state(1);
        let input = array![[0.5, 1.5]];
        let (_, _, cache) = lif.forward(&input, &state);
        let grad_spikes = array![[1.0, 1.0]];
        let grad_mem_next = array![[0.0, 0.0]];
        let (grad_input, grad_mem_prev) = lif.backward(&grad_spikes, &grad_mem_next, &cache);
        assert_eq!(grad_input.shape(), input.shape());
        assert_eq!(grad_mem_prev.shape(), state.mem.shape());
        assert!(grad_input.iter().all(|&g| g.is_finite()));
    }

    #[test]
    fn test_beta_clamping() {
        let lif = Leaky::new(1, 1.5);
        assert_eq!(lif.beta, 1.0);
        let lif = Leaky::new(1, -0.5);
        assert_eq!(lif.beta, 0.0);
    }

    #[test]
    fn test_neuron_mode_simple() {
        let lif = Leaky::new(2, 0.9);
        assert!(lif.is_physics_mode());
        let lif_simple = Leaky::new_simple(2, 0.9);
        assert!(!lif_simple.is_physics_mode());
        assert!(matches!(lif_simple.mode, NeuronMode::Simple));
    }

    #[test]
    fn test_neuron_mode_physics() {
        let lif = Leaky::new_physics(2, 0.01, 0.001);
        assert!(lif.is_physics_mode());
        let expected_beta = (-0.001f32 / 0.01).exp();
        assert!((lif.beta - expected_beta).abs() < 1e-5);
    }

    #[test]
    fn test_physics_mode_from_beta() {
        let mode = NeuronMode::physics_from_beta(0.9, 0.001);
        if let NeuronMode::Physics { tau_m, dt, .. } = mode {
            let expected_tau = -0.001f32 / 0.9f32.ln();
            assert!((tau_m - expected_tau).abs() < 1e-5);
            assert_eq!(dt, 0.001);
        } else {
            panic!("Expected Physics mode");
        }
    }

    #[test]
    fn test_physics_mode_dynamics() {
        let beta = 0.9f32;
        let dt = 0.001f32;
        let tau_m = -dt / beta.ln();
        let lif_simple = Leaky::new(1, beta);
        let lif_physics = Leaky::new_physics(1, tau_m, dt);
        let state = lif_simple.init_state(1);
        let input = array![[0.5]];
        let (_, state_simple, _) = lif_simple.forward(&input, &state);
        let (_, state_physics, _) = lif_physics.forward(&input, &state);
        let diff = (state_simple.mem[[0, 0]] - state_physics.mem[[0, 0]]).abs();
        assert!(diff < 1e-3, "Membrane diff too large: {}", diff);
    }

    #[test]
    fn test_with_mode_builder() {
        let lif = Leaky::new(2, 0.9)
            .with_mode(NeuronMode::physics(0.01, 0.001, default_tau_pulse(), default_v_peak()));
        assert!(lif.is_physics_mode());
        let expected_beta = (-0.001f32 / 0.01).exp();
        assert!((lif.beta - expected_beta).abs() < 1e-5);
    }

    #[test]
    fn test_forward_with_dt() {
        let tau_m = 0.01f32;
        let dt_stored = 0.001f32;
        let lif = Leaky::new_physics(1, tau_m, dt_stored);
        let state = lif.init_state(1);
        let input = array![[0.5]];
        let (_, state1, _) = lif.forward(&input, &state);
        let (_, state2, _) = lif.forward_with_dt(&input, &state, dt_stored * 2.0);
        let (_, state3, _) = lif.forward_with_dt(&input, &state, dt_stored * 0.5);
        let (_, state1b, _) = lif.forward(&input, &state1);
        let (_, state2b, _) = lif.forward_with_dt(&input, &state2, dt_stored * 2.0);
        let (_, state3b, _) = lif.forward_with_dt(&input, &state3, dt_stored * 0.5);
        assert!(state2b.mem[[0, 0]] < state1b.mem[[0, 0]]);
        assert!(state1b.mem[[0, 0]] < state3b.mem[[0, 0]]);
    }

    #[test]
    fn test_forward_with_pulse() {
        let tau_m = 0.01f32;
        let dt = 0.001f32;
        let tau_pulse = 0.00167f32;
        let v_peak = 4.42f32;
        let lif = Leaky::new_physics_with_pulse(1, tau_m, dt, tau_pulse, v_peak);
        let mut state = lif.init_state_with_pulse(1);
        let input = array![[1.5]];
        let (pulse1, new_state, cache1) = lif.forward_with_pulse(&input, &state, dt);
        state = new_state;
        assert_eq!(cache1.spikes[[0, 0]], 1.0);
        assert!((pulse1[[0, 0]] - v_peak).abs() < 0.01);
        let zero_input = array![[0.0]];
        let (pulse2, new_state, cache2) = lif.forward_with_pulse(&zero_input, &state, dt);
        state = new_state;
        assert_eq!(cache2.spikes[[0, 0]], 0.0);
        let expected_decay = v_peak * (-dt / tau_pulse).exp();
        assert!((pulse2[[0, 0]] - expected_decay).abs() < 0.1);
        let (pulse3, _, _) = lif.forward_with_pulse(&zero_input, &state, dt);
        assert!(pulse3[[0, 0]] < pulse2[[0, 0]]);
    }

    #[test]
    fn test_get_dt_and_tau() {
        let lif_simple = Leaky::new_simple(2, 0.9);
        assert!((lif_simple.get_dt() - 0.001).abs() < 1e-6);
        let tau_expected = -0.001f32 / 0.9f32.ln();
        assert!((lif_simple.get_tau_m() - tau_expected).abs() < 1e-5);
        let lif_physics = Leaky::new_physics(2, 0.005, 0.0001);
        assert!((lif_physics.get_dt() - 0.0001).abs() < 1e-8);
        assert!((lif_physics.get_tau_m() - 0.005).abs() < 1e-6);
    }

    #[test]
    fn test_forward_with_adaptation() {
        let tau_m = 0.00949;
        let dt = 0.001;
        let tau_theta = 0.001;
        let theta_low = 1.0;
        let theta_high = 1.5;
        let lif = Leaky::new_physics_with_threshold_adaptation(
            2, tau_m, dt, tau_theta, theta_low, theta_high,
        );
        let mut state = lif.init_state_with_adaptation(1);
        let initial_thresh = state.adaptive_threshold.as_ref().unwrap()[[0, 0]];
        assert!((initial_thresh - theta_low).abs() < 1e-6);
        let strong_input = Array2::from_elem((1, 2), 3.0);
        let (spikes, new_state, _) = lif.forward_with_adaptation(&strong_input, &state, dt);
        assert!(spikes[[0, 0]] > 0.0);
        let thresh_after_spike = new_state.adaptive_threshold.as_ref().unwrap()[[0, 0]];
        assert!(thresh_after_spike > theta_low);
        let decay = (-dt / tau_theta).exp();
        let expected_after_spike = theta_high + (theta_low - theta_high) * decay;
        assert!((thresh_after_spike - expected_after_spike).abs() < 0.01);
        state = new_state;
        let zero_input = Array2::zeros((1, 2));
        let mut no_spike_count = 0;
        for _ in 0..50 {
            let (spikes, next_state, _) = lif.forward_with_adaptation(&zero_input, &state, dt);
            if spikes[[0, 0]] == 0.0 {
                no_spike_count += 1;
            }
            state = next_state;
        }
        assert!(no_spike_count > 40);
        let thresh_after_decay = state.adaptive_threshold.as_ref().unwrap()[[0, 0]];
        assert!((thresh_after_decay - theta_low).abs() < 0.05);
    }

    #[test]
    fn test_physics_voltage_clamping() {
        let tau_m = 0.00949;
        let dt = 0.001;
        let lif = Leaky::new_physics(1, tau_m, dt);
        assert_eq!(lif.mode.v_min(), 0.0);
        assert_eq!(lif.mode.v_max(), 5.0);
        let mut state = lif.init_state(1);
        let huge_input = array![[100.0]];
        let (_, new_state, _) = lif.forward(&huge_input, &state);
        state = new_state;
        assert!(state.mem[[0, 0]] <= 5.0);
        state = lif.init_state(1);
        let spike_input = array![[2.0]];
        let (spikes, new_state, _) = lif.forward(&spike_input, &state);
        state = new_state;
        assert_eq!(spikes[[0, 0]], 1.0);
        let negative_input = array![[-100.0]];
        let (_, new_state, _) = lif.forward(&negative_input, &state);
        assert!(new_state.mem[[0, 0]] >= 0.0);
    }

    #[test]
    fn test_simple_mode_no_clamping() {
        let lif = Leaky::new_simple(1, 0.9);
        assert!(lif.mode.v_min().is_infinite());
        assert!(lif.mode.v_max().is_infinite());
        let state = lif.init_state(1);
        let huge_input = array![[100.0]];
        let (_, new_state, _) = lif.forward(&huge_input, &state);
        assert!(new_state.mem[[0, 0]] > 5.0);
    }
}

