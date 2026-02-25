use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::cache::LeakyCache;
use super::leaky::Leaky;
use super::mode::{default_tau_pulse, default_tau_theta, NeuronMode, ResetMechanism};
use super::state::LeakyState;

impl Leaky {
    /// Compute membrane update based on mode (uses stored dt)
    #[inline]
    fn compute_membrane(&self, mem_prev: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        self.compute_membrane_with_dt(mem_prev, input, None)
    }

    /// Compute membrane update with optional dt override
    #[inline]
    fn compute_membrane_with_dt(
        &self,
        mem_prev: &Array2<f32>,
        input: &Array2<f32>,
        dt_override: Option<f32>,
    ) -> Array2<f32> {
        match &self.mode {
            NeuronMode::Simple => mem_prev * self.beta + input,
            NeuronMode::Physics {
                tau_m,
                dt,
                v_min,
                v_max,
                ..
            } => {
                let effective_dt = dt_override.unwrap_or(*dt);
                let decay = (-effective_dt / tau_m).exp();
                // RC membrane: V(t+dt) = V_ss + (V(t) - V_ss) * exp(-dt/tau)
                // where V_ss = input (steady-state voltage for constant drive)
                let mem_new = mem_prev * decay + input * (1.0 - decay);
                mem_new.mapv(|v| v.clamp(*v_min, *v_max))
            }
        }
    }

    /// Generate spikes from membrane potential
    #[inline]
    fn generate_spikes(mem_shifted: &Array2<f32>) -> Array2<f32> {
        mem_shifted.mapv(|potential| if potential > 0.0 { 1.0 } else { 0.0 })
    }

    /// Apply reset mechanism after spike with per-neuron threshold
    #[inline]
    fn apply_reset_array(
        &self,
        mem: &Array2<f32>,
        spikes: &Array2<f32>,
        threshold: &Array2<f32>,
    ) -> Array2<f32> {
        match self.reset_mechanism {
            ResetMechanism::Subtract => mem - &(spikes * threshold),
            ResetMechanism::Zero => mem * &(1.0 - spikes),
            ResetMechanism::None => mem.clone(),
        }
    }

    /// Apply reset mechanism after spike with scalar threshold
    #[inline]
    fn apply_reset(&self, mem: &Array2<f32>, spikes: &Array2<f32>, threshold: f32) -> Array2<f32> {
        self.apply_reset_array(mem, spikes, &Array2::from_elem(mem.raw_dim(), threshold))
    }

    /// Update time since spike tracking
    #[inline]
    fn update_time_since_spike(
        prev: Option<&Array2<f32>>,
        spikes: &Array2<f32>,
        dt: f32,
    ) -> Option<Array2<f32>> {
        if let Some(time_since_spike) = prev {
            let mut new_time_since_spike = time_since_spike + dt;
            new_time_since_spike
                .iter_mut()
                .zip(spikes.iter())
                .for_each(|(time, &spike)| {
                    if spike > 0.0 {
                        *time = 0.0;
                    }
                });
            Some(new_time_since_spike)
        } else {
            let mut time_since_spike = Array2::from_elem(spikes.raw_dim(), f32::INFINITY);
            time_since_spike
                .iter_mut()
                .zip(spikes.iter())
                .for_each(|(time, &spike)| {
                    if spike > 0.0 {
                        *time = 0.0;
                    }
                });
            Some(time_since_spike)
        }
    }

    /// Compute pulse output from time since spike
    ///
    /// Returns the average pulse voltage over the current timestep [t, t+dt],
    /// matching the charge that SPICE's continuous RC circuit would integrate.
    ///
    /// For a spike at t=0, the physical pulse is v_peak * exp(-t/tau_pulse).
    /// The average over timestep n is:
    ///   v_avg = (v_peak * tau_pulse / dt) * exp(-n*dt/tau_pulse) * (1 - exp(-dt/tau_pulse))
    #[inline]
    fn compute_pulse_output(
        time_since_spike: &Array2<f32>,
        tau_pulse: f32,
        v_peak: f32,
        dt: f32,
    ) -> Array2<f32> {
        let cutoff = 5.0 * tau_pulse;
        // Exact average of exponential pulse over one timestep
        let charge_scale = (tau_pulse / dt) * (1.0 - (-dt / tau_pulse).exp());
        time_since_spike.mapv(|elapsed| {
            if elapsed < cutoff {
                v_peak * charge_scale * (-elapsed / tau_pulse).exp()
            } else {
                0.0
            }
        })
    }

    /// Update adaptive threshold
    #[inline]
    fn update_adaptive_threshold(
        current: &Array2<f32>,
        spikes: &Array2<f32>,
        dt: f32,
        tau_theta: f32,
        theta_low: f32,
        theta_high: f32,
    ) -> Array2<f32> {
        let decay = (-dt / tau_theta).exp();
        let mut new_thresh = current.clone();
        new_thresh
            .iter_mut()
            .zip(spikes.iter())
            .for_each(|(theta, &spike)| {
                let target = if spike > 0.0 { theta_high } else { theta_low };
                *theta = target + (*theta - target) * decay;
            });
        new_thresh
    }

    /// Forward pass for a single timestep
    pub fn forward(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        let mem_new = self.compute_membrane(&state.mem, input);
        let mem_shifted = &mem_new - self.threshold;
        let spikes = Self::generate_spikes(&mem_shifted);
        let mem_reset = self.apply_reset(&mem_new, &spikes, self.threshold);

        let cache = LeakyCache {
            mem_shifted: mem_shifted.clone(),
            spikes: spikes.clone(),
        };

        (spikes, LeakyState::new(mem_reset), cache)
    }

    /// Forward pass with variable timestep
    pub fn forward_with_dt(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        let mem_new = self.compute_membrane_with_dt(&state.mem, input, Some(dt));
        let mem_shifted = &mem_new - self.threshold;
        let spikes = Self::generate_spikes(&mem_shifted);
        let mem_reset = self.apply_reset(&mem_new, &spikes, self.threshold);

        let cache = LeakyCache {
            mem_shifted: mem_shifted.clone(),
            spikes: spikes.clone(),
        };

        (spikes, LeakyState::new(mem_reset), cache)
    }

    /// Forward pass with noise injection (for robustness training)
    pub fn forward_noisy<R: Rng>(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        threshold_noise_std: f32,
        membrane_noise_std: f32,
        rng: &mut R,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        let mut mem_new = self.compute_membrane(&state.mem, input);

        if membrane_noise_std.is_finite() && membrane_noise_std > 0.0 {
            if let Ok(normal) = Normal::new(0.0, membrane_noise_std as f64) {
                for v in mem_new.iter_mut() {
                    *v += normal.sample(rng) as f32;
                }
            }
        }

        let mem_shifted_for_grad = &mem_new - self.threshold;

        let mut effective_threshold = if threshold_noise_std.is_finite()
            && threshold_noise_std > 0.0
            && self.threshold.is_finite()
            && self.threshold != 0.0
        {
            let std = (self.threshold * threshold_noise_std).abs();
            if let Ok(normal) = Normal::new(0.0, std as f64) {
                self.threshold + normal.sample(rng) as f32
            } else {
                self.threshold
            }
        } else {
            self.threshold
        };
        if self.threshold > 0.0 {
            effective_threshold = effective_threshold.max(1e-6);
        }

        let mem_shifted = &mem_new - effective_threshold;
        let spikes = Self::generate_spikes(&mem_shifted);
        let mem_reset = self.apply_reset(&mem_new, &spikes, effective_threshold);

        let cache = LeakyCache {
            mem_shifted: mem_shifted_for_grad,
            spikes: spikes.clone(),
        };

        (spikes, LeakyState::new(mem_reset), cache)
    }

    /// Forward pass with pulse stretching (physics mode)
    pub fn forward_with_pulse(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        let (tau_pulse, v_peak) = match &self.mode {
            NeuronMode::Physics {
                tau_pulse, v_peak, ..
            } => (*tau_pulse, *v_peak),
            NeuronMode::Simple => (default_tau_pulse(), 1.0),
        };

        let mem_new = self.compute_membrane_with_dt(&state.mem, input, Some(dt));
        let mem_shifted = &mem_new - self.threshold;
        let spikes = Self::generate_spikes(&mem_shifted);
        let mem_reset = self.apply_reset(&mem_new, &spikes, self.threshold);

        let new_time_since_spike =
            Self::update_time_since_spike(state.time_since_spike.as_ref(), &spikes, dt);

        let charge_scale = (tau_pulse / dt) * (1.0 - (-dt / tau_pulse).exp());
        let pulse_output = if let Some(ref time_since_spike) = new_time_since_spike {
            Self::compute_pulse_output(time_since_spike, tau_pulse, v_peak, dt)
        } else {
            &spikes * (v_peak * charge_scale)
        };

        let cache = LeakyCache {
            mem_shifted: mem_shifted.clone(),
            spikes: spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_reset,
            time_since_spike: new_time_since_spike,
            adaptive_threshold: None,
            pending_spike_steps: None,
            reset_hold_steps: None,
        };

        (pulse_output, new_state, cache)
    }

    /// Forward pass with threshold adaptation (physics mode)
    pub fn forward_with_adaptation(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        let (tau_theta, theta_low, theta_high) = match &self.mode {
            NeuronMode::Physics {
                tau_theta,
                theta_low,
                theta_high,
                ..
            } => (*tau_theta, *theta_low, *theta_high),
            NeuronMode::Simple => (default_tau_theta(), 1.0, 1.0),
        };

        let mem_new = self.compute_membrane_with_dt(&state.mem, input, Some(dt));

        let current_threshold = state
            .adaptive_threshold
            .clone()
            .unwrap_or_else(|| Array2::from_elem(mem_new.raw_dim(), self.threshold));

        let mem_shifted = &mem_new - &current_threshold;
        let spikes = Self::generate_spikes(&mem_shifted);
        let mem_reset = self.apply_reset_array(&mem_new, &spikes, &current_threshold);

        let new_threshold = if state.adaptive_threshold.is_some() {
            Some(Self::update_adaptive_threshold(
                &current_threshold,
                &spikes,
                dt,
                tau_theta,
                theta_low,
                theta_high,
            ))
        } else {
            None
        };

        let cache = LeakyCache {
            mem_shifted: &mem_reset + &current_threshold - self.threshold,
            spikes: spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_reset,
            time_since_spike: None,
            adaptive_threshold: new_threshold,
            pending_spike_steps: None,
            reset_hold_steps: None,
        };

        (spikes, new_state, cache)
    }

    /// Forward pass with full physics features (pulse + adaptation)
    pub fn forward_full_physics(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        let (tau_pulse, v_peak, tau_theta, theta_low, theta_high) = match &self.mode {
            NeuronMode::Physics {
                tau_pulse,
                v_peak,
                tau_theta,
                theta_low,
                theta_high,
                ..
            } => (*tau_pulse, *v_peak, *tau_theta, *theta_low, *theta_high),
            NeuronMode::Simple => (default_tau_pulse(), 1.0, default_tau_theta(), 1.0, 1.0),
        };

        let mem_new = self.compute_membrane_with_dt(&state.mem, input, Some(dt));

        let current_threshold = state
            .adaptive_threshold
            .clone()
            .unwrap_or_else(|| Array2::from_elem(mem_new.raw_dim(), self.threshold));

        let mem_shifted = &mem_new - &current_threshold;
        let spikes = Self::generate_spikes(&mem_shifted);
        let mem_reset = self.apply_reset_array(&mem_new, &spikes, &current_threshold);

        let new_time_since_spike =
            Self::update_time_since_spike(state.time_since_spike.as_ref(), &spikes, dt);

        let new_threshold = Some(Self::update_adaptive_threshold(
            &current_threshold,
            &spikes,
            dt,
            tau_theta,
            theta_low,
            theta_high,
        ));

        let charge_scale = (tau_pulse / dt) * (1.0 - (-dt / tau_pulse).exp());
        let pulse_output = if let Some(ref time_since_spike) = new_time_since_spike {
            Self::compute_pulse_output(time_since_spike, tau_pulse, v_peak, dt)
        } else {
            &spikes * (v_peak * charge_scale)
        };

        let cache = LeakyCache {
            mem_shifted: &mem_reset + &current_threshold - self.threshold,
            spikes: spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_reset,
            time_since_spike: new_time_since_spike,
            adaptive_threshold: new_threshold,
            pending_spike_steps: None,
            reset_hold_steps: None,
        };

        (pulse_output, new_state, cache)
    }

    /// Forward pass with hardware timing simulation
    pub fn forward_with_hardware_timing(
        &self,
        input: &Array2<f32>,
        state: &LeakyState,
        dt: f32,
    ) -> (Array2<f32>, LeakyState, LeakyCache) {
        use ndarray::Zip;

        let shape = state.mem.raw_dim();
        let comparator_delay = self.mode.comparator_delay();
        let reset_hold = self.mode.reset_hold();
        let (tau_theta, theta_low, theta_high) = match &self.mode {
            NeuronMode::Physics {
                tau_theta,
                theta_low,
                theta_high,
                ..
            } => (*tau_theta, *theta_low, *theta_high),
            NeuronMode::Simple => (default_tau_theta(), 1.0, 1.0),
        };

        let delay_steps = if comparator_delay > 0.0 {
            (comparator_delay / dt).ceil() as u16
        } else {
            0
        };
        let hold_steps = if reset_hold > 0.0 {
            (reset_hold / dt).ceil() as u16
        } else {
            0
        };

        let mut pending = state
            .pending_spike_steps
            .clone()
            .unwrap_or_else(|| Array2::zeros(shape));
        let mut hold = state
            .reset_hold_steps
            .clone()
            .unwrap_or_else(|| Array2::zeros(shape));

        let mut emitted_spikes = Array2::zeros(shape);

        Zip::from(&mut pending)
            .and(&mut emitted_spikes)
            .for_each(|pending_step, emitted| {
                if *pending_step > 0 {
                    *pending_step -= 1;
                    if *pending_step == 0 {
                        *emitted = 1.0;
                    }
                }
            });

        let mut mem_new = state.mem.clone();
        let mem_integrated = self.compute_membrane_with_dt(&state.mem, input, Some(dt));

        let v_min = self.mode.v_min();

        Zip::from(&mut mem_new)
            .and(&mem_integrated)
            .and(&hold)
            .for_each(|membrane, &integrated, &hold_remaining| {
                if hold_remaining == 0 {
                    *membrane = integrated;
                } else {
                    *membrane = v_min;
                }
            });

        let current_threshold = state
            .adaptive_threshold
            .clone()
            .unwrap_or_else(|| Array2::from_elem(shape, self.threshold));
        let mem_shifted = &mem_new - &current_threshold;
        let mut new_crossings = Array2::zeros(shape);
        Zip::from(&mut new_crossings)
            .and(&mem_shifted)
            .and(&pending)
            .and(&hold)
            .and(&emitted_spikes)
            .for_each(
                |crossing, &shifted, &pending_step, &hold_remaining, &just_emitted| {
                    if shifted > 0.0
                        && pending_step == 0
                        && hold_remaining == 0
                        && just_emitted == 0.0
                    {
                        *crossing = 1.0;
                    }
                },
            );

        if delay_steps > 0 {
            Zip::from(&mut pending)
                .and(&new_crossings)
                .for_each(|pending_step, &crossing| {
                    if crossing > 0.0 {
                        *pending_step = delay_steps;
                    }
                });
        } else {
            emitted_spikes = &emitted_spikes + &new_crossings;
        }

        if hold_steps > 0 {
            Zip::from(&mut hold)
                .and(&emitted_spikes)
                .for_each(|hold_remaining, &spike| {
                    if spike > 0.0 {
                        *hold_remaining = hold_steps;
                    }
                });
        }

        hold.mapv_inplace(|hold_remaining| {
            if hold_remaining > 0 {
                hold_remaining - 1
            } else {
                0
            }
        });

        let new_time_since_spike =
            Self::update_time_since_spike(state.time_since_spike.as_ref(), &emitted_spikes, dt);

        let pulse_output = if let Some(ref time_since_spike) = new_time_since_spike {
            let tau_pulse = self.mode.tau_pulse();
            let v_peak = self.mode.v_peak();
            if tau_pulse > 0.0 {
                Self::compute_pulse_output(time_since_spike, tau_pulse, v_peak, dt)
            } else {
                &emitted_spikes * v_peak
            }
        } else {
            emitted_spikes.clone()
        };

        let new_threshold = if state.adaptive_threshold.is_some() {
            let pulse_scale = if comparator_delay > 0.0 {
                (comparator_delay / dt).min(1.0)
            } else {
                1.0
            };
            let target = Array2::from_elem(shape, theta_low)
                + (&emitted_spikes * pulse_scale) * (theta_high - theta_low);
            let decay = (-dt / tau_theta).exp();
            Some(&target * (1.0 - decay) + &current_threshold * decay)
        } else {
            None
        };

        let cache = LeakyCache {
            mem_shifted: &mem_new + &current_threshold - self.threshold,
            spikes: emitted_spikes.clone(),
        };

        let new_state = LeakyState {
            mem: mem_new,
            time_since_spike: new_time_since_spike,
            adaptive_threshold: new_threshold,
            pending_spike_steps: Some(pending),
            reset_hold_steps: Some(hold),
        };

        (pulse_output, new_state, cache)
    }
}
