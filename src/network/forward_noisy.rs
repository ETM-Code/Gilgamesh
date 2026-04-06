use ndarray::Array2;
use rand::Rng;

use super::{Network, NetworkCache, NetworkState};

impl Network {
    /// Full forward pass with noise injection (for robustness training)
    pub fn forward_noisy<R: Rng>(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        weight_noise_std: f32,
        threshold_noise_std: f32,
        membrane_noise_std: f32,
        input_noise_std: f32,
        rng: &mut R,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        use rand_distr::{Distribution, Normal};

        let batch_size = input.shape()[0];
        let mut state = self.init_state(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));

        let weight_normal = (weight_noise_std.is_finite() && weight_noise_std > 0.0)
            .then(|| Normal::new(0.0, weight_noise_std as f64).ok())
            .flatten();
        let input_normal = (input_noise_std.is_finite() && input_noise_std > 0.0)
            .then(|| Normal::new(0.0, input_noise_std as f64).ok())
            .flatten();

        let fc1_weight = if let Some(normal) = &weight_normal {
            self.fc1
                .weight
                .mapv(|w| w * (1.0 + normal.sample(rng) as f32))
        } else {
            self.fc1.weight.clone()
        };
        let fc2_weight = if let Some(normal) = &weight_normal {
            self.fc2
                .weight
                .mapv(|w| w * (1.0 + normal.sample(rng) as f32))
        } else {
            self.fc2.weight.clone()
        };
        let spiking_raw_input = if self.spiking_input {
            Some(Self::normalize_for_spiking_input(input))
        } else {
            None
        };

        for t in 0..num_steps {
            let noisy_input = if self.spiking_input {
                let raw = spiking_raw_input.as_ref().unwrap();
                if let Some(normal) = &input_normal {
                    raw.mapv(|x| (x * (1.0 + normal.sample(rng) as f32)).clamp(0.0, 1.0))
                } else {
                    raw.clone()
                }
            } else if let Some(normal) = &input_normal {
                input.mapv(|x| x * (1.0 + normal.sample(rng) as f32))
            } else {
                input.clone()
            };

            let (fc1_input, input_spikes_cache, input_accum_cache) = self.spiking_input_step(
                noisy_input,
                spiking_raw_input.as_ref(),
                &mut state,
                t,
                num_steps,
            );

            let mut hidden_current = fc1_input.dot(&fc1_weight);
            self.fc1
                .apply_synapse_drive_model_inplace(&mut hidden_current);
            if let Some(ref b) = self.fc1.bias {
                for mut row in hidden_current.rows_mut() {
                    row += b;
                }
            }
            if self.dac_max.is_finite() {
                hidden_current.mapv_inplace(|v| v.clamp(0.0, self.dac_max));
            }
            let (hidden_spikes, lif1_state, lif1_cache) = self.lif1.forward_noisy(
                &hidden_current,
                &state.lif1_state,
                threshold_noise_std,
                membrane_noise_std,
                rng,
            );

            let mut output_current = hidden_spikes.dot(&fc2_weight);
            self.fc2
                .apply_synapse_drive_model_inplace(&mut output_current);
            if let Some(ref b) = self.fc2.bias {
                for mut row in output_current.rows_mut() {
                    row += b;
                }
            }

            let (output_spikes, lif2_state, lif2_cache) = if self.should_use_two_phase_pulse() {
                let (t_on, t_off) = self.pulse_phase_durations();
                let (phase1_spikes, state_after_pulse, phase1_cache) =
                    self.lif2.forward_noisy_with_dt(
                        &output_current,
                        &state.lif2_state,
                        threshold_noise_std,
                        membrane_noise_std,
                        rng,
                        t_on,
                    );

                let zero_input = Array2::zeros(output_current.raw_dim());
                let (phase2_spikes, lif2_state, _) = self.lif2.forward_noisy_with_dt(
                    &zero_input,
                    &state_after_pulse,
                    0.0,
                    0.0,
                    rng,
                    t_off,
                );
                let combined =
                    (&phase1_spikes + &phase2_spikes).mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
                (combined, lif2_state, phase1_cache)
            } else {
                self.lif2.forward_noisy(
                    &output_current,
                    &state.lif2_state,
                    threshold_noise_std,
                    membrane_noise_std,
                    rng,
                )
            };

            spike_count += &output_spikes;
            state = NetworkState {
                lif1_state,
                lif2_state,
                input_accum: state.input_accum,
            };
            let mut cache = NetworkCache::new(
                hidden_current,
                hidden_spikes,
                output_current,
                lif1_cache,
                lif2_cache,
            );
            cache.input_spikes = input_spikes_cache;
            cache.input_accum_pre = input_accum_cache;
            caches.push(cache);
        }

        let final_mem = state.lif2_state.mem;
        (spike_count, final_mem, caches)
    }
}
