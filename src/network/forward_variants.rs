use super::{Network, NetworkCache, NetworkState};
use ndarray::Array2;

impl Network {
    /// Run a forward loop over multiple timesteps using a per-step function.
    ///
    /// Common loop structure shared by forward_with_dt, forward_with_adaptation, etc.
    fn run_forward_loop<F>(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        mut state: NetworkState,
        step_fn: F,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>)
    where
        F: Fn(
            &Self,
            &Array2<f32>,
            &NetworkState,
        ) -> (Array2<f32>, Array2<f32>, NetworkState, NetworkCache),
    {
        let batch_size = input.shape()[0];
        let mut caches = Vec::with_capacity(num_steps);
        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            let (output_spikes, output_membrane, new_state, cache) = step_fn(self, input, &state);
            spike_count += &output_spikes;
            final_mem = output_membrane;
            state = new_state;
            caches.push(cache);
        }

        (spike_count, final_mem, caches)
    }

    /// Full forward pass with variable dt (for physics mode fine-grained simulation)
    pub fn forward_with_dt(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let batch_size = input.shape()[0];
        let state = self.init_state(batch_size);
        self.run_forward_loop(input, num_steps, state, |net, inp, st| {
            net.forward_step_with_dt(inp, st, dt)
        })
    }

    /// Forward pass for a single timestep with pulse stretching
    pub fn forward_step_with_pulse(
        &self,
        input: &Array2<f32>,
        state: &NetworkState,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, NetworkState, NetworkCache) {
        let hidden_current = self.fc1.forward(input);
        let (pulse1, lif1_state, lif1_cache) =
            self.lif1
                .forward_with_pulse(&hidden_current, &state.lif1_state, dt);

        let output_current = self.fc2.forward(&pulse1);
        let (pulse2, lif2_state, lif2_cache) =
            self.lif2
                .forward_with_pulse(&output_current, &state.lif2_state, dt);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
            input_accum: None,
        };

        let cache = NetworkCache::new(
            hidden_current,
            lif1_cache.spikes.clone(),
            output_current,
            lif1_cache,
            lif2_cache,
        );

        (pulse2, new_state.lif2_state.mem.clone(), new_state, cache)
    }

    /// Full forward pass with pulse stretching (physics mode)
    pub fn forward_with_pulse(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let batch_size = input.shape()[0];
        let mut state = self.init_state_with_pulse(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            let (_, output_membrane, new_state, cache) =
                self.forward_step_with_pulse(input, &state, dt);
            spike_count += &cache.lif2_cache.spikes;
            final_mem = output_membrane;
            state = new_state;
            caches.push(cache);
        }

        (spike_count, final_mem, caches)
    }

    /// Forward pass with threshold adaptation for a single timestep
    pub fn forward_step_with_adaptation(
        &self,
        input: &Array2<f32>,
        state: &NetworkState,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, NetworkState, NetworkCache) {
        let hidden_current = self.fc1.forward(input);
        let (hidden_spikes, lif1_state, lif1_cache) =
            self.lif1
                .forward_with_adaptation(&hidden_current, &state.lif1_state, dt);

        let output_current = self.fc2.forward(&hidden_spikes);
        let (output_spikes, lif2_state, lif2_cache) =
            self.lif2
                .forward_with_adaptation(&output_current, &state.lif2_state, dt);

        let new_state = NetworkState {
            lif1_state,
            lif2_state,
            input_accum: None,
        };

        let cache = NetworkCache::new(
            hidden_current,
            hidden_spikes,
            output_current,
            lif1_cache,
            lif2_cache,
        );

        (
            output_spikes,
            new_state.lif2_state.mem.clone(),
            new_state,
            cache,
        )
    }

    /// Full forward pass with threshold adaptation
    pub fn forward_with_adaptation(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        dt: f32,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let batch_size = input.shape()[0];
        let state = self.init_state_with_adaptation(batch_size);
        self.run_forward_loop(input, num_steps, state, |net, inp, st| {
            net.forward_step_with_adaptation(inp, st, dt)
        })
    }

    /// Forward pass with analog output mode
    pub fn forward_with_analog(
        &self,
        input: &Array2<f32>,
        num_steps: usize,
        analog_gain: f32,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let batch_size = input.shape()[0];
        let mut state = self.init_state(batch_size);
        let mut caches = Vec::with_capacity(num_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for _ in 0..num_steps {
            let hidden_current = self.fc1.forward(input);
            let (hidden_spikes, lif1_state, lif1_cache) =
                self.lif1.forward(&hidden_current, &state.lif1_state);

            let layer1_output = if analog_gain > 0.0 {
                &hidden_spikes + &(&lif1_state.mem * analog_gain)
            } else {
                hidden_spikes.clone()
            };

            let output_current = self.fc2.forward(&layer1_output);
            let (output_spikes, lif2_state, lif2_cache) =
                self.lif2.forward(&output_current, &state.lif2_state);

            spike_count += &output_spikes;
            final_mem = lif2_state.mem.clone();

            state = NetworkState {
                lif1_state,
                lif2_state,
                input_accum: state.input_accum,
            };
            caches.push(NetworkCache::new(
                hidden_current,
                hidden_spikes,
                output_current,
                lif1_cache,
                lif2_cache,
            ));
        }

        (spike_count, final_mem, caches)
    }

    /// Forward pass with input encoding (rate-coded or temporal)
    pub fn forward_with_encoding(
        &self,
        input: &Array2<f32>,
        encoder: &crate::data::InputEncoder,
        num_steps: usize,
    ) -> (Array2<f32>, Array2<f32>, Vec<NetworkCache>) {
        let actual_steps = encoder.timesteps_needed(num_steps);
        let batch_size = input.shape()[0];
        let mut state = self.init_state(batch_size);
        let mut caches = Vec::with_capacity(actual_steps);

        let mut spike_count = Array2::zeros((batch_size, self.lif2.size));
        let mut final_mem = Array2::zeros((batch_size, self.lif2.size));

        for t in 0..actual_steps {
            let encoded_input = encoder.encode_timestep(input, t);

            let hidden_current = self.fc1.forward(&encoded_input);
            let (hidden_spikes, lif1_state, lif1_cache) =
                self.lif1.forward(&hidden_current, &state.lif1_state);

            let output_current = self.fc2.forward(&hidden_spikes);
            let (output_spikes, lif2_state, lif2_cache) =
                self.lif2.forward(&output_current, &state.lif2_state);

            spike_count += &output_spikes;
            final_mem = lif2_state.mem.clone();

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
            cache.encoded_input = Some(encoded_input);
            caches.push(cache);
        }

        (spike_count, final_mem, caches)
    }
}
