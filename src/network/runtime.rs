use crate::math_functions::lif::relax_towards;
use crate::network::compiled::{CompiledNetwork, ReadoutRuntimeType, StimulusChannel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationOptions {
    pub t_end: f64,
    pub dt: f64,
    pub record_readout: bool,
    pub return_average: bool,
    #[serde(default)]
    pub stimuli_override: Option<Vec<StimulusChannel>>,
    #[serde(default)]
    pub trace_neurons: Vec<usize>,
    /// Track spike proximity (distance from threshold) for surrogate gradient training.
    #[serde(default)]
    pub return_spike_proximity: bool,
    /// Return eligibility traces (exponentially weighted membrane voltages) for training.
    /// The tau_s value is taken from the first readout or defaults to dt * 10.
    #[serde(default)]
    pub return_eligibility_traces: bool,
}

impl Default for SimulationOptions {
    fn default() -> Self {
        Self {
            t_end: 0.1,
            dt: 1e-4,
            record_readout: true,
            return_average: true,
            stimuli_override: None,
            trace_neurons: Vec::new(),
            return_spike_proximity: false,
            return_eligibility_traces: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SimulationResult {
    pub times: Vec<f64>,
    pub readouts: HashMap<String, Vec<Vec<f64>>>,
    pub average_activity: Option<Vec<f64>>,
    pub spikes: Vec<SpikeEvent>,
    pub final_state: NetworkState,
    pub final_readouts: HashMap<String, Vec<f64>>,
    pub neuron_traces: HashMap<usize, Vec<f64>>,
    /// Average normalized distance from threshold: mean((u - theta) / theta) per neuron.
    /// Useful for surrogate gradient computation in backprop.
    pub spike_proximity: Option<Vec<f64>>,
    /// Eligibility traces: exponentially weighted membrane voltages aligned with readout tau_s.
    /// Used for proper gradient computation in training.
    pub eligibility_traces: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NetworkState {
    pub time: f64,
    pub u: Vec<f64>,
    pub theta: Vec<f64>,
    /// Time at which each neuron's stretched output pulse ends (for spike-based transmission).
    /// Neurons with active pulses (pulse_end_time > current_time) transmit signal to downstream.
    #[serde(default)]
    pub pulse_end_time: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpikeEvent {
    pub time: f64,
    pub neuron: usize,
}

struct StimulusRuntime {
    channel: StimulusChannel,
    tooling: usize,
}

impl StimulusRuntime {
    fn new(channel: &StimulusChannel) -> Self {
        Self {
            channel: channel.clone(),
            tooling: 0,
        }
    }

    #[inline]
    fn sample(&mut self, t: f64) -> f64 {
        let times = &self.channel.times;
        let values = &self.channel.values;
        if times.is_empty() {
            return 0.0;
        }
        while self.tooling + 1 < times.len() && t >= times[self.tooling + 1] {
            self.tooling += 1;
        }
        let idx = self.tooling;
        let t0 = times[idx];
        let v0 = values[idx];
        if idx + 1 >= times.len() {
            return v0;
        }
        let t1 = times[idx + 1];
        let v1 = values[idx + 1];
        if (t1 - t0).abs() < f64::EPSILON {
            return v1;
        }
        let frac = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
        v0 + (v1 - v0) * frac
    }
}

pub fn simulate_network(network: &CompiledNetwork, opts: &SimulationOptions) -> SimulationResult {
    let neuron_count = network.neuron_count();

    let mut state = NetworkState {
        time: network.state0.t,
        u: network.state0.u.clone(),
        theta: network.state0.theta.clone(),
        pulse_end_time: vec![0.0; neuron_count],
    };
    let alpha_out = network.globals.get("alpha_out").copied().unwrap_or(1.0);
    let dt = opts.dt.max(1e-12);
    let steps = (opts.t_end / dt).ceil() as usize;

    let mut readouts: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    let mut readout_state: HashMap<String, Vec<f64>> = HashMap::new();
    let mut neuron_traces: HashMap<usize, Vec<f64>> = HashMap::new();
    if opts.record_readout {
        for readout in &network.readouts {
            readouts.insert(readout.id.clone(), Vec::with_capacity(steps + 1));
            readout_state.insert(readout.id.clone(), vec![0.0; readout.indices.len()]);
        }
    }

    if !opts.trace_neurons.is_empty() {
        for &idx in &opts.trace_neurons {
            neuron_traces.insert(idx, Vec::with_capacity(steps + 1));
        }
    }

    let mut times = Vec::with_capacity(steps + 1);
    let mut average_activity = if opts.return_average {
        Some(vec![0.0; neuron_count])
    } else {
        None
    };
    let mut spike_proximity = if opts.return_spike_proximity {
        Some(vec![0.0; neuron_count])
    } else {
        None
    };
    // Eligibility traces: exponentially weighted membrane voltages
    // Use tau_s from first readout, or default to 10*dt
    let eligibility_tau = network
        .readouts
        .first()
        .and_then(|r| r.tau_s)
        .unwrap_or(dt * 10.0);
    let mut eligibility_traces = if opts.return_eligibility_traces {
        Some(vec![0.0; neuron_count])
    } else {
        None
    };
    let mut spikes = Vec::new();
    let mut external_drive = vec![0.0; neuron_count];
    // Track which neurons spiked for spike-based transmission
    // We use "last step" spikes for transmission (one timestep delay, physically reasonable)
    let mut spiked_last_step = vec![false; neuron_count];
    let mut spiked_this_step = vec![false; neuron_count];
    let stimuli_sources = opts
        .stimuli_override
        .clone()
        .unwrap_or_else(|| network.stimuli.clone());
    let mut stimuli: Vec<StimulusRuntime> =
        stimuli_sources.iter().map(StimulusRuntime::new).collect();

    // precompute exp decay factors
    if opts.record_readout {
        record_readouts(
            alpha_out,
            &state,
            &network.readouts,
            &mut readouts,
            &mut readout_state,
            0.0,
        );
        times.push(state.time);
    }

    if !neuron_traces.is_empty() {
        for (&idx, trace) in neuron_traces.iter_mut() {
            let value = state.u.get(idx).copied().unwrap_or(0.0);
            trace.push(value);
        }
    }

    for step in 0..steps {
        let next_time = state.time + dt;
        external_drive.fill(0.0);
        // Copy this step's spikes to last step, then reset for new detection
        std::mem::swap(&mut spiked_last_step, &mut spiked_this_step);
        spiked_this_step.fill(false);
        for stim in &mut stimuli {
            let amp = stim.sample(next_time);
            for &idx in &stim.channel.target_indices {
                if idx < neuron_count {
                    external_drive[idx] += amp;
                }
            }
        }

        for neuron in 0..neuron_count {
            let start = network.incoming.row_ptr[neuron];
            let end = network.incoming.row_ptr[neuron + 1];
            let mut total = external_drive[neuron];
            for idx in start..end {
                let pre = network.incoming.src[idx];
                let weight = network.incoming.g[idx];
                // For spiking neurons (low threshold), use spike pulse indicator instead of voltage
                // This fixes the issue where voltage resets to 0 on spike, losing information
                let pre_theta = network.neuron.theta0[pre];
                let pre_signal = if pre_theta < 1.0 {
                    // Spiking neuron: check if stretched output pulse is still active
                    // This models the monostable one-shot pulse stretcher from hardware
                    if state.pulse_end_time[pre] > state.time {
                        1.0 // Pulse is active
                    } else {
                        0.0 // Pulse has ended
                    }
                } else {
                    // Non-spiking neuron (high threshold): use voltage directly
                    state.u[pre]
                };
                total += weight * pre_signal;
            }
            let tau = network.neuron.tau_m[neuron];
            let u_inf = if tau > 0.0 { tau * total } else { total };
            let u_next = if tau > 0.0 {
                relax_towards(state.u[neuron], u_inf, tau, dt)
            } else {
                u_inf
            };
            let theta = network.neuron.theta0[neuron];
            if u_next >= theta && theta > 0.0 {
                // simple reset on spike
                spikes.push(SpikeEvent {
                    time: next_time,
                    neuron,
                });
                spiked_this_step[neuron] = true;
                state.u[neuron] = 0.0;

                // Set pulse end time for stretched output
                // If pulse_stretch_duration is configured, use it; otherwise default to dt (single timestep)
                let pulse_duration = network.neuron.pulse_stretch_duration
                    .get(neuron)
                    .copied()
                    .unwrap_or(0.0);
                if pulse_duration > 0.0 {
                    state.pulse_end_time[neuron] = next_time + pulse_duration;
                } else {
                    // No stretching: pulse ends after one timestep (backward compatible)
                    state.pulse_end_time[neuron] = next_time + dt;
                }
            } else {
                state.u[neuron] = u_next;
            }
            if let Some(avg) = average_activity.as_mut() {
                avg[neuron] += state.u[neuron];
            }
            // Track spike proximity for surrogate gradient
            if let Some(prox) = spike_proximity.as_mut() {
                let theta = network.neuron.theta0[neuron];
                if theta > 0.0 {
                    // Normalized distance from threshold: (u - theta) / theta
                    prox[neuron] += (state.u[neuron] - theta) / theta;
                }
            }
            // Update eligibility trace: proper low-pass filter (bounded, not accumulating)
            // e_trace[t+1] = e_trace[t] + (dt/tau) * (u[t] - e_trace[t])
            // This is equivalent to: e = decay * e + (1-decay) * u, keeping e bounded
            if let Some(e_trace) = eligibility_traces.as_mut() {
                let alpha = dt / eligibility_tau;
                e_trace[neuron] += alpha * (state.u[neuron] - e_trace[neuron]);
            }
        }

        state.time = next_time;

        if opts.record_readout {
            record_readouts(
                alpha_out,
                &state,
                &network.readouts,
                &mut readouts,
                &mut readout_state,
                dt,
            );
            times.push(state.time);
        }

        if !neuron_traces.is_empty() {
            for (&idx, trace) in neuron_traces.iter_mut() {
                let value = state.u.get(idx).copied().unwrap_or(0.0);
                trace.push(value);
            }
        }

        // avoid unused warning
        let _ = step;
    }

    if let Some(avg) = average_activity.as_mut() {
        let denom = steps.max(1) as f64;
        for value in avg.iter_mut() {
            *value /= denom;
        }
    }

    if let Some(prox) = spike_proximity.as_mut() {
        let denom = steps.max(1) as f64;
        for value in prox.iter_mut() {
            *value /= denom;
        }
    }

    let final_readouts = readout_state.into_iter().map(|(k, v)| (k, v)).collect();

    SimulationResult {
        times,
        readouts,
        average_activity,
        spikes,
        final_state: state,
        final_readouts,
        neuron_traces,
        spike_proximity,
        eligibility_traces,
    }
}

fn record_readouts(
    alpha_out: f64,
    state: &NetworkState,
    readouts: &[crate::network::compiled::CompiledReadout],
    buffers: &mut HashMap<String, Vec<Vec<f64>>>,
    accumulators: &mut HashMap<String, Vec<f64>>,
    dt: f64,
) {
    for readout in readouts {
        if let Some(entries) = buffers.get_mut(&readout.id) {
            let state_vec = accumulators
                .get_mut(&readout.id)
                .expect("missing readout accumulator state");
            let mut values = Vec::with_capacity(readout.indices.len());
            match readout.r#type {
                ReadoutRuntimeType::Analog => {
                    for (local_idx, &idx) in readout.indices.iter().enumerate() {
                        let input = state.u[idx] * alpha_out;
                        let value =
                            integrate_signal(state_vec, local_idx, input, readout.tau_s, dt);
                        values.push(value);
                    }
                }
                ReadoutRuntimeType::Spike => {
                    for (local_idx, &idx) in readout.indices.iter().enumerate() {
                        let input = if state.u[idx] > 0.0 { 1.0 } else { 0.0 };
                        let value =
                            integrate_signal(state_vec, local_idx, input, readout.tau_s, dt);
                        values.push(value);
                    }
                }
            }
            entries.push(values);
        }
    }
}

fn integrate_signal(
    accumulators: &mut [f64],
    index: usize,
    input: f64,
    tau: Option<f64>,
    dt: f64,
) -> f64 {
    if let Some(tau) = tau {
        if tau <= 0.0 || dt <= 0.0 {
            accumulators[index] = input;
            input
        } else {
            let prev = accumulators[index];
            let delta = (input - prev) * (dt / tau);
            let next = prev + delta;
            accumulators[index] = next;
            next
        }
    } else {
        accumulators[index] = input;
        input
    }
}
