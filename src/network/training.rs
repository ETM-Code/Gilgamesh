use crate::network::compiled::{CompiledNetwork, LayerRuntimeType, StimulusChannel};
use crate::network::runtime::{simulate_network, SimulationOptions, SimulationResult};
use crate::training::surrogate::SurrogateType;
use anyhow::{anyhow, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub targets: HashMap<String, Vec<f64>>,
    pub weight: f64,
    pub simulation: SimulationOptions,
    pub encoder: Option<TemporalEncoder>,
}

impl TrainingExample {
    pub fn new(targets: HashMap<String, Vec<f64>>, simulation: SimulationOptions) -> Self {
        Self {
            targets,
            weight: 1.0,
            simulation,
            encoder: None,
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.max(0.0);
        self
    }

    pub fn with_encoder(mut self, encoder: TemporalEncoder) -> Self {
        self.encoder = Some(encoder);
        self
    }
}

/// Configuration for surrogate gradient during backpropagation.
#[derive(Debug, Clone, Copy)]
pub struct SurrogateConfig {
    /// The surrogate gradient function to use.
    pub surrogate: SurrogateType,
    /// Whether surrogate gradient is enabled.
    pub enabled: bool,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            surrogate: SurrogateType::default(),
            enabled: true,
        }
    }
}

impl SurrogateConfig {
    /// Create a new config with FastSigmoid surrogate.
    pub fn fast_sigmoid(slope: f64) -> Self {
        use crate::training::surrogate::FastSigmoid;
        Self {
            surrogate: SurrogateType::FastSigmoid(FastSigmoid::new(slope)),
            enabled: true,
        }
    }

    /// Create a config with surrogate gradients disabled (for comparison).
    pub fn disabled() -> Self {
        Self {
            surrogate: SurrogateType::default(),
            enabled: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default)]
    pub regularization: f64,
    #[serde(default = "default_row_spacing")]
    pub row_spacing_start: f64,
    #[serde(default = "default_row_spacing")]
    pub row_spacing_end: f64,
    #[serde(default = "default_pulse_width")]
    pub pulse_width: f64,
    #[serde(default = "default_input_scale")]
    pub input_scale: f64,
    #[serde(default)]
    pub sample_limit: Option<usize>,
    /// Surrogate gradient configuration (not serializable, set programmatically).
    #[serde(skip)]
    pub surrogate: SurrogateConfig,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            epochs: default_epochs(),
            regularization: 0.0,
            row_spacing_start: default_row_spacing(),
            row_spacing_end: default_row_spacing(),
            pulse_width: default_pulse_width(),
            input_scale: default_input_scale(),
            sample_limit: None,
            surrogate: SurrogateConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingLog {
    pub epoch: usize,
    pub loss: f64,
}

const EPS: f64 = 1e-12;

fn default_learning_rate() -> f64 {
    1e-3
}

fn default_epochs() -> usize {
    25
}

fn default_row_spacing() -> f64 {
    1e-3
}

fn default_pulse_width() -> f64 {
    0.6
}

fn default_input_scale() -> f64 {
    1.0
}

impl TrainerConfig {
    pub fn spacing_for_epoch(&self, epoch: usize) -> f64 {
        if self.epochs <= 1 {
            return self.row_spacing_start.max(1e-6);
        }
        let last = (self.epochs - 1).max(1) as f64;
        let fraction = (epoch as f64).min(last) / last;
        let spacing =
            self.row_spacing_start + (self.row_spacing_end - self.row_spacing_start) * fraction;
        spacing.max(1e-6)
    }

    pub fn pulse_width_for_spacing(&self, spacing: f64) -> f64 {
        if self.pulse_width <= 0.0 {
            spacing
        } else if self.pulse_width <= 1.0 {
            (spacing * self.pulse_width).max(1e-9)
        } else {
            self.pulse_width.min(spacing)
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemporalEncoder {
    columns: usize,
    rows: usize,
    column_values: Vec<Vec<f64>>,
    neuron_indices: Vec<usize>,
}

impl TemporalEncoder {
    pub fn new(
        columns: usize,
        rows: usize,
        column_values: Vec<Vec<f64>>,
        neuron_indices: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(columns, column_values.len());
        debug_assert_eq!(columns, neuron_indices.len());
        Self {
            columns,
            rows,
            column_values,
            neuron_indices,
        }
    }

    pub fn build_stimuli(
        &self,
        spacing: f64,
        pulse_width: f64,
        scale: f64,
    ) -> Vec<StimulusChannel> {
        let total_time = self.suggested_duration(spacing);
        let effective_width = pulse_width.clamp(1e-9, spacing);
        let mut channels = Vec::with_capacity(self.columns);

        for (col_idx, (column, &neuron_idx)) in self
            .column_values
            .iter()
            .zip(self.neuron_indices.iter())
            .enumerate()
        {
            let mut times = Vec::with_capacity(self.rows * 2 + 2);
            let mut values = Vec::with_capacity(self.rows * 2 + 2);
            times.push(0.0);
            values.push(0.0);

            for (row_idx, &value) in column.iter().enumerate() {
                let start = (row_idx as f64).mul_add(spacing, 0.0);
                let amplitude = value * scale;
                times.push(start);
                values.push(amplitude);

                // Hold amplitude until end for proper square pulse
                let end = (start + effective_width).min(total_time);
                times.push(end);
                values.push(amplitude);  // Still at amplitude

                // Then drop to zero
                times.push(end + 1e-9);
                values.push(0.0);
            }

            times.push(total_time);
            values.push(0.0);
            enforce_monotonic_times(&mut times);

            channels.push(StimulusChannel {
                id: format!("encoder_col{}", col_idx),
                target_indices: vec![neuron_idx],
                times,
                values,
            });
        }

        channels
    }

    pub fn suggested_duration(&self, spacing: f64) -> f64 {
        spacing * (self.rows as f64 + 1.0)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }
}

fn enforce_monotonic_times(times: &mut [f64]) {
    let mut last = -f64::INFINITY;
    for t in times.iter_mut() {
        if *t <= last {
            *t = last + 1e-12;
        }
        last = *t;
    }
}

pub fn train_network(
    network: &mut CompiledNetwork,
    dataset: &[TrainingExample],
    config: &TrainerConfig,
) -> Result<Vec<TrainingLog>> {
    train_network_with_callback(network, dataset, config, |_net, _cfg, _log| Ok(()))
}

pub fn train_network_with_callback<F>(
    network: &mut CompiledNetwork,
    dataset: &[TrainingExample],
    config: &TrainerConfig,
    mut on_epoch: F,
) -> Result<Vec<TrainingLog>>
where
    F: FnMut(&CompiledNetwork, &TrainerConfig, &TrainingLog) -> Result<()>,
{
    if dataset.is_empty() {
        return Err(anyhow!("training dataset must not be empty"));
    }

    let synapse_count = network.synapse_count();
    let neuron_count = network.neuron_count();
    let mut grad = vec![0.0; synapse_count];
    let mut logs = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let spacing = config.spacing_for_epoch(epoch);
        let pulse_width = config.pulse_width_for_spacing(spacing);
        let input_scale = config.input_scale.max(0.0);

        // BATCH PARALLELISM: Process all training samples in parallel
        // Each thread computes its own local gradient, then we sum them
        let results: Vec<_> = dataset
            .par_iter()
            .map(|example| {
                let mut local_grad = vec![0.0; synapse_count];
                let mut local_neuron_error = vec![0.0; neuron_count];

                let mut sim_opts = example.simulation.clone();
                sim_opts.record_readout = true;
                sim_opts.return_average = true;
                sim_opts.return_spike_proximity = config.surrogate.enabled;
                sim_opts.return_eligibility_traces = true;
                sim_opts.return_synapse_eligibility = true;

                if let Some(encoder) = &example.encoder {
                    let stimuli = encoder.build_stimuli(spacing, pulse_width, input_scale);
                    let duration = encoder.suggested_duration(spacing);
                    sim_opts.stimuli_override = Some(stimuli);
                    sim_opts.t_end = sim_opts.t_end.max(duration);
                    let suggested_dt = (spacing / 20.0).max(1e-6);
                    sim_opts.dt = sim_opts.dt.min(suggested_dt);
                }

                let result = simulate_network(network, &sim_opts);

                let activity = match result.eligibility_traces.as_ref() {
                    Some(a) => a,
                    None => return (local_grad, 0.0, 0.0), // Skip on error
                };

                let synapse_eligibility = match result.synapse_eligibility.as_ref() {
                    Some(s) => s,
                    None => return (local_grad, 0.0, 0.0),
                };

                local_neuron_error.fill(0.0);
                let mut local_loss = 0.0;

                for readout in &network.readouts {
                    let target_values = match example.targets.get(&readout.id) {
                        Some(values) => values,
                        None => continue,
                    };

                    let samples = match result.readouts.get(&readout.id) {
                        Some(s) => s,
                        None => continue,
                    };

                    let num_outputs = target_values.len();
                    if samples.is_empty() || samples[0].len() != num_outputs {
                        continue;
                    }

                    // Max-over-time readout
                    let max_over_time: Vec<f64> = (0..num_outputs)
                        .map(|i| samples.iter().map(|s| s[i]).fold(f64::NEG_INFINITY, f64::max))
                        .collect();

                    let temperature = 10.0;
                    let max_z = max_over_time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let scaled: Vec<f64> = max_over_time.iter().map(|z| (z - max_z) / temperature).collect();
                    let exp_sum: f64 = scaled.iter().map(|z| z.exp()).sum();
                    let softmax: Vec<f64> = scaled.iter().map(|z| z.exp() / exp_sum).collect();

                    let target_class = target_values.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    local_loss += -(softmax[target_class] + EPS).ln() * example.weight;

                    for (local_idx, &neuron_idx) in readout.indices.iter().enumerate() {
                        let grad = (softmax[local_idx] - target_values[local_idx]) / temperature;
                        local_neuron_error[neuron_idx] += grad;
                    }
                }

                backpropagate(
                    network,
                    &mut local_grad,
                    &mut local_neuron_error,
                    activity,
                    synapse_eligibility,
                    &result,
                    &config.surrogate,
                    example.weight,
                );

                (local_grad, local_loss, example.weight)
            })
            .collect();

        // Sum all local gradients and losses
        grad.fill(0.0);
        let mut total_loss = 0.0;
        let mut total_weight = 0.0;
        for (local_grad, local_loss, weight) in results {
            for (g, lg) in grad.iter_mut().zip(local_grad.iter()) {
                *g += lg;
            }
            total_loss += local_loss;
            total_weight += weight;
        }

        if total_weight <= EPS {
            total_weight = 1.0;
        }

        apply_gradient(network, &grad, config, total_weight);

        let log = TrainingLog {
            epoch: epoch + 1,
            loss: total_loss / total_weight,
        };
        on_epoch(network, config, &log)?;
        logs.push(log);
    }

    Ok(logs)
}

fn backpropagate(
    network: &CompiledNetwork,
    grad: &mut [f64],
    neuron_error: &mut [f64],
    _activity: &[f64],  // Kept for potential fallback; e-prop uses synapse_eligibility
    synapse_eligibility: &[f64],
    result: &SimulationResult,
    surrogate_config: &SurrogateConfig,
    weight: f64,
) {
    let layers = &network.layers;
    let row_ptr = &network.incoming.row_ptr;
    let src = &network.incoming.src;
    let g = &network.incoming.g;
    let thresholds = &network.neuron.theta0;

    // Get spike proximity if available for surrogate gradient (used for error backprop)
    let spike_proximity = result.spike_proximity.as_ref();

    for layer in layers.iter().rev() {
        let start = layer.offset;
        let end = start + layer.size;

        if matches!(layer.layer_type, LayerRuntimeType::Input) {
            continue;
        }

        // Surrogate gradient only for hidden layers, not output/readout (which use analog voltage)
        let is_hidden_layer = matches!(layer.layer_type, LayerRuntimeType::Hidden);

        for neuron in start..end {
            let err = neuron_error[neuron];
            if err.abs() < EPS {
                continue;
            }

            // Compute surrogate gradient scaling for error backpropagation
            // Only apply to hidden layers - output layers use analog voltage readout
            let surrogate_scale = if surrogate_config.enabled && is_hidden_layer {
                if let Some(prox) = spike_proximity {
                    let theta = thresholds[neuron];
                    let u_minus_theta = prox[neuron] * theta;
                    surrogate_config.surrogate.gradient(u_minus_theta + theta, theta)
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let in_start = row_ptr[neuron];
            let in_end = row_ptr[neuron + 1];
            for edge_idx in in_start..in_end {
                let pre = src[edge_idx];
                // E-PROP: Use per-synapse eligibility trace for weight gradient
                // synapse_eligibility already contains surrogate(post) * pre_activity with temporal weighting
                // So we multiply by error directly, not scaled_err
                grad[edge_idx] += weight * err * synapse_eligibility[edge_idx];
                // Error backpropagation still uses surrogate scaling
                neuron_error[pre] += err * surrogate_scale * g[edge_idx];
            }
            neuron_error[neuron] = 0.0;
        }
    }
}

#[allow(dead_code)]
fn backpropagate_old(
    network: &CompiledNetwork,
    grad: &mut [f64],
    neuron_error: &mut [f64],
    activity: &[f64],
    result: &SimulationResult,
    surrogate_config: &SurrogateConfig,
    weight: f64,
) {
    let layers = &network.layers;
    let row_ptr = &network.incoming.row_ptr;
    let src = &network.incoming.src;
    let g = &network.incoming.g;
    let thresholds = &network.neuron.theta0;
    let spike_proximity = result.spike_proximity.as_ref();

    for layer in layers.iter().rev() {
        let start = layer.offset;
        let end = start + layer.size;

        if matches!(layer.layer_type, LayerRuntimeType::Input) {
            continue;
        }

        let is_hidden_layer = matches!(layer.layer_type, LayerRuntimeType::Hidden);

        for neuron in start..end {
            let err = neuron_error[neuron];
            if err.abs() < EPS {
                continue;
            }

            let surrogate_scale = if surrogate_config.enabled && is_hidden_layer {
                if let Some(prox) = spike_proximity {
                    let theta = thresholds[neuron];
                    let u_minus_theta = prox[neuron] * theta;
                    surrogate_config.surrogate.gradient(u_minus_theta + theta, theta)
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let scaled_err = err * surrogate_scale;

            let in_start = row_ptr[neuron];
            let in_end = row_ptr[neuron + 1];
            for edge_idx in in_start..in_end {
                let pre = src[edge_idx];
                grad[edge_idx] += weight * scaled_err * activity[pre];
                neuron_error[pre] += scaled_err * g[edge_idx];
            }
            neuron_error[neuron] = 0.0;
        }
    }
}

fn apply_gradient(
    network: &mut CompiledNetwork,
    grad: &[f64],
    config: &TrainerConfig,
    total_weight: f64,
) {
    let incoming = &mut network.incoming;
    let lr = config.learning_rate / total_weight;
    let reg = config.regularization;

    for (idx, g_val) in incoming.g.iter_mut().enumerate() {
        let mut update = lr * grad[idx];
        if reg > 0.0 {
            update += reg * (*g_val);
        }
        *g_val -= update;
    }
}
