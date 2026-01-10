//! Simulation trace types for visualization
//!
//! Captures per-timestep data for debugging, visualization, and SPICE comparison.

use ndarray::Array2;

/// Detailed trace of network simulation for visualization and analysis
///
/// Captures per-timestep membrane potentials, spikes, and currents for
/// debugging, visualization, and comparison with SPICE simulation.
#[derive(Clone, Debug)]
pub struct SimulationTrace {
    /// Membrane potentials over time for hidden layer [timestep][batch, neuron]
    pub hidden_mem_history: Vec<Array2<f32>>,
    /// Membrane potentials over time for output layer [timestep][batch, neuron]
    pub output_mem_history: Vec<Array2<f32>>,
    /// Spike events over time for hidden layer [timestep][batch, neuron]
    pub hidden_spike_history: Vec<Array2<f32>>,
    /// Spike events over time for output layer [timestep][batch, neuron]
    pub output_spike_history: Vec<Array2<f32>>,
    /// Input currents to hidden layer [timestep][batch, neuron]
    pub hidden_current_history: Vec<Array2<f32>>,
    /// Input currents to output layer [timestep][batch, neuron]
    pub output_current_history: Vec<Array2<f32>>,
    /// Final accumulated spike counts for output [batch, neuron]
    pub output_spike_count: Array2<f32>,
    /// Final membrane potential for output [batch, neuron]
    pub output_final_mem: Array2<f32>,
}

impl SimulationTrace {
    /// Get prediction for each sample in the batch (argmax of spike counts)
    pub fn predictions(&self) -> Vec<usize> {
        self.output_spike_count
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Get spike counts as probabilities (normalized)
    pub fn probabilities(&self) -> Array2<f32> {
        let mut probs = self.output_spike_count.clone();
        for mut row in probs.rows_mut() {
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            } else {
                let uniform = 1.0 / row.len() as f32;
                row.fill(uniform);
            }
        }
        probs
    }

    /// Number of timesteps in the trace
    pub fn num_timesteps(&self) -> usize {
        self.hidden_mem_history.len()
    }
}
