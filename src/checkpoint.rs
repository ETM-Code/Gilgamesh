//! Checkpoint system for saving and loading trained networks
//!
//! Provides serialization of network weights to JSON format for
//! persistence and inference with pre-trained models.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::layers::Linear;
use crate::network::Network;
use crate::neurons::Leaky;
use crate::surrogate::SurrogateGradient;

/// Checkpoint format version for backwards compatibility
const CHECKPOINT_VERSION: u32 = 1;

/// Serializable checkpoint containing network weights and metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Format version
    pub version: u32,
    /// Network architecture info
    pub architecture: ArchitectureInfo,
    /// Layer weights and biases
    pub weights: NetworkWeights,
    /// Optional training metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<TrainingMetadata>,
}

/// Network architecture specification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchitectureInfo {
    /// Input layer size
    pub input_size: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Output layer size
    pub output_size: usize,
    /// Network mode: "simple" or "physics"
    pub mode: String,
    /// Beta (membrane decay) - for simple mode
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta: Option<f32>,
    /// Membrane time constant - for physics mode
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tau_m: Option<f32>,
    /// Integration timestep - for physics mode
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dt: Option<f32>,
    /// Pulse stretching time constant
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tau_pulse: Option<f32>,
    /// Spike threshold
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Surrogate gradient slope
    #[serde(default = "default_slope")]
    pub slope: f32,
}

fn default_threshold() -> f32 {
    1.0
}
fn default_slope() -> f32 {
    25.0
}

/// Serializable network weights (using nested Vec for JSON compatibility)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkWeights {
    /// FC1 weight matrix [in_features][out_features]
    pub fc1_weight: Vec<Vec<f32>>,
    /// FC1 bias vector (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fc1_bias: Option<Vec<f32>>,
    /// FC2 weight matrix [in_features][out_features]
    pub fc2_weight: Vec<Vec<f32>>,
    /// FC2 bias vector (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fc2_bias: Option<Vec<f32>>,
}

/// Training metadata for checkpoint
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Number of epochs trained
    pub epochs_trained: usize,
    /// Final training accuracy
    pub final_train_accuracy: f32,
    /// Final test accuracy
    pub final_test_accuracy: f32,
    /// Training loss at end
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_loss: Option<f32>,
    /// Config file used (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_file: Option<String>,
}

impl Checkpoint {
    /// Create a checkpoint from a trained network
    pub fn from_network(network: &Network, metadata: Option<TrainingMetadata>) -> Self {
        let is_physics = network.is_physics_mode();

        // Extract architecture info from network using mode accessors
        let architecture = ArchitectureInfo {
            input_size: network.fc1.in_features,
            hidden_size: network.fc1.out_features,
            output_size: network.fc2.out_features,
            mode: if is_physics {
                "physics".to_string()
            } else {
                "simple".to_string()
            },
            beta: if is_physics {
                None
            } else {
                Some(network.lif1.beta)
            },
            tau_m: network.lif1.mode.tau_m(),
            dt: network.lif1.mode.dt(),
            tau_pulse: if network.lif1.mode.tau_pulse() > 0.0 {
                Some(network.lif1.mode.tau_pulse())
            } else {
                None
            },
            threshold: network.lif1.threshold,
            slope: network.lif1.spike_grad.slope(),
        };

        // Convert weights to nested Vec
        let weights = NetworkWeights {
            fc1_weight: array2_to_vec(&network.fc1.weight),
            fc1_bias: network.fc1.bias.as_ref().map(array1_to_vec),
            fc2_weight: array2_to_vec(&network.fc2.weight),
            fc2_bias: network.fc2.bias.as_ref().map(array1_to_vec),
        };

        Self {
            version: CHECKPOINT_VERSION,
            architecture,
            weights,
            metadata,
        }
    }

    /// Save checkpoint to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = serde_json::to_string_pretty(self)
            .with_context(|| "Failed to serialize checkpoint")?;

        fs::write(path.as_ref(), contents)
            .with_context(|| format!("Failed to write checkpoint file: {:?}", path.as_ref()))?;

        Ok(())
    }

    /// Load checkpoint from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read checkpoint file: {:?}", path.as_ref()))?;

        let checkpoint: Checkpoint = serde_json::from_str(&contents)
            .with_context(|| "Failed to parse checkpoint JSON")?;

        // Version check
        if checkpoint.version > CHECKPOINT_VERSION {
            anyhow::bail!(
                "Checkpoint version {} is newer than supported version {}",
                checkpoint.version,
                CHECKPOINT_VERSION
            );
        }

        Ok(checkpoint)
    }

    /// Reconstruct a Network from this checkpoint
    pub fn to_network(&self) -> Result<Network> {
        let arch = &self.architecture;

        // Reconstruct Linear layers with weights
        let fc1 = Linear {
            weight: vec_to_array2(&self.weights.fc1_weight)?,
            bias: self
                .weights
                .fc1_bias
                .as_ref()
                .map(|b| vec_to_array1(b))
                .transpose()?,
            in_features: arch.input_size,
            out_features: arch.hidden_size,
            current_gain: None,
        };

        let fc2 = Linear {
            weight: vec_to_array2(&self.weights.fc2_weight)?,
            bias: self
                .weights
                .fc2_bias
                .as_ref()
                .map(|b| vec_to_array1(b))
                .transpose()?,
            in_features: arch.hidden_size,
            out_features: arch.output_size,
            current_gain: None,
        };

        // Create LIF neurons with correct mode
        let spike_grad = SurrogateGradient::fast_sigmoid(arch.slope);

        let (lif1, lif2) = if arch.mode == "physics" {
            let tau_m = arch.tau_m.unwrap_or(0.0026);
            let dt = arch.dt.unwrap_or(0.001);

            let mut lif1 = if let Some(tau_pulse) = arch.tau_pulse {
                let v_peak = 4.42; // Default hardware peak
                Leaky::new_physics_with_pulse(arch.hidden_size, tau_m, dt, tau_pulse, v_peak)
            } else {
                Leaky::new_physics(arch.hidden_size, tau_m, dt)
            };
            lif1 = lif1.with_threshold(arch.threshold).with_spike_grad(spike_grad);

            let mut lif2 = if let Some(tau_pulse) = arch.tau_pulse {
                let v_peak = 4.42;
                Leaky::new_physics_with_pulse(arch.output_size, tau_m, dt, tau_pulse, v_peak)
            } else {
                Leaky::new_physics(arch.output_size, tau_m, dt)
            };
            lif2 = lif2.with_threshold(arch.threshold).with_spike_grad(spike_grad);

            (lif1, lif2)
        } else {
            let beta = arch.beta.unwrap_or(0.9);
            let lif1 = Leaky::new(arch.hidden_size, beta)
                .with_threshold(arch.threshold)
                .with_spike_grad(spike_grad);
            let lif2 = Leaky::new(arch.output_size, beta)
                .with_threshold(arch.threshold)
                .with_spike_grad(spike_grad);
            (lif1, lif2)
        };

        Ok(Network {
            fc1,
            lif1,
            fc2,
            lif2,
        })
    }
}

// Conversion helpers

fn array2_to_vec(arr: &Array2<f32>) -> Vec<Vec<f32>> {
    arr.rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect()
}

fn array1_to_vec(arr: &Array1<f32>) -> Vec<f32> {
    arr.to_vec()
}

fn vec_to_array2(vec: &[Vec<f32>]) -> Result<Array2<f32>> {
    if vec.is_empty() {
        anyhow::bail!("Cannot create Array2 from empty Vec");
    }

    let rows = vec.len();
    let cols = vec[0].len();

    // Verify all rows have same length
    if !vec.iter().all(|row| row.len() == cols) {
        anyhow::bail!("Inconsistent row lengths in weight matrix");
    }

    let flat: Vec<f32> = vec.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows, cols), flat)
        .with_context(|| "Failed to create Array2 from weight data")
}

fn vec_to_array1(vec: &[f32]) -> Result<Array1<f32>> {
    Ok(Array1::from_vec(vec.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_roundtrip() {
        // Create a network
        let net = Network::new(36, 100, 10, 0.9, 42);

        // Create checkpoint
        let metadata = TrainingMetadata {
            epochs_trained: 15,
            final_train_accuracy: 98.5,
            final_test_accuracy: 96.2,
            final_loss: Some(0.05),
            config_file: None,
        };
        let checkpoint = Checkpoint::from_network(&net, Some(metadata));

        // Verify architecture
        assert_eq!(checkpoint.architecture.input_size, 36);
        assert_eq!(checkpoint.architecture.hidden_size, 100);
        assert_eq!(checkpoint.architecture.output_size, 10);
        assert_eq!(checkpoint.architecture.mode, "physics"); // Now defaults to Physics mode

        // Reconstruct network
        let net2 = checkpoint.to_network().unwrap();

        // Verify weights match
        assert_eq!(net.fc1.weight.shape(), net2.fc1.weight.shape());
        assert_eq!(net.fc2.weight.shape(), net2.fc2.weight.shape());

        // Verify weights are identical
        let diff: f32 = (&net.fc1.weight - &net2.fc1.weight)
            .mapv(|x| x.abs())
            .sum();
        assert!(diff < 1e-6, "FC1 weights should be identical");
    }

    #[test]
    fn test_checkpoint_physics_mode() {
        let net = Network::new_physics(36, 100, 10, 0.0026, 0.001, 42);
        let checkpoint = Checkpoint::from_network(&net, None);

        assert_eq!(checkpoint.architecture.mode, "physics");
        assert!(checkpoint.architecture.tau_m.is_some());
        assert!(checkpoint.architecture.dt.is_some());
        assert!(checkpoint.architecture.beta.is_none());

        let net2 = checkpoint.to_network().unwrap();
        assert!(net2.is_physics_mode());
    }

    #[test]
    fn test_array_conversion() {
        use ndarray::array;

        let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let vec = array2_to_vec(&arr);
        let arr2 = vec_to_array2(&vec).unwrap();

        assert_eq!(arr.shape(), arr2.shape());
        let diff: f32 = (&arr - &arr2).mapv(|x| x.abs()).sum();
        assert!(diff < 1e-6);
    }
}
