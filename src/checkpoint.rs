//! Checkpoint system for saving and loading trained networks
//!
//! Provides serialization of network weights to JSON format for
//! persistence and inference with pre-trained models.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::layers::linear::{DEFAULT_SYNAPSE_NEG_GAIN, DEFAULT_SYNAPSE_POS_GAIN};
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
    /// Layer weights and biases (f32 for training/inference)
    pub weights: NetworkWeights,
    /// Quantized integer weights for hardware deployment
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantized: Option<QuantizedWeights>,
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
    /// Image width (for MNIST downsampling)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_width: Option<usize>,
    /// Image height (for MNIST downsampling)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_height: Option<usize>,
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
    #[serde(default = "ArchitectureInfo::default_threshold")]
    pub threshold: f32,
    /// Surrogate gradient slope
    #[serde(default = "ArchitectureInfo::default_slope")]
    pub slope: f32,
}

impl ArchitectureInfo {
    fn default_threshold() -> f32 {
        1.0
    }
    fn default_slope() -> f32 {
        25.0
    }
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

/// Quantized integer weights for hardware deployment
///
/// Hardware format: 3-bit magnitude + sign select + off
/// Separate scales for positive and negative current sources per layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedWeights {
    /// Number of magnitude bits (3 for the current hardware)
    pub magnitude_bits: u8,
    /// Max magnitude value (7 for 3-bit)
    pub max_magnitude: i8,
    /// FC1 positive scale: positive_weight = magnitude × fc1_pos_scale
    pub fc1_pos_scale: f32,
    /// FC1 negative scale: negative_weight = -magnitude × fc1_neg_scale
    pub fc1_neg_scale: f32,
    /// FC1 quantized weights: sign (bool) + magnitude (0-7), stored as signed i8
    pub fc1_weight: Vec<Vec<i8>>,
    /// FC2 positive scale
    pub fc2_pos_scale: f32,
    /// FC2 negative scale
    pub fc2_neg_scale: f32,
    /// FC2 quantized weights
    pub fc2_weight: Vec<Vec<i8>>,
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
        Self::from_network_quantized(network, metadata, None, None)
    }

    /// Create a checkpoint with optional quantized weights for hardware
    ///
    /// Args:
    ///   network: Trained network
    ///   metadata: Optional training metadata
    ///   quant_bits: If Some(bits), include quantized integer weights
    ///   image_dims: Optional (width, height) for non-square MNIST images
    pub fn from_network_quantized(
        network: &Network,
        metadata: Option<TrainingMetadata>,
        quant_bits: Option<u8>,
        image_dims: Option<(usize, usize)>,
    ) -> Self {
        let is_physics = network.is_physics_mode();

        // Extract architecture info from network using mode accessors
        let architecture = ArchitectureInfo {
            input_size: network.fc1.in_features,
            hidden_size: network.fc1.out_features,
            output_size: network.fc2.out_features,
            image_width: image_dims.map(|(w, _)| w),
            image_height: image_dims.map(|(_, h)| h),
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

        // Generate quantized weights if requested
        let fixed_fc2_scale = if network.fc2.fixed_quant_scale > 0.0 {
            Some(network.fc2.fixed_quant_scale)
        } else {
            None
        };
        let quantized = quant_bits.map(|bits| {
            quantize_network_weights(&network.fc1.weight, &network.fc2.weight, bits, fixed_fc2_scale)
        });

        Self {
            version: CHECKPOINT_VERSION,
            architecture,
            weights,
            quantized,
            metadata,
        }
    }

    /// Save checkpoint to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents =
            serde_json::to_string_pretty(self).with_context(|| "Failed to serialize checkpoint")?;

        fs::write(path.as_ref(), contents)
            .with_context(|| format!("Failed to write checkpoint file: {:?}", path.as_ref()))?;

        Ok(())
    }

    /// Load checkpoint from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read checkpoint file: {:?}", path.as_ref()))?;

        let checkpoint: Checkpoint =
            serde_json::from_str(&contents).with_context(|| "Failed to parse checkpoint JSON")?;

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
            synapse_pos_gain: DEFAULT_SYNAPSE_POS_GAIN,
            synapse_neg_gain: DEFAULT_SYNAPSE_NEG_GAIN,
            total_current_cap: None,
            fixed_quant_scale: 0.0,
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
            synapse_pos_gain: DEFAULT_SYNAPSE_POS_GAIN,
            synapse_neg_gain: DEFAULT_SYNAPSE_NEG_GAIN,
            total_current_cap: None,
            fixed_quant_scale: 0.0,
        };

        // Create LIF neurons with correct mode
        let spike_grad = SurrogateGradient::fast_sigmoid(arch.slope);

        let make_lif = |size: usize| -> Leaky {
            let lif = if arch.mode == "physics" {
                let tau_m = arch.tau_m.unwrap_or(0.0026);
                let dt = arch.dt.unwrap_or(0.001);
                if let Some(tau_pulse) = arch.tau_pulse {
                    let v_peak = 4.42; // Default hardware peak
                    Leaky::new_physics_with_pulse(size, tau_m, dt, tau_pulse, v_peak)
                } else {
                    Leaky::new_physics(size, tau_m, dt)
                }
            } else {
                Leaky::new(size, arch.beta.unwrap_or(0.9))
            };
            lif.with_threshold(arch.threshold)
                .with_spike_grad(spike_grad)
        };

        let lif1 = make_lif(arch.hidden_size);
        let lif2 = make_lif(arch.output_size);

        Ok(Network {
            fc1,
            lif1,
            fc2,
            lif2,
            spiking_input: false,
            spike_scale: 1.0,
        })
    }
}

// Conversion helpers

fn array2_to_vec(matrix: &Array2<f32>) -> Vec<Vec<f32>> {
    matrix.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn array1_to_vec(vector: &Array1<f32>) -> Vec<f32> {
    vector.to_vec()
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

/// Quantize network weights for hardware with separate pos/neg scales
///
/// Hardware: 3-bit magnitude (0-7) + sign select + off
/// Each layer gets separate scales for positive and negative weights
/// To reconstruct:
///   positive: magnitude × pos_scale
///   negative: -magnitude × neg_scale
fn quantize_network_weights(
    fc1: &Array2<f32>,
    fc2: &Array2<f32>,
    bits: u8,
    fixed_fc2_scale: Option<f32>,
) -> QuantizedWeights {
    let max_magnitude = ((1i32 << bits) - 1) as f32;

    fn quantize_layer(
        weights: &Array2<f32>,
        max_magnitude: f32,
        fixed_scale: Option<f32>,
    ) -> (f32, f32, Vec<Vec<i8>>) {
        let pos_scale = match fixed_scale {
            Some(s) if s > 0.0 => s,
            _ => {
                let pos_max = weights.iter().fold(0.0f32, |m, &w| m.max(w.max(0.0)));
                if pos_max > 1e-8 { pos_max / max_magnitude } else { 1.0 }
            }
        };
        let neg_scale = match fixed_scale {
            Some(s) if s > 0.0 => s,
            _ => {
                let neg_max = weights.iter().fold(0.0f32, |m, &w| m.max((-w).max(0.0)));
                if neg_max > 1e-8 { neg_max / max_magnitude } else { 1.0 }
            }
        };

        let quantized = weights
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .map(|&w| {
                        if w >= 0.0 {
                            let q = (w / pos_scale).round() as i8;
                            q.clamp(0, max_magnitude as i8)
                        } else {
                            let q = ((-w) / neg_scale).round() as i8;
                            -q.clamp(0, max_magnitude as i8)
                        }
                    })
                    .collect()
            })
            .collect();

        (pos_scale, neg_scale, quantized)
    }

    let (fc1_pos_scale, fc1_neg_scale, fc1_weight) =
        quantize_layer(fc1, max_magnitude, None); // fc1 always adaptive
    let (fc2_pos_scale, fc2_neg_scale, fc2_weight) =
        quantize_layer(fc2, max_magnitude, fixed_fc2_scale);

    QuantizedWeights {
        magnitude_bits: bits,
        max_magnitude: max_magnitude as i8,
        fc1_pos_scale,
        fc1_neg_scale,
        fc1_weight,
        fc2_pos_scale,
        fc2_neg_scale,
        fc2_weight,
    }
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

        // Default mirror calibration should be preserved for inference.
        assert!((net.fc1.synapse_pos_gain - DEFAULT_SYNAPSE_POS_GAIN).abs() < 1e-6);
        assert!((net.fc1.synapse_neg_gain - DEFAULT_SYNAPSE_NEG_GAIN).abs() < 1e-6);
        assert!((net.fc2.synapse_pos_gain - DEFAULT_SYNAPSE_POS_GAIN).abs() < 1e-6);
        assert!((net.fc2.synapse_neg_gain - DEFAULT_SYNAPSE_NEG_GAIN).abs() < 1e-6);
        assert!((net2.fc1.synapse_pos_gain - DEFAULT_SYNAPSE_POS_GAIN).abs() < 1e-6);
        assert!((net2.fc1.synapse_neg_gain - DEFAULT_SYNAPSE_NEG_GAIN).abs() < 1e-6);
        assert!((net2.fc2.synapse_pos_gain - DEFAULT_SYNAPSE_POS_GAIN).abs() < 1e-6);
        assert!((net2.fc2.synapse_neg_gain - DEFAULT_SYNAPSE_NEG_GAIN).abs() < 1e-6);

        // Verify weights match
        assert_eq!(net.fc1.weight.shape(), net2.fc1.weight.shape());
        assert_eq!(net.fc2.weight.shape(), net2.fc2.weight.shape());

        // Verify weights are identical
        let diff: f32 = (&net.fc1.weight - &net2.fc1.weight).mapv(|x| x.abs()).sum();
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

    #[test]
    fn test_checkpoint_roundtrip_36_9_10() {
        use ndarray::Array2;
        use std::path::Path;

        let cp_path = "models/tarski_36_9_10_emulator.json";
        if !Path::new(cp_path).exists() {
            eprintln!("Skipping: {} not found", cp_path);
            return;
        }

        let cp = Checkpoint::load(cp_path).unwrap();
        let network = cp.to_network().unwrap();

        println!("fc1: {:?}, fc2: {:?}", network.fc1.weight.shape(), network.fc2.weight.shape());
        println!("lif1 beta={:.6} threshold={} size={}", network.lif1.beta, network.lif1.threshold, network.lif1.size);
        println!("lif1 mode: {:?}", network.lif1.mode);

        // Run 25 steps with ones input
        // Use a real MNIST-like input (sample 0 is digit 7, 6x6 normalized)
        // Values from Python: range [-0.424, 0.912]
        let input_vec: Vec<f32> = vec![
            -0.424, -0.424, -0.424, -0.424, -0.424, -0.424,
            -0.424, -0.424,  0.246,  0.912,  0.415, -0.424,
            -0.424, -0.250,  0.744,  0.580,  0.912, -0.424,
            -0.424, -0.424, -0.424,  0.415,  0.580, -0.424,
            -0.424, -0.424,  0.080,  0.746,  0.246, -0.424,
            -0.424, -0.424,  0.415,  0.580, -0.250, -0.424,
        ];
        let input = Array2::from_shape_vec((1, 36), input_vec).unwrap();
        let mut state = network.init_state(1);
        let mut total_spikes = Array2::<f32>::zeros((1, 10));

        // Manually do what forward_step does, with debug prints
        let hidden_current = network.fc1.forward(&input);
        println!("Hidden current (fc1 output): {:?}", hidden_current.row(0));

        let (hidden_spk, lif1_state, _) = network.lif1.forward(&hidden_current, &state.lif1_state);
        println!("After lif1: mem={:?}", lif1_state.mem.row(0));
        println!("After lif1: spk={:?}", hidden_spk.row(0));

        for t in 0..25 {
            let (spk, mem, new_state, _) = network.forward_step(&input, &state);
            total_spikes = &total_spikes + &spk;
            if t == 0 || t == 5 || t == 24 {
                let h_max = new_state.lif1_state.mem.row(0).iter().cloned().reduce(f32::max).unwrap();
                let o_max = mem.row(0).iter().cloned().reduce(f32::max).unwrap();
                let h_spk: f32 = spk.row(0).iter().sum();
                println!("  t={}: hidden_mem_max={:.4} output_mem_max={:.4} output_spikes={:.0}", t, h_max, o_max, h_spk);
            }
            state = new_state;
        }

        println!("Total output spikes: {:?}", total_spikes.row(0));
        let any_spikes = total_spikes.iter().any(|&s| s > 0.0);
        println!("Any spikes: {}", any_spikes);
    }
}
