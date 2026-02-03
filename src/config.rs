//! Configuration system for gilgamesh
//!
//! Provides JSON-configurable parameters for network architecture, training,
//! physics simulation, hardware constraints, and robustness features.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Top-level configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    /// Operation mode: "simple" or "physics"
    #[serde(default = "default_mode")]
    pub mode: String,

    /// Network architecture
    #[serde(default)]
    pub network: NetworkConfig,

    /// Neuron parameters
    #[serde(default)]
    pub neuron: NeuronConfig,

    /// Physics simulation parameters (used when mode="physics")
    #[serde(default)]
    pub physics: PhysicsConfig,

    /// Hardware constraints
    #[serde(default)]
    pub hardware: HardwareConfig,

    /// Training parameters
    #[serde(default)]
    pub training: TrainingConfig,

    /// Input encoding options
    #[serde(default)]
    pub input_encoding: InputEncodingConfig,

    /// Weight quantization (for digipot simulation)
    #[serde(default)]
    pub quantization: QuantizationConfig,

    /// Noise injection for robustness
    #[serde(default)]
    pub noise: NoiseConfig,

    /// Output mode selection
    #[serde(default)]
    pub output: OutputConfig,
}

fn default_mode() -> String {
    "simple".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: default_mode(),
            network: NetworkConfig::default(),
            neuron: NeuronConfig::default(),
            physics: PhysicsConfig::default(),
            hardware: HardwareConfig::default(),
            training: TrainingConfig::default(),
            input_encoding: InputEncodingConfig::default(),
            quantization: QuantizationConfig::default(),
            noise: NoiseConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;

        let config: Config = serde_json::from_str(&contents)
            .with_context(|| "Failed to parse config JSON")?;

        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = serde_json::to_string_pretty(self)
            .with_context(|| "Failed to serialize config")?;

        fs::write(path.as_ref(), contents)
            .with_context(|| format!("Failed to write config file: {:?}", path.as_ref()))?;

        Ok(())
    }

    /// Check if physics mode is enabled
    pub fn is_physics_mode(&self) -> bool {
        self.mode == "physics"
    }
}

/// Network architecture configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Input layer size (default: 36 for 6x6 MNIST)
    #[serde(default = "default_input_size")]
    pub input_size: usize,

    /// Hidden layer size
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Output layer size (default: 10 for MNIST digits)
    #[serde(default = "default_output_size")]
    pub output_size: usize,

    /// Image size for square MNIST downsampling (n×n, produces n² input features)
    /// Default: 6 (produces 36 input features). Ignored if image_width/height set.
    #[serde(default = "default_image_size")]
    pub image_size: usize,

    /// Image width for non-square images (overrides image_size if set)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_width: Option<usize>,

    /// Image height for non-square images (overrides image_size if set)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_height: Option<usize>,
}

fn default_input_size() -> usize { 36 }
fn default_hidden_size() -> usize { 12 }
fn default_output_size() -> usize { 10 }
fn default_image_size() -> usize { 6 }

impl NetworkConfig {
    /// Get effective image width
    pub fn get_width(&self) -> usize {
        self.image_width.unwrap_or(self.image_size)
    }

    /// Get effective image height
    pub fn get_height(&self) -> usize {
        self.image_height.unwrap_or(self.image_size)
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            input_size: default_input_size(),
            hidden_size: default_hidden_size(),
            output_size: default_output_size(),
            image_size: default_image_size(),
            image_width: None,
            image_height: None,
        }
    }
}

/// Neuron parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronConfig {
    /// Membrane decay rate (0 to 1)
    #[serde(default = "default_beta")]
    pub beta: f32,

    /// Spike threshold
    #[serde(default = "default_threshold")]
    pub threshold: f32,

    /// Reset mechanism: "subtract", "zero", or "none"
    #[serde(default = "default_reset_mechanism")]
    pub reset_mechanism: String,

    /// Surrogate gradient slope
    #[serde(default = "default_slope")]
    pub slope: f32,
}

fn default_beta() -> f32 { 0.9 }
fn default_threshold() -> f32 { 1.0 }
fn default_reset_mechanism() -> String { "subtract".to_string() }
fn default_slope() -> f32 { 25.0 }

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            beta: default_beta(),
            threshold: default_threshold(),
            reset_mechanism: default_reset_mechanism(),
            slope: default_slope(),
        }
    }
}

/// Physics simulation parameters (gilgamesh-derived defaults)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// Enable physics mode
    #[serde(default)]
    pub enabled: bool,

    /// Membrane time constant (seconds)
    #[serde(default = "default_tau_m")]
    pub tau_m: f32,

    /// Pulse stretching time constant (seconds)
    #[serde(default = "default_tau_pulse")]
    pub tau_pulse: f32,

    /// Threshold adaptation time constant (seconds)
    #[serde(default = "default_tau_theta")]
    pub tau_theta: f32,

    /// Integration timestep (seconds)
    #[serde(default = "default_dt")]
    pub dt: f32,

    /// Enable threshold adaptation
    #[serde(default)]
    pub adaptation_enabled: bool,

    /// Low threshold value (resting state)
    #[serde(default = "default_theta_low")]
    pub theta_low: f32,

    /// High threshold value (after spike)
    #[serde(default = "default_theta_high")]
    pub theta_high: f32,
}

fn default_tau_m() -> f32 { 0.00396 }     // ~3.96ms (33nF * 120kΩ)
fn default_tau_pulse() -> f32 { 1.5e-6 } // ~1.5us (R_pw * C_pw)
fn default_tau_theta() -> f32 { 0.001 }   // ~1ms
fn default_dt() -> f32 { 0.001 }          // 1ms timestep
fn default_theta_low() -> f32 { 1.0 }     // Resting threshold
fn default_theta_high() -> f32 { 1.2 }    // 20% increase after spike

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tau_m: default_tau_m(),
            tau_pulse: default_tau_pulse(),
            tau_theta: default_tau_theta(),
            dt: default_dt(),
            adaptation_enabled: false,
            theta_low: default_theta_low(),
            theta_high: default_theta_high(),
        }
    }
}

impl PhysicsConfig {
    /// Compute tau_m from beta and dt using: beta = exp(-dt/tau)
    pub fn tau_from_beta(beta: f32, dt: f32) -> f32 {
        -dt / beta.ln()
    }

    /// Compute beta from tau_m and dt using: beta = exp(-dt/tau)
    pub fn beta_from_tau(tau: f32, dt: f32) -> f32 {
        (-dt / tau).exp()
    }
}

/// Hardware constraints (from gilgamesh measurements)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Supply voltage rail
    #[serde(default = "default_v_rail")]
    pub v_rail: f32,

    /// Comparator output rail drop
    #[serde(default = "default_comp_rail_drop")]
    pub comp_rail_drop: f32,

    /// Diode forward voltage drop
    #[serde(default = "default_diode_drop")]
    pub diode_drop: f32,

    /// Minimum membrane voltage
    #[serde(default = "default_v_min")]
    pub v_min: f32,

    /// Maximum membrane voltage
    #[serde(default = "default_v_max")]
    pub v_max: f32,
}

fn default_v_rail() -> f32 { 5.0 }
fn default_comp_rail_drop() -> f32 { 0.21 }
fn default_diode_drop() -> f32 { 0.37 }
fn default_v_min() -> f32 { 0.0 }
fn default_v_max() -> f32 { 5.0 }

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            v_rail: default_v_rail(),
            comp_rail_drop: default_comp_rail_drop(),
            diode_drop: default_diode_drop(),
            v_min: default_v_min(),
            v_max: default_v_max(),
        }
    }
}

impl HardwareConfig {
    /// Compute effective pulse peak voltage
    pub fn pulse_peak(&self) -> f32 {
        self.v_rail - self.comp_rail_drop - self.diode_drop
    }
}

/// Training configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    #[serde(default = "default_lr")]
    pub lr: f32,

    /// Number of training epochs
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    /// Batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Number of timesteps per sample
    #[serde(default = "default_num_steps")]
    pub num_steps: usize,

    /// Random seed
    #[serde(default = "default_seed")]
    pub seed: u64,

    /// Number of parallel workers (0 = auto)
    #[serde(default)]
    pub num_workers: usize,

    /// Truncated BPTT steps (None/0 = full BPTT)
    #[serde(default)]
    pub bptt_steps: Option<usize>,
}

fn default_lr() -> f32 { 0.001 }
fn default_epochs() -> usize { 15 }
fn default_batch_size() -> usize { 128 }
fn default_num_steps() -> usize { 25 }
fn default_seed() -> u64 { 42 }

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lr: default_lr(),
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            num_steps: default_num_steps(),
            seed: default_seed(),
            num_workers: 0,
            bptt_steps: None,
        }
    }
}

/// Input encoding configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputEncodingConfig {
    /// Encoding type: "rate_coded" or "temporal"
    #[serde(default = "default_encoding_type")]
    pub encoding_type: String,

    /// Row spacing for temporal encoding (seconds)
    #[serde(default = "default_row_spacing")]
    pub row_spacing: f32,

    /// Pulse width as fraction of row spacing
    #[serde(default = "default_pulse_width")]
    pub pulse_width: f32,
}

fn default_encoding_type() -> String { "rate_coded".to_string() }
fn default_row_spacing() -> f32 { 0.0015 }
fn default_pulse_width() -> f32 { 0.9 }

impl Default for InputEncodingConfig {
    fn default() -> Self {
        Self {
            encoding_type: default_encoding_type(),
            row_spacing: default_row_spacing(),
            pulse_width: default_pulse_width(),
        }
    }
}

/// Weight quantization configuration (for digipot simulation)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Enable weight quantization
    #[serde(default)]
    pub enabled: bool,

    /// Number of magnitude bits (3 for current sources: 0-7 range)
    #[serde(default = "default_bits")]
    pub bits: u8,

    /// Apply quantization during training (QAT)
    #[serde(default = "default_apply_during_training")]
    pub apply_during_training: bool,

    /// Use symmetric quantization around 0
    #[serde(default = "default_symmetric")]
    pub symmetric: bool,
}

fn default_bits() -> u8 { 3 }
fn default_apply_during_training() -> bool { true }
fn default_symmetric() -> bool { true }

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bits: default_bits(),
            apply_during_training: default_apply_during_training(),
            symmetric: default_symmetric(),
        }
    }
}

impl QuantizationConfig {
    /// Quantize a weight value
    pub fn quantize(&self, w: f32) -> f32 {
        if !self.enabled {
            return w;
        }

        let levels = (1u32 << self.bits) as f32;

        if self.symmetric {
            // Symmetric quantization: [-max, +max] mapped to levels
            let max_val = w.abs();
            if max_val < 1e-8 {
                return 0.0;
            }
            let scale = (levels / 2.0) / max_val;
            ((w * scale).round() / scale).clamp(-max_val, max_val)
        } else {
            // Asymmetric: just round to nearest level
            (w * levels).round() / levels
        }
    }
}

/// Noise injection configuration for robustness training
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Enable noise injection
    #[serde(default)]
    pub enabled: bool,

    /// Only apply noise during training
    #[serde(default = "default_training_only")]
    pub training_only: bool,

    /// Weight noise std (relative to weight magnitude)
    #[serde(default = "default_weight_std")]
    pub weight_std: f32,

    /// Threshold noise std (relative to threshold)
    #[serde(default = "default_threshold_std")]
    pub threshold_std: f32,

    /// Membrane noise std (absolute)
    #[serde(default = "default_membrane_std")]
    pub membrane_std: f32,

    /// Input noise std (relative to input)
    #[serde(default = "default_input_std")]
    pub input_std: f32,
}

fn default_training_only() -> bool { true }
fn default_weight_std() -> f32 { 0.05 }
fn default_threshold_std() -> f32 { 0.02 }
fn default_membrane_std() -> f32 { 0.01 }
fn default_input_std() -> f32 { 0.1 }

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            training_only: default_training_only(),
            weight_std: default_weight_std(),
            threshold_std: default_threshold_std(),
            membrane_std: default_membrane_std(),
            input_std: default_input_std(),
        }
    }
}

/// Output mode configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output mode: "spike_count", "analog_final", "analog_max", "analog_filtered"
    #[serde(default = "default_output_mode")]
    pub mode: String,

    /// Filter time constant for analog_filtered mode
    #[serde(default = "default_filter_tau")]
    pub filter_tau: f32,

    /// Analog gain for hybrid spike+membrane inter-layer transmission (0.0 = disabled)
    #[serde(default)]
    pub analog_gain: f32,
}

fn default_output_mode() -> String { "spike_count".to_string() }
fn default_filter_tau() -> f32 { 0.002 }

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            mode: default_output_mode(),
            filter_tau: default_filter_tau(),
            analog_gain: 0.0, // Disabled by default
        }
    }
}

/// Output mode enum for type-safe mode selection
#[derive(Clone, Debug, PartialEq)]
pub enum OutputMode {
    /// Sum spikes over time (standard snnTorch)
    SpikeCount,
    /// Use final membrane voltage
    AnalogFinal,
    /// Use max membrane voltage over time
    AnalogMax,
    /// Use filtered membrane (low-pass)
    AnalogFiltered { tau_filter: f32 },
}

impl OutputConfig {
    /// Parse mode string into OutputMode enum
    pub fn to_mode(&self) -> OutputMode {
        match self.mode.as_str() {
            "analog_final" => OutputMode::AnalogFinal,
            "analog_max" => OutputMode::AnalogMax,
            "analog_filtered" => OutputMode::AnalogFiltered { tau_filter: self.filter_tau },
            _ => OutputMode::SpikeCount,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.mode, "simple");
        assert_eq!(config.network.input_size, 36);
        assert_eq!(config.neuron.beta, 0.9);
        assert!(!config.quantization.enabled);
        assert!(!config.noise.enabled);
    }

    #[test]
    fn test_tau_beta_conversion() {
        let beta = 0.9f32;
        let dt = 0.001f32;

        let tau = PhysicsConfig::tau_from_beta(beta, dt);
        let beta_back = PhysicsConfig::beta_from_tau(tau, dt);

        assert!((beta - beta_back).abs() < 1e-6);
        // tau should be approximately 9.5ms for beta=0.9, dt=1ms
        assert!((tau - 0.00949).abs() < 0.001);
    }

    #[test]
    fn test_quantization() {
        let mut config = QuantizationConfig::default();
        config.enabled = true;
        config.bits = 8;
        config.symmetric = false;

        // 8-bit quantization should give 256 levels
        let w = 0.5f32;
        let q = config.quantize(w);
        // Should be close to original within 1/256
        assert!((w - q).abs() < 1.0 / 256.0 + 1e-6);
    }

    #[test]
    fn test_hardware_pulse_peak() {
        let hw = HardwareConfig::default();
        let peak = hw.pulse_peak();
        // 5.0 - 0.21 - 0.37 = 4.42V
        assert!((peak - 4.42).abs() < 0.01);
    }

    #[test]
    fn test_output_mode_parsing() {
        let mut config = OutputConfig::default();
        assert_eq!(config.to_mode(), OutputMode::SpikeCount);

        config.mode = "analog_final".to_string();
        assert_eq!(config.to_mode(), OutputMode::AnalogFinal);

        config.mode = "analog_filtered".to_string();
        config.filter_tau = 0.005;
        match config.to_mode() {
            OutputMode::AnalogFiltered { tau_filter } => {
                assert!((tau_filter - 0.005).abs() < 1e-6);
            }
            _ => panic!("Expected AnalogFiltered"),
        }
    }
}
