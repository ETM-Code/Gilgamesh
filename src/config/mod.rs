//! Configuration system for gilgamesh
//!
//! Provides JSON-configurable parameters for network architecture, training,
//! physics simulation, hardware constraints, and robustness features.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;

use crate::neurons::lif::ResetMechanism;

/// Operation mode for the network
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationMode {
    /// Simple discrete-time model (snnTorch-compatible)
    Simple,
    /// Physics-accurate RC circuit model
    Physics,
}

impl fmt::Display for OperationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperationMode::Simple => write!(f, "simple"),
            OperationMode::Physics => write!(f, "physics"),
        }
    }
}

/// Top-level configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Operation mode
    pub mode: OperationMode,
    /// Network architecture
    pub network: NetworkConfig,
    /// Neuron parameters
    pub neuron: NeuronConfig,
    /// Physics simulation parameters (used when mode="physics")
    pub physics: PhysicsConfig,
    /// Hardware constraints
    pub hardware: HardwareConfig,
    /// Training parameters
    pub training: TrainingConfig,
    /// Input encoding options
    pub input_encoding: InputEncodingConfig,
    /// Weight quantization (for digipot simulation)
    pub quantization: QuantizationConfig,
    /// Noise injection for robustness
    pub noise: NoiseConfig,
    /// Output mode selection
    pub output: OutputConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: OperationMode::Simple,
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

        let config: Config =
            serde_json::from_str(&contents).with_context(|| "Failed to parse config JSON")?;

        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents =
            serde_json::to_string_pretty(self).with_context(|| "Failed to serialize config")?;

        fs::write(path.as_ref(), contents)
            .with_context(|| format!("Failed to write config file: {:?}", path.as_ref()))?;

        Ok(())
    }

    /// Check if physics mode is enabled
    pub fn is_physics_mode(&self) -> bool {
        self.mode == OperationMode::Physics
    }
}

/// Network architecture configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NetworkConfig {
    /// Input layer size (default: 36 for 6x6 MNIST)
    pub input_size: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Output layer size (default: 10 for MNIST digits)
    pub output_size: usize,
    /// Image size for square MNIST downsampling (n×n, produces n² input features)
    /// Default: 6 (produces 36 input features). Ignored if image_width/height set.
    pub image_size: usize,
    /// Image width for non-square images (overrides image_size if set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_width: Option<usize>,
    /// Image height for non-square images (overrides image_size if set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_height: Option<usize>,
}

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
            input_size: 36,
            hidden_size: 12,
            output_size: 10,
            image_size: 6,
            image_width: None,
            image_height: None,
        }
    }
}

/// Neuron parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NeuronConfig {
    /// Membrane decay rate (0 to 1)
    pub beta: f32,
    /// Spike threshold
    pub threshold: f32,
    /// Reset mechanism after spike
    pub reset_mechanism: ResetMechanism,
    /// Surrogate gradient slope
    pub slope: f32,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            beta: 0.9,
            threshold: 1.0,
            reset_mechanism: ResetMechanism::Subtract,
            slope: 25.0,
        }
    }
}

/// Physics simulation parameters (gilgamesh-derived defaults)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysicsConfig {
    /// Enable physics mode
    pub enabled: bool,
    /// Membrane time constant (seconds) — ~3.96ms (33nF × 120kΩ)
    pub tau_m: f32,
    /// Pulse stretching time constant (seconds) — ~1.5µs (R_pw × C_pw)
    pub tau_pulse: f32,
    /// Threshold adaptation time constant (seconds) — ~1ms
    pub tau_theta: f32,
    /// Integration timestep (seconds) — 1ms
    pub dt: f32,
    /// Enable threshold adaptation
    pub adaptation_enabled: bool,
    /// Low threshold value (resting state)
    pub theta_low: f32,
    /// High threshold value (after spike, 20% increase)
    pub theta_high: f32,
    /// Spike scale: fraction of timestep the inter-layer spike pulse lasts.
    /// Models hardware pulse stretcher duration relative to integration step.
    /// Default 1.0 = spike lasts full timestep (original behavior).
    /// For hardware: spike_scale = t_pulse_effective / dt.
    /// Tarski PCB with τ_pulse=1.5µs, dt=1ms: spike_scale ≈ 0.00086.
    #[serde(default = "default_spike_scale")]
    pub spike_scale: f32,
    /// Maximum fc1 output the DAC can represent (in gilgamesh normalized units).
    /// On hardware: V_DAC ranges from V_BE to V_DD. The max fc1 value that maps
    /// to V_DD is (V_DD - V_BE) / (θ_norm × R_set_input / R_leak).
    /// For Tarski PCB: (5.0 - 0.65) / (0.387 × 174k / 120k) = 7.76.
    /// Set to null/omit to disable clamping (default: no clamp).
    #[serde(default)]
    pub dac_max: Option<f32>,
}

fn default_spike_scale() -> f32 {
    1.0 // Default: spike lasts full timestep (backward compatible)
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // Updated 2026-03-24 to match final InputSystem PCB schematic
            // C_mem=10nF, R_leak=120kΩ → τ_m = 1.2ms (was 3.96ms with 33nF)
            tau_m: 0.0012,
            tau_pulse: 1.5e-6,   // R_stretch=150kΩ × C_stretch=10pF
            tau_theta: 0.000596, // C_thresh=4.7nF / G_theta (was 0.001)
            dt: 0.001,
            adaptation_enabled: false,
            theta_low: 1.0,
            theta_high: 1.2,
            spike_scale: 1.0,
            dac_max: None,
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
#[serde(default)]
pub struct HardwareConfig {
    /// Supply voltage rail
    pub v_rail: f32,
    /// Comparator output rail drop
    pub comp_rail_drop: f32,
    /// Diode forward voltage drop
    pub diode_drop: f32,
    /// Minimum membrane voltage
    pub v_min: f32,
    /// Maximum membrane voltage
    pub v_max: f32,
    /// Enable current-limit calibration in training/inference.
    /// When enabled, synapse gains are scaled from a 3uA baseline and
    /// layer outputs can be capped by total-current budget.
    pub enable_current_caps: bool,
    /// Per-synapse current cap in microamps (used as relative scale vs 3uA baseline).
    pub synapse_current_max_ua: f32,
    /// Per-neuron total current cap in microamps.
    pub total_current_max_ua: f32,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            v_rail: 5.0,
            comp_rail_drop: 0.21,
            diode_drop: 0.37,
            v_min: 0.0,
            v_max: 5.0,
            enable_current_caps: false,
            synapse_current_max_ua: 3.0,
            total_current_max_ua: 50.0,
        }
    }
}

impl HardwareConfig {
    /// Compute effective pulse peak voltage
    pub fn pulse_peak(&self) -> f32 {
        self.v_rail - self.comp_rail_drop - self.diode_drop
    }

    /// Scale factor relative to a 3uA per-synapse baseline.
    pub fn synapse_scale_from_baseline(&self) -> f32 {
        const BASELINE_SYN_UA: f32 = 3.0;
        (self.synapse_current_max_ua / BASELINE_SYN_UA).max(0.0)
    }

    /// Total-current cap expressed in "equivalent synapse units".
    pub fn total_cap_units(&self) -> Option<f32> {
        if !self.enable_current_caps {
            return None;
        }
        let denom = self.synapse_current_max_ua;
        if denom <= 0.0 {
            return None;
        }
        Some((self.total_current_max_ua / denom).max(0.0))
    }
}

/// Training configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// Learning rate
    pub lr: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Number of timesteps per sample
    pub num_steps: usize,
    /// Random seed
    pub seed: u64,
    /// Number of parallel workers (0 = auto)
    pub num_workers: usize,
    /// Truncated BPTT steps (None/0 = full BPTT)
    pub bptt_steps: Option<usize>,
    /// AdamW weight decay
    pub weight_decay: f32,
    /// Max gradient norm for clipping
    pub max_grad_norm: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            epochs: 15,
            batch_size: 128,
            num_steps: 25,
            seed: 42,
            num_workers: 0,
            bptt_steps: None,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

/// Input encoding type
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingType {
    /// All pixels presented simultaneously each timestep
    RateCoded,
    /// Rows presented sequentially over time
    Temporal,
    /// Pixels generate spike trains at rates proportional to intensity
    /// Uses deterministic accumulator: uniform spiking hardware throughout
    Spiking,
}

/// Input encoding configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct InputEncodingConfig {
    /// Encoding type
    pub encoding_type: EncodingType,
    /// Row spacing for temporal encoding (seconds)
    pub row_spacing: f32,
    /// Pulse width as fraction of row spacing
    pub pulse_width: f32,
}

impl Default for InputEncodingConfig {
    fn default() -> Self {
        Self {
            encoding_type: EncodingType::RateCoded,
            row_spacing: 0.0015,
            pulse_width: 0.9,
        }
    }
}

/// Weight quantization configuration (for digipot simulation)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantizationConfig {
    /// Enable weight quantization
    pub enabled: bool,
    /// Number of magnitude bits (3 for current sources: 0-7 range)
    pub bits: u8,
    /// Apply quantization during training (QAT)
    pub apply_during_training: bool,
    /// Use symmetric quantization around 0
    pub symmetric: bool,
    /// Use split-sign quantization: independent current scaling for positive
    /// and negative weights. Models hardware with separate excitatory/inhibitory
    /// current sources where each uses the full magnitude range independently.
    pub split_sign: bool,
    /// Input quantization bits (0 = no input quantization).
    /// Models DAC resolution for input pixel values.
    pub input_bits: u8,
    /// Fixed quantization scale for fc2 weights (hardware-matched).
    /// When > 0, overrides the adaptive scale (max_weight/max_magnitude)
    /// with this fixed value. Ensures integer weights map to exact hardware
    /// current levels.
    ///
    /// For Tarski PCB: 0.1349 (derived from I_unit × duty × R_leak / (θ_hw × spike_scale))
    #[serde(default)]
    pub fixed_fc2_scale: f32,
    /// Hardware defect model: inhibitory 1x branch is broken (Q4).
    /// When true, negative quantized magnitudes are forced to even values
    /// so the dead 1x branch is never used.
    #[serde(default)]
    pub broken_inhibitory_lsb: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bits: 3,
            apply_during_training: true,
            symmetric: true,
            split_sign: false,
            input_bits: 0,
            fixed_fc2_scale: 0.0, // 0 = adaptive (default/original behavior)
            broken_inhibitory_lsb: false,
        }
    }
}

impl QuantizationConfig {
    /// Quantize a single weight value (legacy per-scalar path).
    ///
    /// This is the original, simple symmetric/asymmetric scalar quantizer and is
    /// **not** on the active forward/training path. Production quantization uses
    /// the split-sign matrix quantizer in
    /// [`crate::layers::linear::quantize_weights`] (driven by the `split_sign`,
    /// `fixed_fc2_scale`, and `broken_inhibitory_lsb` fields), which models the
    /// hardware's independent excitatory/inhibitory current scales. This method
    /// is retained only for the characterization tests that pin its exact
    /// rounding behavior; prefer `quantize_weights` for any new code.
    pub fn quantize(&self, weight: f32) -> f32 {
        if !self.enabled {
            return weight;
        }

        let quantization_levels = (1u32 << self.bits) as f32;

        if self.symmetric {
            // Symmetric quantization: [-max, +max] mapped to levels
            let max_magnitude = weight.abs();
            if max_magnitude < 1e-8 {
                return 0.0;
            }
            let quantization_scale = (quantization_levels / 2.0) / max_magnitude;
            ((weight * quantization_scale).round() / quantization_scale)
                .clamp(-max_magnitude, max_magnitude)
        } else {
            // Asymmetric: just round to nearest level
            (weight * quantization_levels).round() / quantization_levels
        }
    }
}

/// Noise injection configuration for robustness training
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseConfig {
    /// Enable noise injection
    pub enabled: bool,
    /// Only apply noise during training
    pub training_only: bool,
    /// Weight noise std (relative to weight magnitude)
    pub weight_std: f32,
    /// Threshold noise std (relative to threshold)
    pub threshold_std: f32,
    /// Membrane noise std (absolute)
    pub membrane_std: f32,
    /// Input noise std (relative to input)
    pub input_std: f32,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            training_only: false,
            weight_std: 0.05,
            threshold_std: 0.02,
            membrane_std: 0.01,
            input_std: 0.1,
        }
    }
}

/// Output mode type for configuration
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputModeType {
    /// Sum spikes over time (standard snnTorch)
    SpikeCount,
    /// Use final membrane voltage
    AnalogFinal,
    /// Use max membrane voltage over time
    AnalogMax,
    /// Use filtered membrane (low-pass)
    AnalogFiltered,
}

/// Output mode configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Output mode
    pub mode: OutputModeType,
    /// Filter time constant for analog_filtered mode
    pub filter_tau: f32,
    /// Analog gain for hybrid spike+membrane inter-layer transmission (0.0 = disabled)
    pub analog_gain: f32,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            mode: OutputModeType::SpikeCount,
            filter_tau: 0.002,
            analog_gain: 0.0,
        }
    }
}

/// Runtime output mode with associated data
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
    /// Convert config into runtime OutputMode
    pub fn to_mode(&self) -> OutputMode {
        match self.mode {
            OutputModeType::AnalogFinal => OutputMode::AnalogFinal,
            OutputModeType::AnalogMax => OutputMode::AnalogMax,
            OutputModeType::AnalogFiltered => OutputMode::AnalogFiltered {
                tau_filter: self.filter_tau,
            },
            OutputModeType::SpikeCount => OutputMode::SpikeCount,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.mode, OperationMode::Simple);
        assert_eq!(config.network.input_size, 36);
        assert_eq!(config.neuron.beta, 0.9);
        assert!(!config.quantization.enabled);
        assert!(config.noise.enabled);
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

        config.mode = OutputModeType::AnalogFinal;
        assert_eq!(config.to_mode(), OutputMode::AnalogFinal);

        config.mode = OutputModeType::AnalogFiltered;
        config.filter_tau = 0.005;
        match config.to_mode() {
            OutputMode::AnalogFiltered { tau_filter } => {
                assert!((tau_filter - 0.005).abs() < 1e-6);
            }
            _ => panic!("Expected AnalogFiltered"),
        }
    }
}
