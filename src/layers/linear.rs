//! Linear (fully connected) layer
//!
//! Implements a standard linear transformation: y = x @ W + b
//! Matching PyTorch's nn.Linear behavior.
//!
//! Supports optional weight quantization for hardware simulation (digipots).

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Default sign-aware synapse gains from the latest 9x mirror bench calibration.
/// These are cheap static gains that preserve the 3-bit+sign structure while
/// matching measured mirror-vs-ideal current ratios more closely than a single scalar.
pub const DEFAULT_SYNAPSE_POS_GAIN: f32 = 1.07;
pub const DEFAULT_SYNAPSE_NEG_GAIN: f32 = 1.06;

/// Quantize weights with split-sign scaling (hardware-accurate)
///
/// Hardware model: each synapse has n-bit magnitude + 1-bit sign, with
/// separate analog current scales for excitatory vs inhibitory sources.
/// Positive and negative weights each use the full magnitude range independently.
///
/// For 3-bit: magnitude 0-7, so max_magnitude = 7.
/// Positive weight = magnitude × pos_scale, Negative weight = -magnitude × neg_scale.
///
/// This matches the checkpoint export format in checkpoint.rs and the actual hardware.
pub fn quantize_weights(weights: &Array2<f32>, bits: u8) -> Array2<f32> {
    quantize_weights_with_fixed_scale_and_defect(weights, bits, None, None, false)
}

/// Quantize with optional fixed scales (for hardware-matched training).
///
/// When fixed_pos_scale / fixed_neg_scale are None, the scale is derived from
/// the max weight (original adaptive behavior).
///
/// When set, the scale is fixed to match hardware current levels. This ensures
/// the training optimizer learns integer weights that produce exactly the right
/// current on the physical circuit.
///
/// For the Tarski PCB with spike_scale=0.5:
///   fixed_pos_scale = fixed_neg_scale ≈ 0.1349
///   (derived from: I_unit × duty × R_leak / (θ_hw × spike_scale))
pub fn quantize_weights_with_fixed_scale(
    weights: &Array2<f32>,
    bits: u8,
    fixed_pos_scale: Option<f32>,
    fixed_neg_scale: Option<f32>,
) -> Array2<f32> {
    quantize_weights_with_fixed_scale_and_defect(
        weights,
        bits,
        fixed_pos_scale,
        fixed_neg_scale,
        false,
    )
}

/// Quantize with optional fixed scales plus optional inhibitory-LSB defect model.
///
/// When `disable_inhibitory_lsb` is true, negative magnitudes are restricted to
/// even values only (0,2,4,...) to model a broken inhibitory 1x branch.
pub fn quantize_weights_with_fixed_scale_and_defect(
    weights: &Array2<f32>,
    bits: u8,
    fixed_pos_scale: Option<f32>,
    fixed_neg_scale: Option<f32>,
    disable_inhibitory_lsb: bool,
) -> Array2<f32> {
    let max_magnitude = ((1i32 << bits) - 1) as f32;

    let pos_scale = match fixed_pos_scale {
        Some(s) => s,
        None => {
            let max_pos = weights.iter().fold(0.0f32, |m, &w| m.max(w.max(0.0)));
            if max_pos > 1e-8 {
                max_pos / max_magnitude
            } else {
                1.0
            }
        }
    };
    let neg_scale = match fixed_neg_scale {
        Some(s) => s,
        None => {
            let max_neg = weights.iter().fold(0.0f32, |m, &w| m.max((-w).max(0.0)));
            if max_neg > 1e-8 {
                max_neg / max_magnitude
            } else {
                1.0
            }
        }
    };

    weights.mapv(|w| {
        if w >= 0.0 {
            let q = (w / pos_scale).round().clamp(0.0, max_magnitude);
            q * pos_scale
        } else {
            let mut q = ((-w) / neg_scale).round().clamp(0.0, max_magnitude) as i32;
            if disable_inhibitory_lsb {
                q &= !1; // Force odd magnitudes off: 1x inhibitory transistor is broken.
            }
            -((q as f32) * neg_scale)
        }
    })
}

/// Quantize input values to n-bit unsigned resolution (for DAC simulation)
pub fn quantize_input(input: &Array2<f32>, bits: u8) -> Array2<f32> {
    let levels = ((1u32 << bits) - 1) as f32;
    input.mapv(|x| (x * levels).round() / levels)
}

/// Linear (fully connected) layer
#[derive(Clone, Debug)]
pub struct Linear {
    /// Weight matrix [in_features, out_features]
    pub weight: Array2<f32>,
    /// Bias vector [out_features] (optional)
    pub bias: Option<Array1<f32>>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
    /// Current gain for hardware simulation (converts normalized output to physical current)
    /// When set, output is scaled by this factor to produce physical current values.
    /// Use HardwareConfig::compute_current_gain() to get appropriate value.
    pub current_gain: Option<f32>,
    /// Sign-aware synapse gains for weighted-sum paths (applied before bias).
    pub synapse_pos_gain: f32,
    pub synapse_neg_gain: f32,
    /// Optional absolute cap on total output current per neuron (post-bias/gain), in model units.
    pub total_current_cap: Option<f32>,
    /// Fixed quantization scale (for hardware-matched QAT).
    /// When > 0, quantize_weights uses this as pos_scale and neg_scale
    /// instead of deriving from max weight. Set to 0 for adaptive (default).
    pub fixed_quant_scale: f32,
    /// Hardware defect model: disable inhibitory magnitude LSB (1x branch).
    /// When true, negative quantized magnitudes are forced to even values.
    pub disable_inhibitory_lsb: bool,
}

impl Linear {
    /// Create a new linear layer with Kaiming initialization
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self::with_seed(in_features, out_features, bias, 42)
    }

    /// Create with specific random seed for reproducibility
    pub fn with_seed(in_features: usize, out_features: usize, bias: bool, seed: u64) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        // Kaiming uniform initialization (same as PyTorch default for Linear)
        // bound = sqrt(1 / in_features)
        let bound = (1.0 / in_features as f32).sqrt();
        let weight = Array2::random_using(
            (in_features, out_features),
            Uniform::new(-bound, bound),
            &mut rng,
        );

        let bias_vec = if bias {
            Some(Array1::random_using(
                out_features,
                Uniform::new(-bound, bound),
                &mut rng,
            ))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_vec,
            in_features,
            out_features,
            current_gain: None,
            synapse_pos_gain: DEFAULT_SYNAPSE_POS_GAIN,
            synapse_neg_gain: DEFAULT_SYNAPSE_NEG_GAIN,
            total_current_cap: None,
            fixed_quant_scale: 0.0,
            disable_inhibitory_lsb: false,
        }
    }

    /// Set current gain for hardware simulation
    ///
    /// When set, the forward pass output is scaled by this factor to convert
    /// normalized weight outputs to physical synaptic currents.
    ///
    /// # Arguments
    /// * `gain` - Current gain in Amps (use HardwareConfig::compute_current_gain())
    ///
    /// # Example
    /// ```ignore
    /// let hw = HardwareConfig::default();
    /// let fc = Linear::new(100, 50, true)
    ///     .with_current_gain(hw.compute_current_gain());
    /// ```
    pub fn with_current_gain(mut self, gain: f32) -> Self {
        self.current_gain = Some(gain);
        self
    }

    /// Clear current gain (return to normalized output)
    pub fn without_current_gain(mut self) -> Self {
        self.current_gain = None;
        self
    }

    /// Set sign-aware synapse gains for weighted-sum paths.
    pub fn with_synapse_gains(mut self, pos_gain: f32, neg_gain: f32) -> Self {
        self.synapse_pos_gain = pos_gain.max(0.0);
        self.synapse_neg_gain = neg_gain.max(0.0);
        self
    }

    /// Backward-compatible helper: set a single synapse gain for both signs.
    pub fn with_synapse_efficiency(mut self, efficiency: f32) -> Self {
        let gain = efficiency.max(0.0);
        self.synapse_pos_gain = gain;
        self.synapse_neg_gain = gain;
        self
    }

    /// Set optional cap on total output current per neuron (post-bias/gain).
    pub fn with_total_current_cap(mut self, cap: Option<f32>) -> Self {
        self.total_current_cap = cap.filter(|v| *v > 0.0);
        self
    }

    /// Enable/disable broken inhibitory 1x branch modeling during quantization.
    pub fn with_broken_inhibitory_lsb(mut self, broken: bool) -> Self {
        self.disable_inhibitory_lsb = broken;
        self
    }

    /// Quantized weight matrix using this layer's quantization settings.
    pub fn quantized_weight_matrix(&self, bits: u8) -> Array2<f32> {
        let fixed_scale = if self.fixed_quant_scale > 0.0 {
            Some(self.fixed_quant_scale)
        } else {
            None
        };
        quantize_weights_with_fixed_scale_and_defect(
            &self.weight,
            bits,
            fixed_scale,
            fixed_scale,
            self.disable_inhibitory_lsb,
        )
    }

    /// Gain to apply for a single synaptic branch weight sign.
    pub fn synapse_gain_for_weight(&self, weight: f32) -> f32 {
        if weight >= 0.0 {
            self.synapse_pos_gain
        } else {
            self.synapse_neg_gain
        }
    }

    /// Apply sign-aware synapse gain to weighted-sum currents in place (before bias).
    pub fn apply_synapse_drive_model_inplace(&self, output: &mut Array2<f32>) {
        if (self.synapse_pos_gain - 1.0).abs() <= f32::EPSILON
            && (self.synapse_neg_gain - 1.0).abs() <= f32::EPSILON
        {
            return;
        }
        let gp = self.synapse_pos_gain;
        let gn = self.synapse_neg_gain;
        output.mapv_inplace(|v| if v >= 0.0 { v * gp } else { v * gn });
    }

    /// Apply synapse gains, bias addition, and current gain scaling to a raw dot product.
    /// Synapse gains are applied to the weighted sum only (bias path is separate).
    fn apply_bias_and_gain(&self, mut output: Array2<f32>) -> Array2<f32> {
        self.apply_synapse_drive_model_inplace(&mut output);
        if let Some(ref bias) = self.bias {
            for mut row in output.rows_mut() {
                row += bias;
            }
        }
        if let Some(gain) = self.current_gain {
            output *= gain;
        }
        if let Some(cap) = self.total_current_cap {
            output.mapv_inplace(|v| v.clamp(-cap, cap));
        }
        output
    }

    /// Forward pass: y = x @ W + b
    ///
    /// If current_gain is set, output is scaled: y = (x @ W + b) * gain
    /// This converts normalized weights to physical synaptic currents.
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        self.apply_bias_and_gain(input.dot(&self.weight))
    }

    /// Forward pass with weight quantization (for hardware simulation)
    ///
    /// Quantizes weights to n-bit resolution before computing output.
    /// Uses straight-through estimator: quantized forward, full-precision backward.
    pub fn forward_quantized(&self, input: &Array2<f32>, bits: u8) -> Array2<f32> {
        self.apply_bias_and_gain(input.dot(&self.quantized_weight_matrix(bits)))
    }

    /// Forward pass with weight noise injection (for robustness training)
    ///
    /// Adds Gaussian noise to weights before computing output.
    /// Noise is relative to weight magnitude: noisy_w = w + w * N(0, noise_std)
    pub fn forward_noisy<R: Rng>(
        &self,
        input: &Array2<f32>,
        weight_noise_std: f32,
        rng: &mut R,
    ) -> Array2<f32> {
        let effective_weight = if weight_noise_std > 0.0 {
            let normal = Normal::new(0.0, weight_noise_std as f64).unwrap();
            self.weight.mapv(|weight| {
                let noise = normal.sample(rng) as f32;
                weight * (1.0 + noise)
            })
        } else {
            self.weight.clone()
        };
        self.apply_bias_and_gain(input.dot(&effective_weight))
    }

    /// Backward pass
    ///
    /// Args:
    ///   input: Original input [batch, in_features]
    ///   grad_output: Gradient w.r.t. output [batch, out_features]
    ///
    /// Returns:
    ///   (grad_input, grad_weight, grad_bias)
    pub fn backward(
        &self,
        input: &Array2<f32>,
        grad_output: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Option<Array1<f32>>) {
        let gain = self.current_gain.unwrap_or(1.0);

        // y = gain * (f(x @ W) + b), where f is sign-aware gain mapping.
        // grad wrt (f(x@W)+b) includes gain.
        let mut grad_pre_bias = grad_output.clone();
        if (gain - 1.0).abs() > f32::EPSILON {
            grad_pre_bias *= gain;
        }

        // grad wrt linear weighted sum includes sign-aware synapse gain.
        let mut grad_linear = grad_pre_bias.clone();
        if (self.synapse_pos_gain - 1.0).abs() > f32::EPSILON
            || (self.synapse_neg_gain - 1.0).abs() > f32::EPSILON
        {
            let pre_sum = input.dot(&self.weight);
            let gp = self.synapse_pos_gain;
            let gn = self.synapse_neg_gain;
            for (g, s) in grad_linear.iter_mut().zip(pre_sum.iter()) {
                *g *= if *s >= 0.0 { gp } else { gn };
            }
        }

        // grad_input = grad_linear @ W^T
        let grad_input = grad_linear.dot(&self.weight.t());

        // grad_weight = input^T @ grad_linear
        let grad_weight = input.t().dot(&grad_linear);

        // grad_bias = sum(grad_output, axis=0)
        let grad_bias = if self.bias.is_some() {
            Some(grad_pre_bias.sum_axis(ndarray::Axis(0)))
        } else {
            None
        };

        (grad_input, grad_weight, grad_bias)
    }

    /// Apply gradient update with learning rate
    pub fn apply_gradient(
        &mut self,
        grad_weight: &Array2<f32>,
        grad_bias: Option<&Array1<f32>>,
        lr: f32,
    ) {
        self.weight = &self.weight - &(grad_weight * lr);
        if let (Some(ref mut bias), Some(bias_grad)) = (&mut self.bias, grad_bias) {
            *bias = &*bias - &(bias_grad * lr);
        }
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = if self.bias.is_some() {
            self.out_features
        } else {
            0
        };
        weight_params + bias_params
    }
}

/// Cache for linear layer backward pass
#[derive(Clone, Debug)]
pub struct LinearCache {
    pub input: Array2<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_forward() {
        let layer = Linear::with_seed(3, 2, true, 42);

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let output = layer.forward(&input);

        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_linear_backward() {
        let layer = Linear::with_seed(3, 2, true, 42);

        let input = array![[1.0, 2.0, 3.0]];
        let _output = layer.forward(&input);
        let grad_output = array![[1.0, 1.0]];

        let (grad_input, grad_weight, grad_bias) = layer.backward(&input, &grad_output);

        assert_eq!(grad_input.shape(), &[1, 3]);
        assert_eq!(grad_weight.shape(), &[3, 2]);
        assert!(grad_bias.is_some());
        assert_eq!(grad_bias.unwrap().len(), 2);
    }

    #[test]
    fn test_linear_no_bias() {
        let layer = Linear::with_seed(2, 3, false, 42);
        assert!(layer.bias.is_none());

        let input = array![[1.0, 2.0]];
        let output = layer.forward(&input);
        assert_eq!(output.shape(), &[1, 3]);

        let grad_output = array![[1.0, 1.0, 1.0]];
        let (_, _, grad_bias) = layer.backward(&input, &grad_output);
        assert!(grad_bias.is_none());
    }

    #[test]
    fn test_parameter_count() {
        let layer_with_bias = Linear::new(10, 5, true);
        assert_eq!(layer_with_bias.num_parameters(), 10 * 5 + 5);

        let layer_no_bias = Linear::new(10, 5, false);
        assert_eq!(layer_no_bias.num_parameters(), 10 * 5);
    }

    #[test]
    fn test_quantize_weights_split_sign() {
        // Test 3-bit split-sign quantization (hardware model)
        let weights = array![[0.1, 0.5, -0.3], [0.8, -0.2, 0.4]];
        let quantized = quantize_weights(&weights, 3);

        // Shape should be preserved
        assert_eq!(quantized.shape(), weights.shape());

        // All values should be finite
        assert!(quantized.iter().all(|v| v.is_finite()));

        // Max positive should be preserved (maps to magnitude 7)
        let max_pos_orig = weights
            .iter()
            .filter(|&&w| w > 0.0)
            .fold(0.0f32, |a, &w| a.max(w));
        let max_pos_quant = quantized
            .iter()
            .filter(|&&w| w > 0.0)
            .fold(0.0f32, |a, &w| a.max(w));
        assert!(
            (max_pos_orig - max_pos_quant).abs() < 1e-6,
            "Max positive should be exact"
        );

        // Max negative should be preserved (maps to magnitude 7)
        let max_neg_orig = weights
            .iter()
            .filter(|&&w| w < 0.0)
            .fold(0.0f32, |a, &w| a.min(w));
        let max_neg_quant = quantized
            .iter()
            .filter(|&&w| w < 0.0)
            .fold(0.0f32, |a, &w| a.min(w));
        assert!(
            (max_neg_orig - max_neg_quant).abs() < 1e-6,
            "Max negative should be exact"
        );
    }

    #[test]
    fn test_quantize_weights_8bit() {
        let weights = array![[0.1, 0.5, -0.3], [0.8, -0.2, 0.4]];
        let quantized = quantize_weights(&weights, 8);

        // Shape should be preserved
        assert_eq!(quantized.shape(), weights.shape());

        // With 8-bit (255 levels per sign), values should be close
        for (orig, quant) in weights.iter().zip(quantized.iter()) {
            assert!(
                (orig - quant).abs() < 0.02,
                "orig={}, quant={}",
                orig,
                quant
            );
        }
    }

    #[test]
    fn test_quantize_weights_broken_inhibitory_lsb() {
        let weights = array![[-0.10, -0.20, -0.30, -0.70, 0.35]];
        let quantized =
            quantize_weights_with_fixed_scale_and_defect(&weights, 3, Some(0.10), Some(0.10), true);

        // Negative magnitudes should be even-only: {0,2,4,6} for 3-bit.
        assert!((quantized[[0, 0]] - 0.0).abs() < 1e-6); // 1 -> 0
        assert!((quantized[[0, 1]] + 0.2).abs() < 1e-6); // 2 -> 2
        assert!((quantized[[0, 2]] + 0.2).abs() < 1e-6); // 3 -> 2
        assert!((quantized[[0, 3]] + 0.6).abs() < 1e-6); // 7 -> 6
        assert!((quantized[[0, 4]] - 0.4).abs() < 1e-6); // positive path unchanged
    }

    #[test]
    fn test_forward_quantized() {
        let layer = Linear::with_seed(3, 2, true, 42);

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let output_normal = layer.forward(&input);
        let output_quantized = layer.forward_quantized(&input, 8);

        // Shapes should match
        assert_eq!(output_normal.shape(), output_quantized.shape());

        // With 8-bit quantization, outputs should be very similar
        for (normal, quant) in output_normal.iter().zip(output_quantized.iter()) {
            assert!(
                (normal - quant).abs() < 0.1,
                "normal={}, quant={}",
                normal,
                quant
            );
        }
    }

    #[test]
    fn test_low_bit_quantization() {
        let layer = Linear::with_seed(3, 2, true, 42);
        let input = array![[1.0, 2.0, 3.0]];

        // 4-bit quantization should still produce valid output
        let output_4bit = layer.forward_quantized(&input, 4);
        assert_eq!(output_4bit.shape(), &[1, 2]);
        assert!(output_4bit.iter().all(|v| v.is_finite()));

        // 2-bit quantization (very coarse) should still work
        let output_2bit = layer.forward_quantized(&input, 2);
        assert_eq!(output_2bit.shape(), &[1, 2]);
        assert!(output_2bit.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_current_gain_scaling() {
        let layer = Linear::with_seed(3, 2, true, 42);
        let input = array![[1.0, 2.0, 3.0]];

        // Output without gain
        let output_no_gain = layer.forward(&input);

        // Add current gain
        let gain = 1e-6; // 1µA per unit
        let layer_with_gain = layer.clone().with_current_gain(gain);
        let output_with_gain = layer_with_gain.forward(&input);

        // Output should be scaled by gain
        for (no_gain, with_gain) in output_no_gain.iter().zip(output_with_gain.iter()) {
            let expected = no_gain * gain;
            assert!(
                (with_gain - expected).abs() < 1e-10,
                "expected={}, got={}",
                expected,
                with_gain
            );
        }
    }

    #[test]
    fn test_current_gain_with_quantization() {
        let gain = 5e-6; // 5µA per unit
        let layer = Linear::with_seed(3, 2, true, 42).with_current_gain(gain);
        let input = array![[1.0, 2.0, 3.0]];

        // Both quantized and non-quantized should apply gain
        let output_normal = layer.forward(&input);
        let output_quantized = layer.forward_quantized(&input, 8);

        // Both outputs should be in µA range (very small)
        assert!(output_normal.iter().all(|&v| v.abs() < 1e-4));
        assert!(output_quantized.iter().all(|&v| v.abs() < 1e-4));

        // Shapes should match
        assert_eq!(output_normal.shape(), output_quantized.shape());
    }

    #[test]
    fn test_without_current_gain() {
        let gain = 1e-6;
        let layer = Linear::with_seed(3, 2, true, 42)
            .with_current_gain(gain)
            .without_current_gain();
        let input = array![[1.0, 2.0, 3.0]];

        // After removing gain, output should be normal scale
        let output = layer.forward(&input);

        // Should NOT be in µA range
        assert!(output.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn test_synapse_sign_gains_scaling() {
        let mut layer = Linear::with_seed(2, 2, false, 42).with_synapse_gains(1.1, 0.9);
        layer.weight = array![[1.0, -1.0], [0.0, 0.0]];
        let input = array![[1.0, 1.0]];
        let out = layer.forward(&input);
        assert!((out[[0, 0]] - 1.1).abs() < 1e-6);
        assert!((out[[0, 1]] + 0.9).abs() < 1e-6);
    }
}
