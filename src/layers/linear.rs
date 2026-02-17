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
    let max_magnitude = ((1i32 << bits) - 1) as f32;

    // Find max separately for positive and negative weights
    let (max_pos, max_neg) = weights.iter().fold((0.0f32, 0.0f32), |(pos, neg), &w| {
        (pos.max(w.max(0.0)), neg.max((-w).max(0.0)))
    });

    let pos_scale = if max_pos > 1e-8 { max_pos / max_magnitude } else { 1.0 };
    let neg_scale = if max_neg > 1e-8 { max_neg / max_magnitude } else { 1.0 };

    weights.mapv(|w| {
        if w >= 0.0 {
            let q = (w / pos_scale).round().clamp(0.0, max_magnitude);
            q * pos_scale
        } else {
            let q = ((-w) / neg_scale).round().clamp(0.0, max_magnitude);
            -(q * neg_scale)
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

    /// Apply bias addition and current gain scaling to a raw dot product
    fn apply_bias_and_gain(&self, mut output: Array2<f32>) -> Array2<f32> {
        if let Some(ref bias) = self.bias {
            for mut row in output.rows_mut() {
                row += bias;
            }
        }
        if let Some(gain) = self.current_gain {
            output *= gain;
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
        self.apply_bias_and_gain(input.dot(&quantize_weights(&self.weight, bits)))
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
        // grad_input = grad_output @ W^T
        let grad_input = grad_output.dot(&self.weight.t());

        // grad_weight = input^T @ grad_output
        let grad_weight = input.t().dot(grad_output);

        // grad_bias = sum(grad_output, axis=0)
        let grad_bias = if self.bias.is_some() {
            Some(grad_output.sum_axis(ndarray::Axis(0)))
        } else {
            None
        };

        (grad_input, grad_weight, grad_bias)
    }

    /// Apply gradient update with learning rate
    pub fn apply_gradient(&mut self, grad_weight: &Array2<f32>, grad_bias: Option<&Array1<f32>>, lr: f32) {
        self.weight = &self.weight - &(grad_weight * lr);
        if let (Some(ref mut bias), Some(bias_grad)) = (&mut self.bias, grad_bias) {
            *bias = &*bias - &(bias_grad * lr);
        }
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = if self.bias.is_some() { self.out_features } else { 0 };
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
        let max_pos_orig = weights.iter().filter(|&&w| w > 0.0).fold(0.0f32, |a, &w| a.max(w));
        let max_pos_quant = quantized.iter().filter(|&&w| w > 0.0).fold(0.0f32, |a, &w| a.max(w));
        assert!((max_pos_orig - max_pos_quant).abs() < 1e-6, "Max positive should be exact");

        // Max negative should be preserved (maps to magnitude 7)
        let max_neg_orig = weights.iter().filter(|&&w| w < 0.0).fold(0.0f32, |a, &w| a.min(w));
        let max_neg_quant = quantized.iter().filter(|&&w| w < 0.0).fold(0.0f32, |a, &w| a.min(w));
        assert!((max_neg_orig - max_neg_quant).abs() < 1e-6, "Max negative should be exact");
    }

    #[test]
    fn test_quantize_weights_8bit() {
        let weights = array![[0.1, 0.5, -0.3], [0.8, -0.2, 0.4]];
        let quantized = quantize_weights(&weights, 8);

        // Shape should be preserved
        assert_eq!(quantized.shape(), weights.shape());

        // With 8-bit (255 levels per sign), values should be close
        for (orig, quant) in weights.iter().zip(quantized.iter()) {
            assert!((orig - quant).abs() < 0.02, "orig={}, quant={}", orig, quant);
        }
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
            assert!((normal - quant).abs() < 0.1, "normal={}, quant={}", normal, quant);
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
}
