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

/// Quantize a single weight value to n-bit resolution
///
/// Uses symmetric quantization around 0, mapping [-max, +max] to n-bit levels.
/// For 8-bit: 127 positive levels, 127 negative levels, 1 zero (255 total values).
#[inline]
pub fn quantize_weight(w: f32, bits: u8, w_max: f32) -> f32 {
    if w_max < 1e-8 || bits < 2 {
        return 0.0;
    }
    // Clamp bits to reasonable range
    let bits = bits.min(16);
    // Use 2^(bits-1) - 1 for proper symmetric signed quantization
    let levels = ((1u32 << (bits - 1)) - 1) as f32;
    let scale = levels / w_max;
    let q = (w * scale).round() / scale;
    q.clamp(-w_max, w_max)
}

/// Quantize a weight array in-place (for efficiency)
pub fn quantize_weights(weights: &Array2<f32>, bits: u8) -> Array2<f32> {
    // Find max absolute weight for scaling
    let w_max = weights.iter().fold(0.0f32, |acc, &w| acc.max(w.abs()));

    weights.mapv(|w| quantize_weight(w, bits, w_max))
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
        }
    }

    /// Forward pass: y = x @ W + b
    ///
    /// Args:
    ///   input: [batch, in_features]
    ///
    /// Returns:
    ///   output: [batch, out_features]
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.dot(&self.weight);
        if let Some(ref b) = self.bias {
            // Broadcast bias across batch dimension
            for mut row in output.rows_mut() {
                row += b;
            }
        }
        output
    }

    /// Forward pass with weight quantization (for hardware simulation)
    ///
    /// Quantizes weights to n-bit resolution before computing output.
    /// Uses straight-through estimator: quantized forward, full-precision backward.
    ///
    /// Args:
    ///   input: [batch, in_features]
    ///   bits: Number of quantization bits (e.g., 8 for digipot)
    ///
    /// Returns:
    ///   output: [batch, out_features]
    pub fn forward_quantized(&self, input: &Array2<f32>, bits: u8) -> Array2<f32> {
        let quantized_weight = quantize_weights(&self.weight, bits);
        let mut output = input.dot(&quantized_weight);
        if let Some(ref b) = self.bias {
            for mut row in output.rows_mut() {
                row += b;
            }
        }
        output
    }

    /// Forward pass with weight noise injection (for robustness training)
    ///
    /// Adds Gaussian noise to weights before computing output.
    /// Noise is relative to weight magnitude: noisy_w = w + w * N(0, noise_std)
    ///
    /// Args:
    ///   input: [batch, in_features]
    ///   weight_noise_std: Relative noise std (e.g., 0.05 for 5% mismatch)
    ///   rng: Random number generator
    ///
    /// Returns:
    ///   output: [batch, out_features]
    pub fn forward_noisy<R: Rng>(
        &self,
        input: &Array2<f32>,
        weight_noise_std: f32,
        rng: &mut R,
    ) -> Array2<f32> {
        // Add relative noise to weights
        let noisy_weight = if weight_noise_std > 0.0 {
            let normal = Normal::new(0.0, weight_noise_std as f64).unwrap();
            self.weight.mapv(|w| {
                let noise = normal.sample(rng) as f32;
                w * (1.0 + noise)
            })
        } else {
            self.weight.clone()
        };

        let mut output = input.dot(&noisy_weight);
        if let Some(ref b) = self.bias {
            for mut row in output.rows_mut() {
                row += b;
            }
        }
        output
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
        if let (Some(ref mut b), Some(gb)) = (&mut self.bias, grad_bias) {
            *b = &*b - &(gb * lr);
        }
    }

    /// Zero out accumulated gradients (for batch processing)
    pub fn zero_grad(&mut self) {
        // No-op since we don't store gradients in the layer
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
        let output = layer.forward(&input);
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
    fn test_quantize_weight() {
        // Test 8-bit quantization
        let w_max = 1.0;

        // Zero should stay zero
        assert_eq!(quantize_weight(0.0, 8, w_max), 0.0);

        // Max should stay max
        assert!((quantize_weight(1.0, 8, w_max) - 1.0).abs() < 0.01);

        // Values should be rounded to discrete levels
        let q = quantize_weight(0.5, 8, w_max);
        assert!(q.is_finite());
    }

    #[test]
    fn test_quantize_weights_array() {
        let weights = array![[0.1, 0.5, -0.3], [0.8, -0.2, 0.4]];
        let quantized = quantize_weights(&weights, 8);

        // Shape should be preserved
        assert_eq!(quantized.shape(), weights.shape());

        // Values should be similar (8-bit has good precision)
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
}
