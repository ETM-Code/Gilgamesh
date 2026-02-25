//! Surrogate gradient functions for spiking neural networks
//!
//! These functions provide differentiable approximations to the Heaviside step function
//! used in spike generation. The forward pass uses the true Heaviside function,
//! while the backward pass uses a smooth surrogate gradient.
//!
//! Matches snnTorch's surrogate gradient implementations exactly.

use serde::{Deserialize, Serialize};

/// Surrogate gradient function type
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SurrogateGradient {
    /// Fast sigmoid surrogate (SuperSpike)
    /// grad = 1 / (slope * |x| + 1)^2
    FastSigmoid { slope: f32 },

    /// Arctangent surrogate
    /// grad = alpha / 2 / (1 + (pi/2 * alpha * x)^2)
    ATan { alpha: f32 },

    /// Sigmoid surrogate
    /// grad = slope * exp(-slope * x) / (exp(-slope * x) + 1)^2
    Sigmoid { slope: f32 },

    /// Straight-through estimator
    /// grad = 1
    StraightThrough,

    /// Triangular surrogate
    Triangular { threshold: f32 },
}

impl Default for SurrogateGradient {
    fn default() -> Self {
        // snnTorch defaults to ATan with alpha=2.0
        SurrogateGradient::ATan { alpha: 2.0 }
    }
}

impl SurrogateGradient {
    /// Create FastSigmoid surrogate with default slope=25
    pub fn fast_sigmoid(slope: f32) -> Self {
        SurrogateGradient::FastSigmoid { slope }
    }

    /// Create ATan surrogate with default alpha=2.0
    pub fn atan(alpha: f32) -> Self {
        SurrogateGradient::ATan { alpha }
    }

    /// Create Sigmoid surrogate with default slope=25
    pub fn sigmoid(slope: f32) -> Self {
        SurrogateGradient::Sigmoid { slope }
    }

    /// Forward pass: Heaviside step function
    /// Returns 1.0 if shifted_potential > 0, else 0.0
    #[inline]
    pub fn forward(&self, shifted_potential: f32) -> f32 {
        if shifted_potential > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    /// Backward pass: Surrogate gradient
    /// shifted_potential is (membrane - threshold), i.e., the input to the Heaviside function
    #[inline]
    pub fn backward(&self, shifted_potential: f32) -> f32 {
        match self {
            SurrogateGradient::FastSigmoid { slope } => {
                // grad = 1 / (slope * |shifted_potential| + 1)^2
                let denom = slope * shifted_potential.abs() + 1.0;
                1.0 / (denom * denom)
            }
            SurrogateGradient::ATan { alpha } => {
                // grad = alpha / 2 / (1 + (pi/2 * alpha * shifted_potential)^2)
                let pi_half = std::f32::consts::FRAC_PI_2;
                let scaled = pi_half * alpha * shifted_potential;
                alpha / 2.0 / (1.0 + scaled * scaled)
            }
            SurrogateGradient::Sigmoid { slope } => {
                // grad = slope * exp(-slope * shifted_potential) / (exp(-slope * shifted_potential) + 1)^2
                let exp_neg = (-slope * shifted_potential).exp();
                let denom = exp_neg + 1.0;
                slope * exp_neg / (denom * denom)
            }
            SurrogateGradient::StraightThrough => 1.0,
            SurrogateGradient::Triangular { threshold } => {
                if shifted_potential < 0.0 {
                    *threshold
                } else {
                    -threshold
                }
            }
        }
    }

    /// Vectorized forward pass
    pub fn forward_batch(&self, shifted_potentials: &[f32]) -> Vec<f32> {
        shifted_potentials
            .iter()
            .map(|&v| self.forward(v))
            .collect()
    }

    /// Vectorized backward pass
    pub fn backward_batch(&self, shifted_potentials: &[f32]) -> Vec<f32> {
        shifted_potentials
            .iter()
            .map(|&v| self.backward(v))
            .collect()
    }

    /// Get the slope/sharpness parameter for this surrogate gradient
    pub fn slope(&self) -> f32 {
        match self {
            SurrogateGradient::FastSigmoid { slope } => *slope,
            SurrogateGradient::ATan { alpha } => *alpha,
            SurrogateGradient::Sigmoid { slope } => *slope,
            SurrogateGradient::StraightThrough => 1.0,
            SurrogateGradient::Triangular { threshold } => *threshold,
        }
    }
}

/// Spike function with surrogate gradient
/// Forward: Heaviside(mem - threshold)
/// Backward: surrogate_grad(mem - threshold)
pub struct SpikeFunction {
    pub surrogate: SurrogateGradient,
    pub threshold: f32,
}

impl SpikeFunction {
    pub fn new(surrogate: SurrogateGradient, threshold: f32) -> Self {
        Self {
            surrogate,
            threshold,
        }
    }

    /// Generate spike and compute gradient
    /// Returns (spike, gradient_scale)
    #[inline]
    pub fn apply(&self, mem: f32) -> (f32, f32) {
        let shifted = mem - self.threshold;
        let spike = self.surrogate.forward(shifted);
        let grad = self.surrogate.backward(shifted);
        (spike, grad)
    }

    /// Batch spike generation
    pub fn apply_batch(&self, mem: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut spikes = Vec::with_capacity(mem.len());
        let mut grads = Vec::with_capacity(mem.len());

        for &membrane in mem {
            let shifted = membrane - self.threshold;
            spikes.push(self.surrogate.forward(shifted));
            grads.push(self.surrogate.backward(shifted));
        }

        (spikes, grads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_sigmoid_forward() {
        let sg = SurrogateGradient::fast_sigmoid(25.0);
        assert_eq!(sg.forward(0.5), 1.0);
        assert_eq!(sg.forward(-0.5), 0.0);
        assert_eq!(sg.forward(0.0), 0.0); // Edge case: x=0 -> 0
    }

    #[test]
    fn test_fast_sigmoid_backward() {
        let sg = SurrogateGradient::fast_sigmoid(25.0);

        // At x=0, grad should be 1.0
        let grad_zero = sg.backward(0.0);
        assert!((grad_zero - 1.0).abs() < 1e-5);

        // Gradient should be symmetric around 0
        let grad_pos = sg.backward(0.1);
        let grad_neg = sg.backward(-0.1);
        assert!((grad_pos - grad_neg).abs() < 1e-5);

        // Gradient should decrease away from 0
        let grad_far = sg.backward(1.0);
        assert!(grad_far < grad_zero);
    }

    #[test]
    fn test_atan_backward() {
        let sg = SurrogateGradient::atan(2.0);

        // At x=0, grad should be alpha/2 = 1.0
        let grad_zero = sg.backward(0.0);
        assert!((grad_zero - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_spike_function() {
        let sf = SpikeFunction::new(SurrogateGradient::fast_sigmoid(25.0), 1.0);

        // Below threshold
        let (spike, _) = sf.apply(0.5);
        assert_eq!(spike, 0.0);

        // Above threshold
        let (spike, _) = sf.apply(1.5);
        assert_eq!(spike, 1.0);
    }
}
