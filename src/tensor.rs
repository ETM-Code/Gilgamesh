//! Tensor operations with automatic differentiation support
//!
//! This module provides a simple tensor abstraction built on ndarray
//! with gradient tracking for backpropagation.

use ndarray::{Array1, Array2, Axis};

/// A 2D tensor with optional gradient storage
#[derive(Clone, Debug)]
pub struct Tensor2D {
    pub data: Array2<f32>,
    pub grad: Option<Array2<f32>>,
    pub requires_grad: bool,
}

impl Tensor2D {
    pub fn new(data: Array2<f32>, requires_grad: bool) -> Self {
        let grad = if requires_grad {
            Some(Array2::zeros(data.raw_dim()))
        } else {
            None
        };
        Self {
            data,
            grad,
            requires_grad,
        }
    }

    pub fn zeros(shape: (usize, usize), requires_grad: bool) -> Self {
        Self::new(Array2::zeros(shape), requires_grad)
    }

    pub fn zeros_like(&self) -> Self {
        Self::new(Array2::zeros(self.data.raw_dim()), self.requires_grad)
    }

    pub fn shape(&self) -> (usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1])
    }

    /// Accumulate gradient
    pub fn accumulate_grad(&mut self, grad: &Array2<f32>) {
        if let Some(ref mut g) = self.grad {
            *g += grad;
        }
    }
}

/// A 1D tensor (vector) with optional gradient storage
#[derive(Clone, Debug)]
pub struct Tensor1D {
    pub data: Array1<f32>,
    pub grad: Option<Array1<f32>>,
    pub requires_grad: bool,
}

impl Tensor1D {
    pub fn new(data: Array1<f32>, requires_grad: bool) -> Self {
        let grad = if requires_grad {
            Some(Array1::zeros(data.len()))
        } else {
            None
        };
        Self {
            data,
            grad,
            requires_grad,
        }
    }

    pub fn zeros(size: usize, requires_grad: bool) -> Self {
        Self::new(Array1::zeros(size), requires_grad)
    }

    pub fn zeros_like(&self) -> Self {
        Self::new(Array1::zeros(self.data.len()), self.requires_grad)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Accumulate gradient
    pub fn accumulate_grad(&mut self, grad: &Array1<f32>) {
        if let Some(ref mut g) = self.grad {
            *g += grad;
        }
    }
}

/// Matrix multiplication: (batch, in) @ (in, out) -> (batch, out)
pub fn matmul(lhs: &Array2<f32>, rhs: &Array2<f32>) -> Array2<f32> {
    lhs.dot(rhs)
}

/// Batched matrix-vector multiplication with bias
pub fn linear_forward(
    input: &Array2<f32>,
    weight: &Array2<f32>,
    bias: Option<&Array1<f32>>,
) -> Array2<f32> {
    let mut output = input.dot(weight);
    if let Some(bias_vec) = bias {
        output += bias_vec;
    }
    output
}

/// Softmax along the last axis
pub fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let max_vals = logits.map_axis(Axis(1), |row| {
        row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    });

    let mut exp_logits = logits.clone();
    for (i, mut row) in exp_logits.outer_iter_mut().enumerate() {
        row -= max_vals[i];
        row.mapv_inplace(|x| x.exp());
    }

    let sums = exp_logits.sum_axis(Axis(1));
    for (i, mut row) in exp_logits.outer_iter_mut().enumerate() {
        row /= sums[i];
    }

    exp_logits
}

/// Cross-entropy loss for classification
/// Returns (loss, gradient w.r.t. logits)
pub fn cross_entropy_loss(logits: &Array2<f32>, targets: &[usize]) -> (f32, Array2<f32>) {
    let batch_size = logits.shape()[0];
    let probs = softmax(logits);

    // Compute loss: -sum(log(p[target]))
    let mut loss = 0.0;
    for (i, &target) in targets.iter().enumerate() {
        loss -= probs[[i, target]].max(1e-7).ln();
    }
    loss /= batch_size as f32;

    // Gradient: softmax - one_hot(target)
    let mut grad = probs;
    for (i, &target) in targets.iter().enumerate() {
        grad[[i, target]] -= 1.0;
    }
    grad /= batch_size as f32;

    (loss, grad)
}

/// Class-weighted cross-entropy loss for imbalanced classification.
/// `class_weights` should be indexed by class id.
/// Returns (loss, gradient w.r.t. logits).
pub fn cross_entropy_loss_weighted(
    logits: &Array2<f32>,
    targets: &[usize],
    class_weights: &[f32],
) -> (f32, Array2<f32>) {
    let batch_size = logits.shape()[0];
    let probs = softmax(logits);

    let mut loss = 0.0;
    let mut grad = probs.clone();

    for (i, &target) in targets.iter().enumerate() {
        let weight = *class_weights.get(target).unwrap_or(&1.0);
        loss -= weight * probs[[i, target]].max(1e-7).ln();
        grad[[i, target]] -= 1.0;
        // Scale per-sample gradient row by class weight.
        grad.row_mut(i).mapv_inplace(|v| v * weight);
    }

    loss /= batch_size as f32;
    grad /= batch_size as f32;

    (loss, grad)
}

/// Heaviside step function (spike generation)
#[inline]
pub fn heaviside(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Element-wise heaviside for arrays
pub fn heaviside_array(values: &Array2<f32>) -> Array2<f32> {
    values.mapv(heaviside)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_softmax() {
        let logits = array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]];
        let probs = softmax(&logits);

        // Each row should sum to 1
        for row in probs.outer_iter() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cross_entropy() {
        let logits = array![[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]];
        let targets = vec![0, 2];
        let (loss, grad) = cross_entropy_loss(&logits, &targets);

        assert!(loss > 0.0);
        assert_eq!(grad.shape(), logits.shape());
    }

    #[test]
    fn test_cross_entropy_weighted() {
        let logits = array![[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]];
        let targets = vec![0, 2];
        let class_weights = vec![1.0, 1.0, 2.0];
        let (loss_w, grad_w) = cross_entropy_loss_weighted(&logits, &targets, &class_weights);
        let (loss, grad) = cross_entropy_loss(&logits, &targets);

        assert!(loss_w > loss);
        assert_eq!(grad_w.shape(), grad.shape());
    }
}
