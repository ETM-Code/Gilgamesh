//! Network gradient types for training
//!
//! Contains accumulated gradients for network parameters.

use ndarray::{Array1, Array2};

use super::Network;

/// Accumulated gradients for network parameters
#[derive(Clone, Debug)]
pub struct NetworkGradients {
    pub fc1_weight: Array2<f32>,
    pub fc1_bias: Option<Array1<f32>>,
    pub fc2_weight: Array2<f32>,
    pub fc2_bias: Option<Array1<f32>>,
}

impl NetworkGradients {
    pub fn zeros_like(net: &Network) -> Self {
        Self {
            fc1_weight: Array2::zeros(net.fc1.weight.raw_dim()),
            fc1_bias: net.fc1.bias.as_ref().map(|b| Array1::zeros(b.len())),
            fc2_weight: Array2::zeros(net.fc2.weight.raw_dim()),
            fc2_bias: net.fc2.bias.as_ref().map(|b| Array1::zeros(b.len())),
        }
    }

    /// Add another gradient to this one
    pub fn add(&mut self, other: &NetworkGradients) {
        self.fc1_weight = &self.fc1_weight + &other.fc1_weight;
        if let (Some(ref mut a), Some(ref b)) = (&mut self.fc1_bias, &other.fc1_bias) {
            *a = &*a + b;
        }
        self.fc2_weight = &self.fc2_weight + &other.fc2_weight;
        if let (Some(ref mut a), Some(ref b)) = (&mut self.fc2_bias, &other.fc2_bias) {
            *a = &*a + b;
        }
    }

    /// Scale gradients by a factor
    pub fn scale(&mut self, factor: f32) {
        self.fc1_weight *= factor;
        if let Some(ref mut b) = self.fc1_bias {
            *b *= factor;
        }
        self.fc2_weight *= factor;
        if let Some(ref mut b) = self.fc2_bias {
            *b *= factor;
        }
    }

    /// Compute the total L2 norm of all gradients
    pub fn total_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        sum_sq += self.fc1_weight.iter().map(|x| x * x).sum::<f32>();
        if let Some(ref b) = self.fc1_bias {
            sum_sq += b.iter().map(|x| x * x).sum::<f32>();
        }
        sum_sq += self.fc2_weight.iter().map(|x| x * x).sum::<f32>();
        if let Some(ref b) = self.fc2_bias {
            sum_sq += b.iter().map(|x| x * x).sum::<f32>();
        }
        sum_sq.sqrt()
    }

    /// Clip gradients by global norm (in place)
    /// Returns the original norm before clipping
    pub fn clip_norm(&mut self, max_norm: f32) -> f32 {
        let total_norm = self.total_norm();
        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            self.scale(scale);
        }
        total_norm
    }
}
