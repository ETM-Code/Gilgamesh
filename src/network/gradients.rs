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
        if let (Some(ref mut acc), Some(ref other_bias)) = (&mut self.fc1_bias, &other.fc1_bias) {
            *acc = &*acc + other_bias;
        }
        self.fc2_weight = &self.fc2_weight + &other.fc2_weight;
        if let (Some(ref mut acc), Some(ref other_bias)) = (&mut self.fc2_bias, &other.fc2_bias) {
            *acc = &*acc + other_bias;
        }
    }

    /// Scale gradients by a factor
    pub fn scale(&mut self, factor: f32) {
        self.fc1_weight *= factor;
        if let Some(ref mut bias) = self.fc1_bias {
            *bias *= factor;
        }
        self.fc2_weight *= factor;
        if let Some(ref mut bias) = self.fc2_bias {
            *bias *= factor;
        }
    }

    /// Compute the total L2 norm of all gradients
    pub fn total_norm(&self) -> f32 {
        let mut squared_sum = 0.0f32;
        squared_sum += self.fc1_weight.iter().map(|x| x * x).sum::<f32>();
        if let Some(ref bias) = self.fc1_bias {
            squared_sum += bias.iter().map(|x| x * x).sum::<f32>();
        }
        squared_sum += self.fc2_weight.iter().map(|x| x * x).sum::<f32>();
        if let Some(ref bias) = self.fc2_bias {
            squared_sum += bias.iter().map(|x| x * x).sum::<f32>();
        }
        squared_sum.sqrt()
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
