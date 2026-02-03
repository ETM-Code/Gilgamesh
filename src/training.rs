//! Training utilities for spiking neural networks
//!
//! Provides training loop, optimizer, and parallel batch processing.

use crate::data::{BatchIterator, InputEncoder, MnistDataset};
use crate::network::{Network, NetworkGradients};
use crate::tensor::cross_entropy_loss;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Training configuration
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// Learning rate
    pub lr: f32,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Number of timesteps per sample
    pub num_steps: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of parallel workers (0 = auto-detect)
    pub num_workers: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,       // snnTorch default
            epochs: 15,
            batch_size: 128,
            num_steps: 25,
            seed: 42,
            num_workers: 0, // auto-detect
        }
    }
}

/// Adam optimizer state with optional weight decay (AdamW)
#[derive(Clone)]
pub struct AdamOptimizer {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub timestep: usize,
    // Moment estimates for each parameter group
    pub first_moment_fc1_weight: Array2<f32>,
    pub second_moment_fc1_weight: Array2<f32>,
    pub first_moment_fc1_bias: Option<Array1<f32>>,
    pub second_moment_fc1_bias: Option<Array1<f32>>,
    pub first_moment_fc2_weight: Array2<f32>,
    pub second_moment_fc2_weight: Array2<f32>,
    pub first_moment_fc2_bias: Option<Array1<f32>>,
    pub second_moment_fc2_bias: Option<Array1<f32>>,
}

impl AdamOptimizer {
    pub fn new(net: &Network, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            timestep: 0,
            first_moment_fc1_weight: Array2::zeros(net.fc1.weight.raw_dim()),
            second_moment_fc1_weight: Array2::zeros(net.fc1.weight.raw_dim()),
            first_moment_fc1_bias: net.fc1.bias.as_ref().map(|b| Array1::zeros(b.len())),
            second_moment_fc1_bias: net.fc1.bias.as_ref().map(|b| Array1::zeros(b.len())),
            first_moment_fc2_weight: Array2::zeros(net.fc2.weight.raw_dim()),
            second_moment_fc2_weight: Array2::zeros(net.fc2.weight.raw_dim()),
            first_moment_fc2_bias: net.fc2.bias.as_ref().map(|b| Array1::zeros(b.len())),
            second_moment_fc2_bias: net.fc2.bias.as_ref().map(|b| Array1::zeros(b.len())),
        }
    }

    /// Set learning rate (for LR scheduling)
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.lr
    }

    /// Set weight decay (L2 regularization strength)
    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.weight_decay = weight_decay;
    }

    pub fn step(&mut self, net: &mut Network, grads: &NetworkGradients) {
        self.timestep += 1;

        // Bias correction factors
        let beta1_correction = 1.0 - self.beta1.powi(self.timestep as i32);
        let beta2_correction = 1.0 - self.beta2.powi(self.timestep as i32);

        // Update FC1 weight (with AdamW weight decay)
        self.first_moment_fc1_weight = &self.first_moment_fc1_weight * self.beta1 + &grads.fc1_weight * (1.0 - self.beta1);
        self.second_moment_fc1_weight = &self.second_moment_fc1_weight * self.beta2 + &grads.fc1_weight.mapv(|x| x * x) * (1.0 - self.beta2);
        let corrected_first_moment = &self.first_moment_fc1_weight / beta1_correction;
        let corrected_second_moment = &self.second_moment_fc1_weight / beta2_correction;
        // AdamW: weight decay applied separately from gradient
        net.fc1.weight = &net.fc1.weight * (1.0 - self.lr * self.weight_decay)
            - &(&corrected_first_moment / &(corrected_second_moment.mapv(|x| x.sqrt()) + self.eps) * self.lr);

        // Update FC1 bias
        if let (Some(ref mut first_moment), Some(ref mut second_moment), Some(ref grad), Some(ref mut bias)) = (
            &mut self.first_moment_fc1_bias,
            &mut self.second_moment_fc1_bias,
            &grads.fc1_bias,
            &mut net.fc1.bias,
        ) {
            *first_moment = &*first_moment * self.beta1 + grad * (1.0 - self.beta1);
            *second_moment = &*second_moment * self.beta2 + &grad.mapv(|x| x * x) * (1.0 - self.beta2);
            let corrected_first_moment = &*first_moment / beta1_correction;
            let corrected_second_moment = &*second_moment / beta2_correction;
            *bias = &*bias - &(&corrected_first_moment / &(corrected_second_moment.mapv(|x| x.sqrt()) + self.eps) * self.lr);
        }

        // Update FC2 weight (with AdamW weight decay)
        self.first_moment_fc2_weight = &self.first_moment_fc2_weight * self.beta1 + &grads.fc2_weight * (1.0 - self.beta1);
        self.second_moment_fc2_weight = &self.second_moment_fc2_weight * self.beta2 + &grads.fc2_weight.mapv(|x| x * x) * (1.0 - self.beta2);
        let corrected_first_moment = &self.first_moment_fc2_weight / beta1_correction;
        let corrected_second_moment = &self.second_moment_fc2_weight / beta2_correction;
        // AdamW: weight decay applied separately from gradient
        net.fc2.weight = &net.fc2.weight * (1.0 - self.lr * self.weight_decay)
            - &(&corrected_first_moment / &(corrected_second_moment.mapv(|x| x.sqrt()) + self.eps) * self.lr);

        // Update FC2 bias
        if let (Some(ref mut first_moment), Some(ref mut second_moment), Some(ref grad), Some(ref mut bias)) = (
            &mut self.first_moment_fc2_bias,
            &mut self.second_moment_fc2_bias,
            &grads.fc2_bias,
            &mut net.fc2.bias,
        ) {
            *first_moment = &*first_moment * self.beta1 + grad * (1.0 - self.beta1);
            *second_moment = &*second_moment * self.beta2 + &grad.mapv(|x| x * x) * (1.0 - self.beta2);
            let corrected_first_moment = &*first_moment / beta1_correction;
            let corrected_second_moment = &*second_moment / beta2_correction;
            *bias = &*bias - &(&corrected_first_moment / &(corrected_second_moment.mapv(|x| x.sqrt()) + self.eps) * self.lr);
        }
    }
}

/// Learning rate scheduler with cosine annealing
#[derive(Clone, Debug)]
pub struct LRScheduler {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Minimum learning rate
    pub min_lr: f32,
    /// Total number of epochs
    pub total_epochs: usize,
}

impl LRScheduler {
    pub fn new(initial_lr: f32, total_epochs: usize) -> Self {
        Self {
            initial_lr,
            min_lr: initial_lr * 0.01, // Default min is 1% of initial
            total_epochs,
        }
    }

    /// Create scheduler with custom minimum LR
    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Compute learning rate for given epoch using cosine annealing
    /// lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * epoch / total_epochs))
    pub fn get_lr(&self, epoch: usize) -> f32 {
        let progress = (epoch as f32) / (self.total_epochs as f32);
        let cosine = (std::f32::consts::PI * progress).cos();
        self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + cosine)
    }
}

/// Training result for an epoch
#[derive(Clone, Debug)]
pub struct EpochResult {
    pub epoch: usize,
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub test_accuracy: f32,
}

/// Noise configuration for robustness training
#[derive(Clone, Debug, Default)]
pub struct NoiseParams {
    /// Weight noise std (relative, e.g., 0.05 for 5%)
    pub weight_std: f32,
    /// Threshold noise std (relative, e.g., 0.02 for 2%)
    pub threshold_std: f32,
    /// Membrane noise std (absolute)
    pub membrane_std: f32,
    /// Input noise std (relative, e.g., 0.1 for 10%)
    pub input_std: f32,
}

impl NoiseParams {
    pub fn is_enabled(&self) -> bool {
        self.weight_std > 0.0
            || self.threshold_std > 0.0
            || self.membrane_std > 0.0
            || self.input_std > 0.0
    }
}

/// Trainer with parallel batch processing
pub struct Trainer {
    pub config: TrainingConfig,
    pub network: Network,
    pub optimizer: AdamOptimizer,
    /// Optional weight quantization bits (None = no quantization)
    pub quant_bits: Option<u8>,
    /// Noise parameters for robustness training
    pub noise: NoiseParams,
    /// RNG for noise injection
    rng: Xoshiro256PlusPlus,
    /// Enable threshold adaptation
    pub adaptation_enabled: bool,
    /// Integration timestep for physics/adaptation mode
    pub dt: f32,
    /// Optional LR scheduler for cosine annealing
    pub lr_scheduler: Option<LRScheduler>,
    /// Current epoch (for LR scheduling)
    pub current_epoch: usize,
    /// Maximum gradient norm for clipping (None = no clipping)
    pub max_grad_norm: Option<f32>,
    /// Analog gain for hybrid spike+membrane inter-layer transmission (0.0 = disabled)
    pub analog_gain: f32,
    /// Optional input encoder for temporal encoding (None = rate-coded)
    pub input_encoder: Option<InputEncoder>,
    /// Truncated BPTT steps (None = full BPTT through all timesteps)
    pub bptt_steps: Option<usize>,
}

impl Trainer {
    pub fn new(network: Network, config: TrainingConfig) -> Self {
        let optimizer = AdamOptimizer::new(&network, config.lr);
        let rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);

        // Configure rayon thread pool
        if config.num_workers > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_workers)
                .build_global()
                .ok(); // Ignore if already initialized
        }

        Self {
            config,
            network,
            optimizer,
            quant_bits: None,
            noise: NoiseParams::default(),
            rng,
            adaptation_enabled: false,
            dt: 0.001, // Default 1ms timestep
            lr_scheduler: None,
            current_epoch: 0,
            max_grad_norm: None,
            analog_gain: 0.0, // Disabled by default
            input_encoder: None, // Rate-coded by default
            bptt_steps: None, // Full BPTT by default
        }
    }

    /// Enable gradient clipping by global norm
    pub fn with_grad_clip(mut self, max_norm: f32) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Enable cosine annealing LR schedule
    pub fn with_lr_schedule(mut self, total_epochs: usize) -> Self {
        self.lr_scheduler = Some(LRScheduler::new(self.config.lr, total_epochs));
        self
    }

    /// Update learning rate for the given epoch (call at start of each epoch)
    /// Returns the new learning rate if scheduler is enabled
    pub fn update_lr_for_epoch(&mut self, epoch: usize) -> Option<f32> {
        self.current_epoch = epoch;
        if let Some(ref scheduler) = self.lr_scheduler {
            let new_lr = scheduler.get_lr(epoch);
            self.optimizer.set_lr(new_lr);
            Some(new_lr)
        } else {
            None
        }
    }

    /// Enable weight quantization during training (QAT)
    pub fn with_quantization(mut self, bits: u8) -> Self {
        self.quant_bits = Some(bits);
        self
    }

    /// Enable noise injection during training
    pub fn with_noise(mut self, params: NoiseParams) -> Self {
        self.noise = params;
        self
    }

    /// Enable threshold adaptation during training
    pub fn with_adaptation(mut self, dt: f32) -> Self {
        self.adaptation_enabled = true;
        self.dt = dt;
        self
    }

    /// Train for one epoch with parallel batch processing
    pub fn train_epoch(&mut self, dataset: &MnistDataset) -> (f32, f32) {
        let batch_iter = BatchIterator::new(dataset, self.config.batch_size, true, true);
        let _num_batches = batch_iter.num_batches();

        let mut total_loss = 0.0;
        let mut correct = 0usize;
        let mut total = 0usize;

        for (images, labels) in batch_iter {
            let batch_size = images.shape()[0];

            // Forward pass (with optional quantization, noise, adaptation, analog, or encoding)
            let (spike_count, _, caches) = if self.noise.is_enabled() {
                // Use noisy forward for robustness training
                self.network.forward_noisy(
                    &images,
                    self.config.num_steps,
                    self.noise.weight_std,
                    self.noise.threshold_std,
                    self.noise.membrane_std,
                    self.noise.input_std,
                    &mut self.rng,
                )
            } else if self.adaptation_enabled {
                // Use adaptation forward for threshold adaptation
                self.network.forward_with_adaptation(&images, self.config.num_steps, self.dt)
            } else if let Some(ref encoder) = self.input_encoder {
                // Use encoding forward for temporal or custom input encoding
                self.network.forward_with_encoding(&images, encoder, self.config.num_steps)
            } else if self.analog_gain > 0.0 {
                // Use analog forward for hybrid spike+membrane transmission
                self.network.forward_with_analog(&images, self.config.num_steps, self.analog_gain)
            } else {
                // Use quantized forward (handles None for no quantization)
                self.network.forward_quantized(&images, self.config.num_steps, self.quant_bits)
            };

            // Compute loss and gradient
            let (loss, grad_output) = cross_entropy_loss(&spike_count, &labels);

            // Backward pass
            let mut grads = self.network.backward_truncated(&images, &caches, &grad_output, self.bptt_steps);

            // Gradient clipping (if enabled)
            if let Some(max_norm) = self.max_grad_norm {
                grads.clip_norm(max_norm);
            }

            // Update weights
            self.optimizer.step(&mut self.network, &grads);

            // Track metrics
            total_loss += loss * batch_size as f32;

            // Compute accuracy
            for (i, &target) in labels.iter().enumerate() {
                let row = spike_count.row(i);
                let predicted = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if predicted == target {
                    correct += 1;
                }
                total += 1;
            }
        }

        let avg_loss = total_loss / total as f32;
        let accuracy = 100.0 * correct as f32 / total as f32;

        (avg_loss, accuracy)
    }

    /// Evaluate on test set
    pub fn evaluate(&self, dataset: &MnistDataset) -> f32 {
        let batch_iter = BatchIterator::new(dataset, self.config.batch_size, false, false);

        let mut correct = 0usize;
        let mut total = 0usize;

        for (images, labels) in batch_iter {
            // Use same forward mode as training for consistent evaluation
            let (spike_count, _, _) = if let Some(ref encoder) = self.input_encoder {
                // Use encoding forward for temporal encoding
                self.network.forward_with_encoding(&images, encoder, self.config.num_steps)
            } else {
                // Use quantized forward (handles None for no quantization)
                self.network.forward_quantized(&images, self.config.num_steps, self.quant_bits)
            };

            for (i, &target) in labels.iter().enumerate() {
                let row = spike_count.row(i);
                let predicted = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if predicted == target {
                    correct += 1;
                }
                total += 1;
            }
        }

        100.0 * correct as f32 / total as f32
    }

    /// Full training loop
    pub fn train(&mut self, dataset: &MnistDataset, callback: Option<&dyn Fn(&EpochResult)>) -> Vec<EpochResult> {
        let mut results = Vec::with_capacity(self.config.epochs);

        for epoch in 1..=self.config.epochs {
            // Update learning rate if scheduler is enabled
            self.update_lr_for_epoch(epoch);

            let (train_loss, train_acc) = self.train_epoch(dataset);
            let test_acc = self.evaluate(dataset);

            let result = EpochResult {
                epoch,
                train_loss,
                train_accuracy: train_acc,
                test_accuracy: test_acc,
            };

            if let Some(cb) = callback {
                cb(&result);
            }

            results.push(result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_optimizer() {
        let net = Network::new(49, 100, 10, 0.9, 42);
        let mut optimizer = AdamOptimizer::new(&net, 1e-3);

        // Create mock gradients
        let grads = NetworkGradients::zeros_like(&net);

        let mut net_mut = net.clone();
        optimizer.step(&mut net_mut, &grads);

        assert_eq!(optimizer.timestep, 1);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.lr, 1e-3);
        assert_eq!(config.epochs, 15);
        assert_eq!(config.batch_size, 128);
        assert_eq!(config.num_steps, 25);
    }
}
