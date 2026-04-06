//! Training utilities for spiking neural networks
//!
//! Provides training loop, optimizer, and parallel batch processing.

use crate::data::{BatchIterator, InputEncoder, MnistDataset};
use crate::network::{Network, NetworkGradients};
use crate::tensor::cross_entropy_loss;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub use crate::config::TrainingConfig;

const ADAM_DEFAULT_BETA1: f32 = 0.9;
const ADAM_DEFAULT_BETA2: f32 = 0.999;
const ADAM_DEFAULT_EPS: f32 = 1e-8;
const LR_SCHEDULER_MIN_FACTOR: f32 = 0.01;
const DEFAULT_TRAINER_DT: f32 = 0.001;
const DISABLED_ANALOG_GAIN: f32 = 0.0;
const PERCENT_SCALE: f32 = 100.0;

/// AdamW weight update: updates moments and applies weight decay + bias-corrected gradient step
fn adam_update_weight(
    weight: &mut Array2<f32>,
    grad: &Array2<f32>,
    first_moment: &mut Array2<f32>,
    second_moment: &mut Array2<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    beta1_correction: f32,
    beta2_correction: f32,
) {
    *first_moment = &*first_moment * beta1 + grad * (1.0 - beta1);
    *second_moment = &*second_moment * beta2 + &grad.mapv(|x| x * x) * (1.0 - beta2);
    let corrected_first = &*first_moment / beta1_correction;
    let corrected_second = &*second_moment / beta2_correction;
    *weight = &*weight * (1.0 - lr * weight_decay)
        - &(&corrected_first / &(corrected_second.mapv(|x| x.sqrt()) + eps) * lr);
}

/// Adam bias update: updates moments and applies bias-corrected gradient step (no weight decay)
fn adam_update_bias(
    bias: &mut Array1<f32>,
    grad: &Array1<f32>,
    first_moment: &mut Array1<f32>,
    second_moment: &mut Array1<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    beta1_correction: f32,
    beta2_correction: f32,
) {
    *first_moment = &*first_moment * beta1 + grad * (1.0 - beta1);
    *second_moment = &*second_moment * beta2 + &grad.mapv(|x| x * x) * (1.0 - beta2);
    let corrected_first = &*first_moment / beta1_correction;
    let corrected_second = &*second_moment / beta2_correction;
    *bias = &*bias - &(&corrected_first / &(corrected_second.mapv(|x| x.sqrt()) + eps) * lr);
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
            beta1: ADAM_DEFAULT_BETA1,
            beta2: ADAM_DEFAULT_BETA2,
            eps: ADAM_DEFAULT_EPS,
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
        let beta1_correction = 1.0 - self.beta1.powi(self.timestep as i32);
        let beta2_correction = 1.0 - self.beta2.powi(self.timestep as i32);

        // Update FC1 weight
        adam_update_weight(
            &mut net.fc1.weight,
            &grads.fc1_weight,
            &mut self.first_moment_fc1_weight,
            &mut self.second_moment_fc1_weight,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            beta1_correction,
            beta2_correction,
        );

        // Update FC1 bias
        if let (Some(ref mut fm), Some(ref mut sm), Some(ref g), Some(ref mut b)) = (
            &mut self.first_moment_fc1_bias,
            &mut self.second_moment_fc1_bias,
            &grads.fc1_bias,
            &mut net.fc1.bias,
        ) {
            adam_update_bias(
                b,
                g,
                fm,
                sm,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                beta1_correction,
                beta2_correction,
            );
        }

        // Update FC2 weight
        adam_update_weight(
            &mut net.fc2.weight,
            &grads.fc2_weight,
            &mut self.first_moment_fc2_weight,
            &mut self.second_moment_fc2_weight,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            beta1_correction,
            beta2_correction,
        );

        // Update FC2 bias
        if let (Some(ref mut fm), Some(ref mut sm), Some(ref g), Some(ref mut b)) = (
            &mut self.first_moment_fc2_bias,
            &mut self.second_moment_fc2_bias,
            &grads.fc2_bias,
            &mut net.fc2.bias,
        ) {
            adam_update_bias(
                b,
                g,
                fm,
                sm,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                beta1_correction,
                beta2_correction,
            );
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
            min_lr: initial_lr * LR_SCHEDULER_MIN_FACTOR, // Default min is 1% of initial
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

/// Count correct predictions by comparing argmax of spike counts against labels
fn count_correct(spike_count: &Array2<f32>, labels: &[usize]) -> (usize, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;
    for (i, &target) in labels.iter().enumerate() {
        let predicted = spike_count
            .row(i)
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
    (correct, total)
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
    /// Apply noise during evaluation too (not just training)
    pub noise_during_eval: bool,
    /// Use split-sign weight quantization (3-bit magnitude + 1-bit sign, independent scaling)
    pub split_sign_quant: bool,
    /// Input quantization bits (0 = no input quantization)
    pub input_quant_bits: u8,
}

impl Trainer {
    pub fn new(network: Network, config: TrainingConfig) -> Self {
        let optimizer = AdamOptimizer::new(&network, config.lr);
        let rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
        let bptt_steps = config.bptt_steps;

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
            dt: DEFAULT_TRAINER_DT, // Default 1ms timestep
            lr_scheduler: None,
            current_epoch: 0,
            max_grad_norm: None,
            analog_gain: DISABLED_ANALOG_GAIN, // Disabled by default
            input_encoder: None,               // Rate-coded by default
            bptt_steps,
            noise_during_eval: false,
            split_sign_quant: false,
            input_quant_bits: 0,
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
                self.network
                    .forward_with_adaptation(&images, self.config.num_steps, self.dt)
            } else if let Some(ref encoder) = self.input_encoder {
                // Use encoding forward for temporal or custom input encoding
                self.network
                    .forward_with_encoding(&images, encoder, self.config.num_steps)
            } else if self.analog_gain > DISABLED_ANALOG_GAIN {
                // Use analog forward for hybrid spike+membrane transmission
                self.network
                    .forward_with_analog(&images, self.config.num_steps, self.analog_gain)
            } else if self.input_quant_bits > 0 {
                // Use full quantized forward with input quantization
                self.network.forward_quantized_full(
                    &images,
                    self.config.num_steps,
                    self.quant_bits,
                    self.split_sign_quant,
                    self.input_quant_bits,
                )
            } else {
                // Use quantized forward (handles None for no quantization)
                self.network
                    .forward_quantized(&images, self.config.num_steps, self.quant_bits)
            };

            // Compute loss and gradient
            let (loss, grad_output) = cross_entropy_loss(&spike_count, &labels);

            // Backward pass
            let mut grads =
                self.network
                    .backward_truncated(&images, &caches, &grad_output, self.bptt_steps);

            // Gradient clipping (if enabled)
            if let Some(max_norm) = self.max_grad_norm {
                grads.clip_norm(max_norm);
            }

            // Update weights
            self.optimizer.step(&mut self.network, &grads);

            // Track metrics
            total_loss += loss * batch_size as f32;

            // Compute accuracy
            let (batch_correct, batch_total) = count_correct(&spike_count, &labels);
            correct += batch_correct;
            total += batch_total;
        }

        let avg_loss = total_loss / total as f32;
        let accuracy = PERCENT_SCALE * correct as f32 / total as f32;

        (avg_loss, accuracy)
    }

    /// Evaluate on test set
    pub fn evaluate(&mut self, dataset: &MnistDataset) -> f32 {
        let batch_iter = BatchIterator::new(dataset, self.config.batch_size, false, false);

        let mut correct = 0usize;
        let mut total = 0usize;

        for (images, labels) in batch_iter {
            // Use noisy forward during eval if noise_during_eval is set
            let (spike_count, _, _) = if self.noise_during_eval && self.noise.is_enabled() {
                self.network.forward_noisy(
                    &images,
                    self.config.num_steps,
                    self.noise.weight_std,
                    self.noise.threshold_std,
                    self.noise.membrane_std,
                    self.noise.input_std,
                    &mut self.rng,
                )
            } else if let Some(ref encoder) = self.input_encoder {
                // Use encoding forward for temporal encoding
                self.network
                    .forward_with_encoding(&images, encoder, self.config.num_steps)
            } else if self.input_quant_bits > 0 {
                // Use full quantized forward with input quantization
                self.network.forward_quantized_full(
                    &images,
                    self.config.num_steps,
                    self.quant_bits,
                    self.split_sign_quant,
                    self.input_quant_bits,
                )
            } else {
                // Use quantized forward (handles None for no quantization)
                self.network
                    .forward_quantized(&images, self.config.num_steps, self.quant_bits)
            };

            let (batch_correct, batch_total) = count_correct(&spike_count, &labels);
            correct += batch_correct;
            total += batch_total;
        }

        PERCENT_SCALE * correct as f32 / total as f32
    }

    /// Full training loop
    pub fn train(
        &mut self,
        dataset: &MnistDataset,
        callback: Option<&dyn Fn(&EpochResult)>,
    ) -> Vec<EpochResult> {
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
