//! Characterization tests for src/training/mod.rs

mod common;
use common::*;

use gilgamesh::config::TrainingConfig;
use gilgamesh::data::ArrayDataset;
use gilgamesh::network::{Network, NetworkGradients};
use gilgamesh::training::{AdamOptimizer, LRScheduler, NoiseParams, Trainer};
use ndarray::Array2;

#[test]
fn adam_step_numeric_golden() {
    let net = Network::new(4, 3, 2, 0.9, 42);
    let mut opt = AdamOptimizer::new(&net, 1e-3);
    let mut net_mut = net.clone();

    let mut g = NetworkGradients::zeros_like(&net);
    g.fc1_weight.fill(0.01);
    g.fc2_weight.fill(0.01);

    opt.step(&mut net_mut, &g);
    assert_eq!(opt.timestep, 1);
    close32(net_mut.fc1.weight[[0, 0]], 0.31330508, 1e-6);
    close32(net_mut.fc2.weight[[0, 0]], -0.3842985, 1e-6);

    opt.step(&mut net_mut, &g);
    assert_eq!(opt.timestep, 2);
    close32(net_mut.fc1.weight[[0, 0]], 0.3123051, 1e-6);
    close32(net_mut.fc2.weight[[0, 0]], -0.3852985, 1e-6);
}

#[test]
fn lr_scheduler_cosine_golden() {
    let s = LRScheduler::new(1e-3, 10);
    // epoch 0 -> initial; epoch 10 -> min (1% of initial by default); epoch 5 -> midpoint.
    close32(s.get_lr(0), 1e-3, 1e-9);
    close32(s.get_lr(10), 1e-5, 1e-9); // min = initial * 0.01
    // midpoint: min + 0.5*(init-min)*(1+cos(pi/2)) = min + 0.5*(init-min)
    let mid = 1e-5 + 0.5 * (1e-3 - 1e-5);
    close32(s.get_lr(5), mid, 1e-7);

    // with_min_lr override.
    let s2 = LRScheduler::new(1e-3, 10).with_min_lr(2e-4);
    close32(s2.get_lr(10), 2e-4, 1e-9);
}

#[test]
fn train_epoch_deterministic_on_array_dataset() {
    // Replicates the in-source MockDataset (sin/cos deterministic) via the public
    // ArrayDataset. Trainer rng is seeded Xoshiro(seed) so shuffle is deterministic.
    // Noise disabled so no extra rng draws.
    let train_images = Array2::from_shape_fn((32, 4), |(r, c)| ((r + c) as f32).sin() * 0.1);
    let test_images = Array2::from_shape_fn((16, 4), |(r, c)| ((r + c) as f32).cos() * 0.1);
    let ds = ArrayDataset {
        train_images,
        train_labels: (0..32).map(|i| i % 2).collect(),
        test_images,
        test_labels: (0..16).map(|i| i % 2).collect(),
    };

    let net = Network::new(4, 3, 2, 0.9, 42);
    let config = TrainingConfig {
        lr: 1e-3,
        epochs: 1,
        batch_size: 8,
        num_steps: 3,
        seed: 42,
        num_workers: 0,
        bptt_steps: None,
        weight_decay: 0.01,
        max_grad_norm: 1.0,
    };
    let mut trainer = Trainer::new(net, config);
    trainer.noise = NoiseParams::default(); // all zero -> disabled

    let (loss, train_acc) = trainer.train_epoch(&ds);
    let test_acc = trainer.evaluate(&ds);

    // CHARACTERIZATION: default Physics neurons produce no spikes here, so the
    // network is degenerate; loss/accuracy are locked at their observed values.
    close32(loss, 0.6931472, 1e-5);
    close32(train_acc, 50.0, 1e-4);
    close32(test_acc, 50.0, 1e-4);
}

#[test]
fn noise_params_is_enabled() {
    assert!(!NoiseParams::default().is_enabled());
    let mut p = NoiseParams::default();
    p.weight_std = 0.05;
    assert!(p.is_enabled());
}
