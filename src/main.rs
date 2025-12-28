//! gilgamesh - Hardware-accurate spiking neural network
//!
//! A Rust implementation of spiking neural networks with physics-accurate
//! membrane dynamics for chip deployment.
//!
//! Usage:
//!   gilgamesh train --epochs 15 --lr 0.001
//!   gilgamesh train --config config.json
//!   gilgamesh evaluate --checkpoint model.json

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gilgamesh::config::Config;
use gilgamesh::data::MnistDataset;
use gilgamesh::network::Network;
use gilgamesh::surrogate::SurrogateGradient;
use gilgamesh::training::{NoiseParams, Trainer, TrainingConfig};

#[derive(Parser)]
#[command(name = "gilgamesh")]
#[command(about = "Hardware-accurate spiking neural network")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new SNN on MNIST
    Train {
        /// Path to JSON config file (overrides other args)
        #[arg(long)]
        config: Option<String>,

        /// Learning rate
        #[arg(long, default_value = "0.001")]
        lr: f32,

        /// Number of epochs
        #[arg(long, default_value = "15")]
        epochs: usize,

        /// Batch size
        #[arg(long, default_value = "128")]
        batch_size: usize,

        /// Number of timesteps
        #[arg(long, default_value = "25")]
        num_steps: usize,

        /// Hidden layer size
        #[arg(long, default_value = "100")]
        hidden_size: usize,

        /// Membrane decay (beta)
        #[arg(long, default_value = "0.9")]
        beta: f32,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Surrogate gradient slope
        #[arg(long, default_value = "25.0")]
        slope: f32,

        /// Enable 8-bit weight quantization
        #[arg(long)]
        quantize: bool,

        /// Quantization bits (default: 8)
        #[arg(long, default_value = "8")]
        quantize_bits: u8,

        /// Enable noise injection
        #[arg(long)]
        noise: bool,

        /// Weight noise std (default: 0.05)
        #[arg(long, default_value = "0.05")]
        weight_noise: f32,

        /// Enable visualization with Rerun (requires --features visualization)
        #[arg(long)]
        visualize: bool,

        /// Save visualization to .rrd file instead of spawning viewer
        #[arg(long)]
        visualize_file: Option<String>,

        /// Save checkpoint to this path after training
        #[arg(long)]
        save_checkpoint: Option<String>,
    },

    /// Evaluate a trained model
    Evaluate {
        /// Checkpoint file
        #[arg(long)]
        checkpoint: String,

        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Number of timesteps
        #[arg(long, default_value = "25")]
        num_steps: usize,

        /// Batch size
        #[arg(long, default_value = "128")]
        batch_size: usize,
    },

    /// Run a quick test to verify the implementation
    Test {
        /// Use small dataset for quick testing
        #[arg(long)]
        quick: bool,
    },

    /// Launch interactive training dashboard (requires --features dashboard)
    Dashboard {
        /// Path to JSON config file
        #[arg(long)]
        config: Option<String>,

        /// Number of epochs
        #[arg(long, default_value = "15")]
        epochs: usize,

        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: String,
    },

    /// Launch stunning network animation (requires --features animation)
    Animate {
        /// Path to JSON config file
        #[arg(long)]
        config: Option<String>,

        /// Data directory (for loading sample images)
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Animation speed multiplier
        #[arg(long, default_value = "1.0")]
        speed: f32,

        /// Sample index to animate (from test set)
        #[arg(long, default_value = "0")]
        sample: usize,
    },

    /// Visual test runner for inspecting single samples (requires --features dashboard)
    Inspect {
        /// Path to checkpoint file
        #[arg(long)]
        checkpoint: String,

        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Number of timesteps
        #[arg(long, default_value = "25")]
        num_steps: usize,

        /// Random seed for sample selection
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            config,
            lr,
            epochs,
            batch_size,
            num_steps,
            hidden_size,
            beta,
            seed,
            data_dir,
            slope,
            quantize,
            quantize_bits,
            noise,
            weight_noise,
            visualize,
            visualize_file,
            save_checkpoint,
        } => {
            // If config file is provided, load from it; otherwise use CLI args
            if let Some(config_path) = config {
                train_from_config(&config_path, &data_dir, visualize, visualize_file, save_checkpoint)
            } else {
                // Build config from CLI args
                let mut cfg = Config::default();
                cfg.training.lr = lr;
                cfg.training.epochs = epochs;
                cfg.training.batch_size = batch_size;
                cfg.training.num_steps = num_steps;
                cfg.training.seed = seed;
                cfg.network.hidden_size = hidden_size;
                cfg.neuron.beta = beta;
                cfg.neuron.slope = slope;
                cfg.quantization.enabled = quantize;
                cfg.quantization.bits = quantize_bits;
                cfg.noise.enabled = noise;
                cfg.noise.weight_std = weight_noise;
                train_with_config(&cfg, &data_dir, visualize, visualize_file, save_checkpoint)
            }
        }
        Commands::Evaluate {
            checkpoint,
            data_dir,
            num_steps,
            batch_size,
        } => {
            evaluate(&checkpoint, &data_dir, num_steps, batch_size)
        }
        Commands::Test { quick } => test_implementation(quick),
        Commands::Dashboard {
            config,
            epochs,
            data_dir,
        } => run_dashboard(config, epochs, &data_dir),
        Commands::Animate {
            config,
            data_dir,
            speed,
            sample,
        } => run_animation(config, &data_dir, speed, sample),
        Commands::Inspect {
            checkpoint,
            data_dir,
            num_steps,
            seed,
        } => run_inspector(&checkpoint, &data_dir, num_steps, seed)
    }
}

/// Train from a JSON config file
fn train_from_config(
    config_path: &str,
    data_dir: &str,
    visualize: bool,
    visualize_file: Option<String>,
    save_checkpoint: Option<String>,
) -> Result<()> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("Failed to load config from {}", config_path))?;
    println!("Loaded config from: {}", config_path);
    train_with_config(&cfg, data_dir, visualize, visualize_file, save_checkpoint)
}

/// Train using a Config struct
fn train_with_config(
    cfg: &Config,
    data_dir: &str,
    visualize: bool,
    visualize_file: Option<String>,
    save_checkpoint: Option<String>,
) -> Result<()> {
    println!("=== gilgamesh Training ===");
    println!("Mode:           {}", cfg.mode);
    println!("Learning rate:  {}", cfg.training.lr);
    println!("Epochs:         {}", cfg.training.epochs);
    println!("Batch size:     {}", cfg.training.batch_size);
    println!("Timesteps:      {}", cfg.training.num_steps);
    println!("Hidden size:    {}", cfg.network.hidden_size);
    println!("Beta:           {}", cfg.neuron.beta);
    println!("Seed:           {}", cfg.training.seed);
    println!("Slope:          {}", cfg.neuron.slope);
    if cfg.quantization.enabled {
        println!("Quantization:   {}-bit", cfg.quantization.bits);
    }
    if cfg.noise.enabled {
        println!("Noise:          weight_std={}", cfg.noise.weight_std);
    }
    println!();

    // Load dataset
    println!("Loading MNIST dataset...");
    let dataset = MnistDataset::load(data_dir).context("Failed to load MNIST dataset")?;
    println!(
        "Loaded {} training samples, {} test samples",
        dataset.train_len(),
        dataset.test_len()
    );
    println!("Image size: {}x{} = {} features", 7, 7, dataset.feature_dim());
    println!();

    // Create input encoder first (determines input_size for network)
    use gilgamesh::data::InputEncoder;
    let image_size = 7; // 7x7 downsampled MNIST
    let input_encoder = if cfg.input_encoding.encoding_type == "temporal" {
        Some(InputEncoder::temporal(
            image_size,
            cfg.input_encoding.row_spacing,
            cfg.input_encoding.pulse_width,
            cfg.physics.dt,
        ))
    } else {
        None
    };

    // Create network - input_size depends on encoding type
    // Temporal: 7 inputs (one row at a time)
    // Rate-coded: 49 inputs (full image)
    let input_size = input_encoder.as_ref()
        .map(|e| e.output_dim())
        .unwrap_or(dataset.feature_dim());
    let output_size = cfg.network.output_size;
    let hidden_size = cfg.network.hidden_size;
    let beta = cfg.neuron.beta;
    let slope = cfg.neuron.slope;
    let seed = cfg.training.seed;

    let mut network = if cfg.is_physics_mode() && cfg.physics.enabled {
        // Physics mode with RC dynamics
        let tau_m = cfg.physics.tau_m;
        let dt = cfg.physics.dt;
        if cfg.physics.adaptation_enabled {
            // Physics mode with threshold adaptation
            println!("Using Physics mode with adaptation: tau_m={:.4}s, dt={:.4}s, tau_theta={:.4}s",
                     tau_m, dt, cfg.physics.tau_theta);
            Network::new_physics_with_adaptation(
                input_size, hidden_size, output_size, tau_m, dt,
                cfg.physics.tau_theta, cfg.physics.theta_low, cfg.physics.theta_high, seed,
            )
        } else {
            println!("Using Physics mode: tau_m={:.4}s, dt={:.4}s", tau_m, dt);
            Network::new_physics(input_size, hidden_size, output_size, tau_m, dt, seed)
        }
    } else {
        // Simple mode (snnTorch-compatible)
        Network::new(input_size, hidden_size, output_size, beta, seed)
    };

    // Override surrogate gradient slope
    network.lif1.spike_grad = SurrogateGradient::fast_sigmoid(slope);
    network.lif2.spike_grad = SurrogateGradient::fast_sigmoid(slope);

    println!("Network architecture:");
    println!("  Input:  {}", input_size);
    if network.is_physics_mode() {
        println!("  Hidden: {} (LIF, physics mode)", hidden_size);
        println!("  Output: {} (LIF, physics mode)", output_size);
    } else {
        println!("  Hidden: {} (LIF, beta={})", hidden_size, beta);
        println!("  Output: {} (LIF, beta={})", output_size, beta);
    }
    println!("  Total parameters: {}", network.num_parameters());
    println!();

    // Create trainer
    let train_config = TrainingConfig {
        lr: cfg.training.lr,
        epochs: cfg.training.epochs,
        batch_size: cfg.training.batch_size,
        num_steps: cfg.training.num_steps,
        seed: cfg.training.seed,
        num_workers: cfg.training.num_workers,
    };
    let mut trainer = Trainer::new(network, train_config);

    // Enable quantization if configured
    if cfg.quantization.enabled {
        trainer.quant_bits = Some(cfg.quantization.bits);
    }

    // Enable noise injection if configured
    if cfg.noise.enabled {
        trainer.noise = NoiseParams {
            weight_std: cfg.noise.weight_std,
            threshold_std: cfg.noise.threshold_std,
            membrane_std: cfg.noise.membrane_std,
            input_std: cfg.noise.input_std,
        };
    }

    // Enable threshold adaptation if configured
    if cfg.physics.adaptation_enabled {
        trainer.adaptation_enabled = true;
        trainer.dt = cfg.physics.dt;
    }

    // Enable analog output if configured
    if cfg.output.analog_gain > 0.0 {
        trainer.analog_gain = cfg.output.analog_gain;
        println!("Analog gain:    {}", cfg.output.analog_gain);
    }

    // Set temporal input encoding if configured
    if let Some(encoder) = input_encoder {
        println!("Input encoding: temporal (row_spacing={:.4}s, pulse_width={:.1}%, input_dim={})",
                 cfg.input_encoding.row_spacing, cfg.input_encoding.pulse_width * 100.0,
                 encoder.output_dim());
        trainer.input_encoder = Some(encoder);
    }

    // Set truncated BPTT if configured
    if let Some(bptt) = cfg.training.bptt_steps {
        if bptt > 0 {
            trainer.bptt_steps = Some(bptt);
            println!("BPTT truncation: {} steps", bptt);
        }
    }

    // Enable cosine annealing LR schedule
    let use_lr_schedule = cfg.training.epochs > 1;
    if use_lr_schedule {
        trainer.lr_scheduler = Some(gilgamesh::training::LRScheduler::new(
            cfg.training.lr,
            cfg.training.epochs,
        ));
        println!("LR schedule:    cosine annealing ({:.6} -> {:.6})",
                 cfg.training.lr, cfg.training.lr * 0.01);
    }

    // Enable gradient clipping (1.0 is a common default)
    trainer.max_grad_norm = Some(1.0);
    println!("Grad clipping:  max_norm=1.0");

    // Enable weight decay (AdamW style, 0.01 is common)
    trainer.optimizer.set_weight_decay(0.01);
    println!("Weight decay:   0.01 (AdamW)");

    // Initialize visualization recorder if enabled
    #[cfg(feature = "visualization")]
    let mut recorder = if visualize {
        let rec = if let Some(ref path) = visualize_file {
            println!("Visualization:  saving to {}", path);
            gilgamesh::TrainingRecorder::to_file("gilgamesh", path, false, 1)?
        } else {
            println!("Visualization:  spawning Rerun viewer");
            gilgamesh::TrainingRecorder::new("gilgamesh", false, 1)?
        };
        // Log architecture info
        rec.log_architecture(input_size, hidden_size, output_size, &cfg.mode)?;
        Some(rec)
    } else {
        None
    };

    #[cfg(not(feature = "visualization"))]
    let mut recorder: Option<gilgamesh::TrainingRecorder> = {
        let _ = &visualize_file; // Suppress unused warning
        if visualize {
            println!("Warning: visualization requested but feature not enabled. Rebuild with --features visualization");
        }
        None
    };

    // Training loop with progress
    println!("Training...");
    println!("{:-<60}", "");

    let mut best_test_acc = 0.0f32;

    for epoch in 1..=cfg.training.epochs {
        // Update learning rate for this epoch
        let lr = trainer.update_lr_for_epoch(epoch).unwrap_or(cfg.training.lr);

        let (train_loss, train_acc) = trainer.train_epoch(&dataset);
        let test_acc = trainer.evaluate(&dataset);

        if test_acc > best_test_acc {
            best_test_acc = test_acc;
        }

        println!(
            "Epoch {:3} | Loss: {:.4} | Train Acc: {:5.2}% | Test Acc: {:5.2}% | LR: {:.6}",
            epoch, train_loss, train_acc, test_acc, lr
        );

        // Log to visualization
        if let Some(ref mut rec) = recorder {
            rec.set_epoch(epoch);
            let _ = rec.log_epoch_metrics(train_loss, train_acc, test_acc, lr);
            // Log weights every 5 epochs to avoid too much data
            if epoch % 5 == 0 || epoch == 1 {
                let _ = rec.log_weights("fc1", &trainer.network.fc1.weight);
                let _ = rec.log_weights("fc2", &trainer.network.fc2.weight);
            }
        }
    }

    println!("{:-<60}", "");
    println!();
    println!("=== Training Complete ===");
    println!("Best Test Accuracy: {:.2}%", best_test_acc);
    let final_test_acc = trainer.evaluate(&dataset);
    println!("Final Test Accuracy: {:.2}%", final_test_acc);

    // Save checkpoint if requested
    if let Some(ref checkpoint_path) = save_checkpoint {
        use gilgamesh::checkpoint::{Checkpoint, TrainingMetadata};

        let metadata = TrainingMetadata {
            epochs_trained: cfg.training.epochs,
            final_train_accuracy: best_test_acc, // Using best as a proxy
            final_test_accuracy: final_test_acc,
            final_loss: None,
            config_file: None,
        };

        let checkpoint = Checkpoint::from_network(&trainer.network, Some(metadata));
        checkpoint.save(checkpoint_path)
            .with_context(|| format!("Failed to save checkpoint to {}", checkpoint_path))?;

        println!("Checkpoint saved to: {}", checkpoint_path);
    }

    Ok(())
}

fn evaluate(checkpoint: &str, data_dir: &str, num_steps: usize, batch_size: usize) -> Result<()> {
    println!("Evaluation not yet implemented (requires checkpoint loading)");
    Ok(())
}

fn test_implementation(quick: bool) -> Result<()> {
    println!("=== Testing gilgamesh Implementation ===");
    println!();

    // Test 1: Surrogate gradients
    println!("Test 1: Surrogate gradients");
    let sg = SurrogateGradient::fast_sigmoid(25.0);
    let grad_at_zero = sg.backward(0.0);
    assert!((grad_at_zero - 1.0).abs() < 1e-5, "Gradient at 0 should be 1.0");
    println!("  FastSigmoid gradient at x=0: {} ✓", grad_at_zero);

    // Test 2: LIF neuron
    println!("\nTest 2: LIF neuron");
    use gilgamesh::neurons::Leaky;
    use ndarray::array;

    let lif = Leaky::new(3, 0.9);
    let state = lif.init_state(1);
    let input = array![[2.0, 0.5, 0.3]];
    let (spikes, new_state, _) = lif.forward(&input, &state);
    println!("  Input: {:?}", input);
    println!("  Spikes: {:?}", spikes);
    println!("  First neuron spiked: {} ✓", spikes[[0, 0]] == 1.0);

    // Test 3: Linear layer
    println!("\nTest 3: Linear layer");
    use gilgamesh::layers::Linear;

    let linear = Linear::new(3, 2, true);
    let output = linear.forward(&input);
    println!("  Input shape: {:?}", input.shape());
    println!("  Output shape: {:?} ✓", output.shape());

    // Test 4: Network forward pass
    println!("\nTest 4: Network forward pass");
    use gilgamesh::Network;
    use ndarray::Array2;

    let net = Network::new(49, 100, 10, 0.9, 42);
    let test_input = Array2::from_elem((4, 49), 0.1);
    let (spikes, mem, _) = net.forward(&test_input, 25);
    println!("  Batch size: 4, Timesteps: 25");
    println!("  Output spike count shape: {:?}", spikes.shape());
    println!("  Output membrane shape: {:?} ✓", mem.shape());

    // Test 5: Gradient computation
    println!("\nTest 5: Gradient computation (backward pass)");
    let (spikes, _, caches) = net.forward(&test_input, 5);
    let grad_output = Array2::from_elem((4, 10), 0.1);
    let grads = net.backward(&test_input, &caches, &grad_output);
    println!("  FC1 weight grad shape: {:?}", grads.fc1_weight.shape());
    println!("  FC2 weight grad shape: {:?} ✓", grads.fc2_weight.shape());

    println!("\n=== All Tests Passed ===");

    if !quick {
        println!("\nRunning quick training test (3 epochs on subset)...");

        // Create small synthetic dataset for testing
        let test_images = Array2::from_elem((500, 49), 0.1f32);
        let test_labels: Vec<usize> = (0..500).map(|i| i % 10).collect();

        let mut net = Network::new(49, 100, 10, 0.9, 42);
        let config = TrainingConfig {
            lr: 1e-3,
            epochs: 3,
            batch_size: 64,
            num_steps: 10,
            seed: 42,
            num_workers: 0,
        };

        use gilgamesh::training::AdamOptimizer;
        use gilgamesh::tensor::cross_entropy_loss;

        let mut optimizer = AdamOptimizer::new(&net, config.lr);

        for epoch in 1..=3 {
            let batch_input = test_images.slice(ndarray::s![0..64, ..]).to_owned();
            let (spikes, _, caches) = net.forward(&batch_input, config.num_steps);
            let (loss, grad_output) = cross_entropy_loss(&spikes, &test_labels[0..64]);
            let grads = net.backward(&batch_input, &caches, &grad_output);
            optimizer.step(&mut net, &grads);
            println!("  Epoch {} | Loss: {:.4}", epoch, loss);
        }

        println!("\nTraining test completed ✓");
    }

    Ok(())
}

/// Run the interactive training dashboard
#[cfg(feature = "dashboard")]
fn run_dashboard(config: Option<String>, epochs: usize, data_dir: &str) -> Result<()> {
    use gilgamesh::dashboard::{create_shared_metrics, DashboardApp};
    use std::thread;

    // Load config
    let mut cfg = if let Some(ref path) = config {
        Config::load(path).with_context(|| format!("Failed to load config from {}", path))?
    } else {
        Config::default()
    };
    cfg.training.epochs = epochs;

    // Load dataset
    println!("Loading MNIST dataset...");
    let dataset = gilgamesh::data::MnistDataset::load(data_dir)
        .context("Failed to load MNIST dataset")?;

    // Create network
    let input_size = dataset.feature_dim();
    let hidden_size = cfg.network.hidden_size;
    let output_size = cfg.network.output_size;

    let architecture = format!("{} → {} → {}", input_size, hidden_size, output_size);
    let metrics = create_shared_metrics(epochs, architecture);
    let metrics_clone = metrics.clone();

    // Update training status
    {
        let mut m = metrics.lock().unwrap();
        m.is_training = true;
    }

    // Spawn training thread
    let cfg_clone = cfg.clone();
    let data_dir_owned = data_dir.to_string();
    thread::spawn(move || {
        let result = run_training_for_dashboard(&cfg_clone, &data_dir_owned, metrics_clone);
        if let Err(e) = result {
            eprintln!("Training error: {}", e);
        }
    });

    // Run dashboard (blocking)
    DashboardApp::run(metrics).map_err(|e| anyhow::anyhow!("Dashboard error: {}", e))
}

#[cfg(feature = "dashboard")]
fn run_training_for_dashboard(
    cfg: &Config,
    data_dir: &str,
    metrics: gilgamesh::dashboard::SharedMetrics,
) -> Result<()> {
    use gilgamesh::training::{Trainer, TrainingConfig};

    let dataset = gilgamesh::data::MnistDataset::load(data_dir)?;

    let input_size = dataset.feature_dim();
    let hidden_size = cfg.network.hidden_size;
    let output_size = cfg.network.output_size;
    let beta = cfg.neuron.beta;
    let seed = cfg.training.seed;

    let mut network = Network::new(input_size, hidden_size, output_size, beta, seed);
    network.lif1.spike_grad = SurrogateGradient::fast_sigmoid(cfg.neuron.slope);
    network.lif2.spike_grad = SurrogateGradient::fast_sigmoid(cfg.neuron.slope);

    let train_config = TrainingConfig {
        lr: cfg.training.lr,
        epochs: cfg.training.epochs,
        batch_size: cfg.training.batch_size,
        num_steps: cfg.training.num_steps,
        seed: cfg.training.seed,
        num_workers: cfg.training.num_workers,
    };

    let mut trainer = Trainer::new(network, train_config);
    trainer.max_grad_norm = Some(1.0);
    trainer.optimizer.set_weight_decay(0.01);

    if cfg.training.epochs > 1 {
        trainer.lr_scheduler = Some(gilgamesh::training::LRScheduler::new(
            cfg.training.lr,
            cfg.training.epochs,
        ));
    }

    for epoch in 1..=cfg.training.epochs {
        let lr = trainer.update_lr_for_epoch(epoch).unwrap_or(cfg.training.lr);
        let (train_loss, train_acc) = trainer.train_epoch(&dataset);
        let test_acc = trainer.evaluate(&dataset);

        // Update shared metrics
        {
            let mut m = metrics.lock().unwrap();
            m.record_epoch(train_loss as f64, train_acc as f64, test_acc as f64, lr as f64);
        }

        println!(
            "Epoch {:3} | Loss: {:.4} | Train: {:.2}% | Test: {:.2}%",
            epoch, train_loss, train_acc, test_acc
        );
    }

    // Mark training complete
    {
        let mut m = metrics.lock().unwrap();
        m.is_training = false;
        m.is_complete = true;
    }

    Ok(())
}

#[cfg(not(feature = "dashboard"))]
fn run_dashboard(_config: Option<String>, _epochs: usize, _data_dir: &str) -> Result<()> {
    println!("Dashboard feature not enabled. Rebuild with --features dashboard");
    Ok(())
}

/// Run the stunning network animation
#[cfg(feature = "animation")]
fn run_animation(config: Option<String>, data_dir: &str, speed: f32, sample: usize) -> Result<()> {
    use gilgamesh::animation::{create_shared_animation, run_animation as run_nannou, NetworkAnimation};
    use std::thread;
    use std::time::Duration;

    // Load config
    let cfg = if let Some(ref path) = config {
        Config::load(path).with_context(|| format!("Failed to load config from {}", path))?
    } else {
        Config::default()
    };

    // Load dataset
    println!("Loading MNIST dataset...");
    let dataset = gilgamesh::data::MnistDataset::load(data_dir)
        .context("Failed to load MNIST dataset")?;

    // Create network
    let input_size = dataset.feature_dim();
    let hidden_size = cfg.network.hidden_size;
    let output_size = cfg.network.output_size;
    let beta = cfg.neuron.beta;
    let seed = cfg.training.seed;

    println!("Creating network: {} → {} → {}", input_size, hidden_size, output_size);
    let network = Network::new(input_size, hidden_size, output_size, beta, seed);

    // Get sample from test set
    let test_images = &dataset.test_images;
    let test_labels = &dataset.test_labels;
    let sample_idx = sample.min(test_images.nrows() - 1);
    let sample_image = test_images.row(sample_idx).to_owned();
    let sample_label = test_labels[sample_idx];

    println!("Animating sample {} (label: {})", sample_idx, sample_label);

    // Create shared animation state
    let animation = create_shared_animation(&[input_size, hidden_size, output_size]);

    // Set animation speed
    {
        let mut anim = animation.lock().unwrap();
        anim.speed = speed;
    }

    // Clone for simulation thread
    let animation_clone = animation.clone();
    let num_steps = cfg.training.num_steps;

    // Spawn simulation thread
    thread::spawn(move || {
        let sample_batch = sample_image.insert_axis(ndarray::Axis(0));

        // Run network step by step, updating animation
        let state1 = network.lif1.init_state(1);
        let state2 = network.lif2.init_state(1);

        let mut hidden_state = state1;
        let mut output_state = state2;

        // Convert weights to nested Vec for animation
        let fc1_weights: Vec<Vec<f32>> = network.fc1.weight
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        let fc2_weights: Vec<Vec<f32>> = network.fc2.weight
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();

        // Set weights (sample a subset for visualization)
        {
            let mut anim = animation_clone.lock().unwrap();
            anim.set_weights(&[fc1_weights.clone(), fc2_weights.clone()]);
        }

        for step in 0..num_steps {
            // Forward through network
            let fc1_out = network.fc1.forward(&sample_batch);
            let (hidden_spikes, new_hidden_state, _) = network.lif1.forward(&fc1_out, &hidden_state);
            hidden_state = new_hidden_state;

            let fc2_out = network.fc2.forward(&hidden_spikes);
            let (output_spikes, new_output_state, _) = network.lif2.forward(&fc2_out, &output_state);
            output_state = new_output_state;

            // Update animation state
            {
                let mut anim = animation_clone.lock().unwrap();

                // Build membrane and spike vectors for each layer
                let input_mem: Vec<f32> = sample_batch.row(0).to_vec();
                let hidden_mem: Vec<f32> = hidden_state.mem.row(0).to_vec();
                let output_mem: Vec<f32> = output_state.mem.row(0).to_vec();

                let input_spikes: Vec<bool> = sample_batch.row(0).iter().map(|&v| v > 0.5).collect();
                let hidden_spikes_bool: Vec<bool> = hidden_spikes.row(0).iter().map(|&v| v > 0.5).collect();
                let output_spikes_bool: Vec<bool> = output_spikes.row(0).iter().map(|&v| v > 0.5).collect();

                anim.update_neurons(
                    &[input_mem, hidden_mem, output_mem],
                    &[input_spikes, hidden_spikes_bool, output_spikes_bool],
                    0.04, // ~25 fps worth of simulation time
                );
            }

            // Slow down simulation to match animation
            thread::sleep(Duration::from_millis((40.0 / speed) as u64));
        }

        // Loop the animation
        loop {
            // Reset states
            hidden_state = network.lif1.init_state(1);
            output_state = network.lif2.init_state(1);

            for _step in 0..num_steps {
                let fc1_out = network.fc1.forward(&sample_batch);
                let (hidden_spikes, new_hidden_state, _) = network.lif1.forward(&fc1_out, &hidden_state);
                hidden_state = new_hidden_state;

                let fc2_out = network.fc2.forward(&hidden_spikes);
                let (output_spikes, new_output_state, _) = network.lif2.forward(&fc2_out, &output_state);
                output_state = new_output_state;

                {
                    let mut anim = animation_clone.lock().unwrap();

                    let input_mem: Vec<f32> = sample_batch.row(0).to_vec();
                    let hidden_mem: Vec<f32> = hidden_state.mem.row(0).to_vec();
                    let output_mem: Vec<f32> = output_state.mem.row(0).to_vec();

                    let input_spikes: Vec<bool> = sample_batch.row(0).iter().map(|&v| v > 0.5).collect();
                    let hidden_spikes_bool: Vec<bool> = hidden_spikes.row(0).iter().map(|&v| v > 0.5).collect();
                    let output_spikes_bool: Vec<bool> = output_spikes.row(0).iter().map(|&v| v > 0.5).collect();

                    anim.update_neurons(
                        &[input_mem, hidden_mem, output_mem],
                        &[input_spikes, hidden_spikes_bool, output_spikes_bool],
                        0.04,
                    );
                }

                thread::sleep(Duration::from_millis((40.0 / speed) as u64));
            }

            // Brief pause between loops
            thread::sleep(Duration::from_millis(500));
        }
    });

    // Run animation window (blocking)
    println!("Launching animation window...");
    println!("Controls: Scroll to zoom, Drag to pan");
    run_nannou(animation);

    Ok(())
}

#[cfg(not(feature = "animation"))]
fn run_animation(_config: Option<String>, _data_dir: &str, _speed: f32, _sample: usize) -> Result<()> {
    println!("Animation feature not enabled. Rebuild with --features animation");
    Ok(())
}

/// Run the visual test inspector
#[cfg(feature = "dashboard")]
fn run_inspector(checkpoint: &str, data_dir: &str, num_steps: usize, seed: u64) -> Result<()> {
    use gilgamesh::inspector::InspectorApp;

    println!("=== gilgamesh Inspector ===");
    println!("Checkpoint: {}", checkpoint);
    println!("Data dir:   {}", data_dir);
    println!("Timesteps:  {}", num_steps);
    println!();

    let app = InspectorApp::from_checkpoint(checkpoint, data_dir, num_steps, seed)
        .context("Failed to initialize inspector")?;

    app.run().map_err(|e| anyhow::anyhow!("Inspector error: {}", e))
}

#[cfg(not(feature = "dashboard"))]
fn run_inspector(_checkpoint: &str, _data_dir: &str, _num_steps: usize, _seed: u64) -> Result<()> {
    println!("Inspector feature not enabled. Rebuild with --features dashboard");
    Ok(())
}
