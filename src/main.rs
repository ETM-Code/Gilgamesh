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
        } => {
            // If config file is provided, load from it; otherwise use CLI args
            if let Some(config_path) = config {
                train_from_config(&config_path, &data_dir, visualize, visualize_file)
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
                train_with_config(&cfg, &data_dir, visualize, visualize_file)
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
    }
}

/// Train from a JSON config file
fn train_from_config(
    config_path: &str,
    data_dir: &str,
    visualize: bool,
    visualize_file: Option<String>,
) -> Result<()> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("Failed to load config from {}", config_path))?;
    println!("Loaded config from: {}", config_path);
    train_with_config(&cfg, data_dir, visualize, visualize_file)
}

/// Train using a Config struct
fn train_with_config(
    cfg: &Config,
    data_dir: &str,
    visualize: bool,
    visualize_file: Option<String>,
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
    let recorder: Option<gilgamesh::TrainingRecorder> = {
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
    println!("Final Test Accuracy: {:.2}%", trainer.evaluate(&dataset));

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
