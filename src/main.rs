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

    /// Launch interactive network visualizer (requires --features animation)
    Animate {
        /// Path to checkpoint file (auto-finds latest if not specified)
        #[arg(long)]
        checkpoint: Option<String>,

        /// Data directory (for loading sample images)
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Animation speed multiplier
        #[arg(long, default_value = "1.0")]
        speed: f32,

        /// Starting sample index
        #[arg(long, default_value = "0")]
        sample: usize,
    },

    /// Visual test runner for inspecting single samples (requires --features dashboard)
    Inspect {
        /// Path to checkpoint file (uses most recent if not specified)
        #[arg(long)]
        checkpoint: Option<String>,

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

    /// Compare gilgamesh simulation with ngspice for hardware validation
    Spice {
        /// Path to checkpoint file (uses most recent if not specified)
        #[arg(long)]
        checkpoint: Option<String>,

        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Sample index from test set (random if not specified)
        #[arg(long)]
        sample: Option<usize>,

        /// Output directory for netlists and results
        #[arg(long, default_value = "./spice_output")]
        output_dir: String,

        /// Run ngspice automatically (requires ngspice in PATH)
        #[arg(long)]
        run_ngspice: bool,

        /// Number of timesteps
        #[arg(long, default_value = "25")]
        num_steps: usize,

        /// Enable analog output stage in SPICE netlist
        #[arg(long)]
        analog_output: bool,

        /// Disable pulse stretching circuit
        #[arg(long)]
        no_pulse_stretch: bool,
    },

    /// Run single-neuron test for SPICE comparison
    NeuronTest {
        /// Membrane time constant (seconds)
        #[arg(long, default_value = "0.0012")]
        tau_m: f32,

        /// Integration timestep (seconds)
        #[arg(long, default_value = "0.000001")]
        dt: f32,

        /// Spike threshold voltage (V)
        #[arg(long, default_value = "3.3")]
        threshold: f32,

        /// Reference voltage (V)
        #[arg(long, default_value = "2.5")]
        vref: f32,

        /// Input current (A)
        #[arg(long, default_value = "0.000001")]
        input_current: f32,

        /// Simulation duration (seconds)
        #[arg(long, default_value = "0.05")]
        duration: f32,

        /// Pulse stretch time constant (seconds, 0 to disable)
        #[arg(long, default_value = "0.00167")]
        tau_pulse: f32,

        /// Output CSV file
        #[arg(long, default_value = "neuron_output.csv")]
        output: String,
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
            checkpoint,
            data_dir,
            speed,
            sample,
        } => run_animation(checkpoint, &data_dir, speed, sample),
        Commands::Inspect {
            checkpoint,
            data_dir,
            num_steps,
            seed,
        } => run_inspector(checkpoint, &data_dir, num_steps, seed),
        Commands::Spice {
            checkpoint,
            data_dir,
            sample,
            output_dir,
            run_ngspice,
            num_steps,
            analog_output,
            no_pulse_stretch,
        } => run_spice(checkpoint, &data_dir, sample, &output_dir, run_ngspice, num_steps, analog_output, !no_pulse_stretch),
        Commands::NeuronTest {
            tau_m,
            dt,
            threshold,
            vref,
            input_current,
            duration,
            tau_pulse,
            output,
        } => run_neuron_test(tau_m, dt, threshold, vref, input_current, duration, tau_pulse, &output),
    }
}

/// Find the most recently modified checkpoint file in common locations
fn find_latest_checkpoint() -> Option<String> {
    use std::fs;
    use std::time::SystemTime;

    let search_patterns = [
        "./*.json",
        "./checkpoints/*.json",
        "./models/*.json",
        "./*.checkpoint.json",
    ];

    let mut candidates: Vec<(String, SystemTime)> = Vec::new();

    for pattern in &search_patterns {
        if let Ok(entries) = glob::glob(pattern) {
            for entry in entries.flatten() {
                if let Ok(metadata) = fs::metadata(&entry) {
                    if let Ok(modified) = metadata.modified() {
                        // Quick check: try to see if it looks like a checkpoint
                        if let Ok(contents) = fs::read_to_string(&entry) {
                            if contents.contains("\"architecture\"") && contents.contains("\"weights\"") {
                                candidates.push((entry.to_string_lossy().to_string(), modified));
                            }
                        }
                    }
                }
            }
        }
    }

    // Also check current directory for any .json that looks like a checkpoint
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(metadata) = fs::metadata(&path) {
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(contents) = fs::read_to_string(&path) {
                            if contents.contains("\"architecture\"") && contents.contains("\"weights\"") {
                                let path_str = path.to_string_lossy().to_string();
                                if !candidates.iter().any(|(p, _)| p == &path_str) {
                                    candidates.push((path_str, modified));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort by modification time (most recent first)
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.into_iter().next().map(|(path, _)| path)
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

fn evaluate(_checkpoint: &str, _data_dir: &str, _num_steps: usize, _batch_size: usize) -> Result<()> {
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
    let (spikes, _new_state, _) = lif.forward(&input, &state);
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
    let (_spikes, _, caches) = net.forward(&test_input, 5);
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

/// Run the interactive network animation with checkpoint loading
#[cfg(feature = "animation")]
fn run_animation(checkpoint: Option<String>, data_dir: &str, speed: f32, start_sample: usize) -> Result<()> {
    use gilgamesh::animation::{create_shared_animation, run_animation as run_nannou};
    use gilgamesh::checkpoint::Checkpoint;
    use std::thread;
    use std::time::Duration;

    println!("=== gilgamesh Interactive Visualizer ===");
    println!();

    // Find checkpoint: use provided path, or find most recent
    let checkpoint_path = match checkpoint {
        Some(path) => path,
        None => {
            match find_latest_checkpoint() {
                Some(path) => {
                    println!("Using most recent checkpoint: {}", path);
                    path
                }
                None => {
                    anyhow::bail!("No checkpoint specified and no checkpoint files found.\n\
                        Train a model first: gilgamesh train --save-checkpoint model.json");
                }
            }
        }
    };

    // Load checkpoint and reconstruct network
    println!("Loading checkpoint: {}", checkpoint_path);
    let cp = Checkpoint::load(&checkpoint_path)
        .with_context(|| format!("Failed to load checkpoint from {}", checkpoint_path))?;
    let network = cp.to_network()
        .with_context(|| "Failed to reconstruct network from checkpoint")?;

    let input_size = cp.architecture.input_size;
    let hidden_size = cp.architecture.hidden_size;
    let output_size = cp.architecture.output_size;

    println!("Network: {} → {} → {} ({})", input_size, hidden_size, output_size, cp.architecture.mode);
    if let Some(ref meta) = cp.metadata {
        println!("Trained: {} epochs, {:.2}% test accuracy", meta.epochs_trained, meta.final_test_accuracy);
    }

    // Load dataset
    println!("Loading MNIST dataset...");
    let dataset = gilgamesh::data::MnistDataset::load(data_dir)
        .context("Failed to load MNIST dataset")?;

    let test_images = dataset.test_images.clone();
    let test_labels = dataset.test_labels.clone();
    let total_samples = test_images.nrows();

    println!("Loaded {} test samples", total_samples);
    println!();
    println!("Controls:");
    println!("  ← / →  : Previous / Next sample");
    println!("  Space  : Pause / Resume");
    println!("  R      : Restart current sample");
    println!();

    // Create shared animation state
    let animation = create_shared_animation(&[input_size, hidden_size, output_size]);

    // Initialize animation state
    let initial_sample = start_sample.min(total_samples - 1);
    {
        let mut anim = animation.lock().unwrap();
        anim.speed = speed;
        anim.viewer.total_samples = total_samples;
        anim.viewer.total_steps = 25; // Default timesteps

        // Set initial sample
        let image = test_images.row(initial_sample).to_vec();
        anim.reset_for_sample(initial_sample, test_labels[initial_sample] as u8, &image);
    }

    // Convert weights to nested Vec for animation (once)
    let fc1_weights: Vec<Vec<f32>> = network.fc1.weight
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let fc2_weights: Vec<Vec<f32>> = network.fc2.weight
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();

    {
        let mut anim = animation.lock().unwrap();
        anim.set_weights(&[fc1_weights, fc2_weights]);
    }

    // Clone for simulation thread
    let animation_clone = animation.clone();
    let num_steps = 25usize;

    // Spawn simulation thread that handles sample navigation
    thread::spawn(move || {
        let mut current_sample = initial_sample;

        loop {
            // Check for sample change requests
            let (should_change, new_sample) = {
                let mut anim = animation_clone.lock().unwrap();
                if let Some(delta) = anim.viewer.sample_request.take() {
                    let new_idx = if delta == 0 {
                        // Restart current
                        current_sample
                    } else {
                        // Navigate
                        let new = current_sample as i32 + delta;
                        new.clamp(0, (total_samples - 1) as i32) as usize
                    };
                    (true, new_idx)
                } else {
                    (false, current_sample)
                }
            };

            if should_change || current_sample != new_sample {
                current_sample = new_sample;

                // Reset for new sample
                let image = test_images.row(current_sample).to_vec();
                let label = test_labels[current_sample] as u8;

                {
                    let mut anim = animation_clone.lock().unwrap();
                    anim.reset_for_sample(current_sample, label, &image);
                }
            }

            // Run simulation for current sample
            let sample_image = test_images.row(current_sample).to_owned();
            let sample_batch = sample_image.insert_axis(ndarray::Axis(0));

            // Reset step counter at start of each simulation run
            {
                let mut anim = animation_clone.lock().unwrap();
                anim.viewer.current_step = 0;
                anim.viewer.simulation_complete = false;
            }

            let mut hidden_state = network.lif1.init_state(1);
            let mut output_state = network.lif2.init_state(1);

            for step in 0..num_steps {
                // Check if we need to switch samples mid-simulation
                {
                    let anim = animation_clone.lock().unwrap();
                    if anim.viewer.sample_request.is_some() {
                        break; // Exit loop to handle sample change
                    }
                    if anim.paused {
                        // When paused, just wait
                        drop(anim);
                        thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                }

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

                    // Set step explicitly (don't rely on update_neurons incrementing)
                    anim.viewer.current_step = step + 1;

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

            // Mark simulation complete for this sample
            {
                let mut anim = animation_clone.lock().unwrap();
                anim.viewer.simulation_complete = true;
            }

            // Brief pause before looping/checking for new sample
            thread::sleep(Duration::from_millis(300));
        }
    });

    // Run animation window (blocking)
    println!("Launching visualization window...");
    run_nannou(animation);

    Ok(())
}

#[cfg(not(feature = "animation"))]
fn run_animation(_checkpoint: Option<String>, _data_dir: &str, _speed: f32, _sample: usize) -> Result<()> {
    println!("Animation feature not enabled. Rebuild with --features animation");
    Ok(())
}

/// Run the visual test inspector
#[cfg(feature = "dashboard")]
fn run_inspector(checkpoint: Option<String>, data_dir: &str, num_steps: usize, seed: u64) -> Result<()> {
    use gilgamesh::inspector::InspectorApp;

    // Find checkpoint: use provided path, or find most recent
    let checkpoint_path = match checkpoint {
        Some(path) => path,
        None => {
            match find_latest_checkpoint() {
                Some(path) => {
                    println!("Using most recent checkpoint: {}", path);
                    path
                }
                None => {
                    anyhow::bail!("No checkpoint specified and no checkpoint files found.\n\
                        Train a model first: gilgamesh train --save-checkpoint model.json");
                }
            }
        }
    };

    println!("=== gilgamesh Inspector ===");
    println!("Checkpoint: {}", checkpoint_path);
    println!("Data dir:   {}", data_dir);
    println!("Timesteps:  {}", num_steps);
    println!();

    let app = InspectorApp::from_checkpoint(&checkpoint_path, data_dir, num_steps, seed)
        .context("Failed to initialize inspector")?;

    app.run().map_err(|e| anyhow::anyhow!("Inspector error: {}", e))
}

#[cfg(not(feature = "dashboard"))]
fn run_inspector(_checkpoint: Option<String>, _data_dir: &str, _num_steps: usize, _seed: u64) -> Result<()> {
    println!("Inspector feature not enabled. Rebuild with --features dashboard");
    Ok(())
}

/// Run SPICE comparison
fn run_spice(
    checkpoint: Option<String>,
    data_dir: &str,
    sample: Option<usize>,
    output_dir: &str,
    should_run_ngspice: bool,
    num_steps: usize,
    analog_output: bool,
    pulse_stretch: bool,
) -> Result<()> {
    use gilgamesh::checkpoint::Checkpoint;
    use gilgamesh::data::MnistDataset;
    use gilgamesh::spice::{ComparisonResult, SpiceNetlist, SpiceParams, run_ngspice};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::fs;
    use std::path::Path;

    println!("=== gilgamesh SPICE Comparison (Detailed Pulse-Stretch Model) ===");
    println!();

    // Find checkpoint: use provided path, find most recent, or use untrained network
    let checkpoint_path = match checkpoint {
        Some(path) => Some(path),
        None => {
            if let Some(path) = find_latest_checkpoint() {
                println!("Using most recent checkpoint: {}", path);
                Some(path)
            } else {
                None
            }
        }
    };

    // Load or create network
    let network = if let Some(ref path) = checkpoint_path {
        println!("Loading checkpoint: {}", path);
        let cp = Checkpoint::load(path)
            .with_context(|| format!("Failed to load checkpoint from {}", path))?;
        cp.to_network()
            .with_context(|| "Failed to reconstruct network from checkpoint")?
    } else {
        println!("No checkpoint found, using untrained network");
        Network::new(49, 100, 10, 0.9, 42)
    };

    // Load dataset
    println!("Loading MNIST dataset from {}...", data_dir);
    let dataset = MnistDataset::load(data_dir)
        .context("Failed to load MNIST dataset")?;

    // Select sample
    let sample_idx = sample.unwrap_or_else(|| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        use rand::Rng;
        rng.gen_range(0..dataset.test_len())
    });
    let sample_idx = sample_idx.min(dataset.test_len() - 1);

    let (images, labels) = dataset.get_test_batch(&[sample_idx]);
    let input = images.row(0).to_owned();
    let label = labels[0];

    println!("Selected sample: {} (true label: {})", sample_idx, label);
    println!();

    // Build physics parameters with user options
    let params = SpiceParams::from_network(&network)
        .with_analog_output(analog_output)
        .with_pulse_stretch(pulse_stretch);

    println!("Circuit parameters:");
    println!("  Supply:        VDD={:.1}V, Vref={:.1}V", params.supply.vdd, params.supply.vref);
    println!("  Membrane:      C={:.3e}F, R={:.3e}Ω, tau={:.2}ms",
             params.membrane.c_mem, params.membrane.r_leak, params.tau_m() * 1000.0);
    println!("  Threshold:     Vref+{:.2}V, hysteresis={:.3}V",
             params.threshold.over_vref, params.threshold.hysteresis);
    println!("  Pulse stretch: {} (tau={:.2}ms)",
             if pulse_stretch { "enabled" } else { "disabled" },
             params.tau_pulse() * 1000.0);
    println!("  Analog output: {}", if analog_output { "enabled" } else { "disabled" });
    println!();

    // Run gilgamesh simulation
    println!("Running gilgamesh simulation ({} steps)...", num_steps);
    let input_batch = input.clone().insert_axis(ndarray::Axis(0));
    let trace = network.forward_traced(&input_batch, num_steps);

    let gilgamesh_spikes: Vec<f32> = trace.output_spike_count.row(0).to_vec();
    let gilgamesh_pred = gilgamesh_spikes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("Gilgamesh prediction: {} (correct: {})", gilgamesh_pred, gilgamesh_pred == label);

    // Create output directory
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

    // Generate SPICE netlist
    println!();
    println!("Generating SPICE netlist (detailed pulse-stretch model)...");
    let netlist = SpiceNetlist::from_network(&network, input.as_slice().unwrap(), num_steps, &params);

    let netlist_path = output_path.join("gilgamesh.cir");
    netlist.write(&netlist_path)?;
    println!("Netlist written to: {:?}", netlist_path);

    // Optionally run ngspice
    if should_run_ngspice {
        println!();
        println!("Running ngspice...");

        match run_ngspice(&netlist_path, output_path) {
            Ok(spice_output) => {
                println!("SPICE simulation complete");
                println!();

                // Compare results
                let comparison = ComparisonResult::compare(&trace, &spice_output, &params);
                comparison.print_summary();
            }
            Err(e) => {
                println!("Error running ngspice: {}", e);
                println!("Make sure ngspice is installed and in PATH");
                println!();
                println!("To run manually:");
                println!("  cd {} && ngspice -b gilgamesh.cir", output_dir);
            }
        }
    } else {
        println!();
        println!("Netlist generated. To run simulation:");
        println!("  cd {} && ngspice -b gilgamesh.cir", output_dir);
        println!();
        println!("Or use --run-ngspice flag to run automatically");
    }

    Ok(())
}

/// Run single-neuron test for SPICE comparison
fn run_neuron_test(
    tau_m: f32,
    dt: f32,
    threshold: f32,
    vref: f32,
    input_current: f32,
    duration: f32,
    tau_pulse: f32,
    output_path: &str,
) -> Result<()> {
    use gilgamesh::neurons::Leaky;
    use std::fs::File;
    use std::io::Write;

    // Circuit parameters (matching SPICE defaults)
    let c_mem = 10e-9; // 10 nF membrane capacitance
    let r_leak = tau_m / c_mem; // Leak resistance from tau = RC

    println!("=== Single Neuron Test ===");
    println!();
    println!("Parameters:");
    println!("  tau_m      = {:.4} ms", tau_m * 1000.0);
    println!("  dt         = {:.4} us", dt * 1e6);
    println!("  threshold  = {:.3} V (over vref)", threshold - vref);
    println!("  vref       = {:.3} V", vref);
    println!("  input      = {:.4} uA", input_current * 1e6);
    println!("  duration   = {:.2} ms", duration * 1000.0);
    println!("  tau_pulse  = {:.4} ms", tau_pulse * 1000.0);
    println!("  C_mem      = {:.1} nF", c_mem * 1e9);
    println!("  R_leak     = {:.1} kΩ", r_leak / 1000.0);
    println!();

    // Create a single LIF neuron with physics mode
    // Threshold is relative to ground (membrane starts at 0 internally)
    let over_vref_threshold = threshold - vref; // e.g., 3.3 - 2.5 = 0.8V over vref
    let neuron = if tau_pulse > 0.0 {
        Leaky::new_physics_with_pulse(1, tau_m, dt, tau_pulse, 5.0)
            .with_threshold(over_vref_threshold)
    } else {
        Leaky::new_physics(1, tau_m, dt)
            .with_threshold(over_vref_threshold)
    };

    let num_steps = (duration / dt) as usize;
    println!("Running {} timesteps...", num_steps);

    // Open output file
    let mut file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {}", output_path))?;

    // Write CSV header
    writeln!(file, "time,membrane,spike,pulse")?;

    // Initialize state
    let mut state = neuron.init_state(1);
    let mut spike_count = 0;
    let mut pulse_value = 0.0f32;

    // For passive RC membrane (matching real neuromorphic hardware):
    // dV/dt = I/C - V/tau
    // Per timestep: dV = (I * dt) / C
    //
    // Steady-state: V_ss = I * R = I * tau / C
    // With I=1µA, tau=1.2ms, C=10nF: V_ss = 0.12V
    //
    // This matches the passive SPICE model (no op-amp TIA).
    // For TIA-based SPICE (with op-amp gain), the effective transimpedance
    // is ~20x higher, but that's not realistic for ASIC/PCB implementation.
    let input_per_step = (input_current * dt) / c_mem;
    println!("  Input/step = {:.6} V", input_per_step);
    println!();

    // Create input tensor (single neuron, single batch)
    let input = ndarray::array![[input_per_step]];

    for step in 0..num_steps {
        let time = step as f32 * dt;

        // Forward pass
        let (spikes, new_state, _) = neuron.forward(&input, &state);

        // Update pulse decay (simple exponential for now)
        if spikes[[0, 0]] > 0.5 {
            pulse_value = 5.0; // VDD
            spike_count += 1;
        } else if tau_pulse > 0.0 {
            let decay = (-dt / tau_pulse).exp();
            pulse_value *= decay;
        }

        // Get membrane voltage (ground-referenced, matching SPICE v(mem))
        // Internal membrane is relative to virtual ground, same as SPICE
        let membrane = state.mem[[0, 0]];

        // Write to CSV
        writeln!(file, "{:.9},{:.6},{:.1},{:.6}", time, membrane, spikes[[0, 0]], pulse_value)?;

        state = new_state;
    }

    println!();
    println!("Simulation complete:");
    println!("  Total spikes: {}", spike_count);
    println!("  Output: {}", output_path);

    Ok(())
}
