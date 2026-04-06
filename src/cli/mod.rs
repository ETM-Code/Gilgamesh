use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gilgamesh::config::Config;

pub(crate) mod commands;
pub(crate) mod utils;

const DEFAULT_TRAIN_CONFIG_PATH: &str = "configs/tarski_pcb_v7.json";

#[derive(Parser)]
#[command(name = "gilgamesh")]
#[command(about = "Hardware-accurate spiking neural network")]
pub(crate) struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Train a new SNN on MNIST
    Train {
        /// Path to JSON config file (overrides other args)
        #[arg(long)]
        config: Option<String>,

        /// Learning rate
        #[arg(long)]
        lr: Option<f32>,

        /// Number of epochs
        #[arg(long)]
        epochs: Option<usize>,

        /// Batch size
        #[arg(long)]
        batch_size: Option<usize>,

        /// Number of timesteps
        #[arg(long)]
        num_steps: Option<usize>,

        /// Hidden layer size (uses config default if not specified)
        #[arg(long)]
        hidden_size: Option<usize>,

        /// Image size for square images (e.g., 6 for 6x6). Determines input layer size.
        #[arg(long)]
        image_size: Option<usize>,

        /// Image width (for non-square). Use with --image-height.
        #[arg(long)]
        image_width: Option<usize>,

        /// Image height (for non-square). Use with --image-width.
        #[arg(long)]
        image_height: Option<usize>,

        /// Membrane decay (beta)
        #[arg(long)]
        beta: Option<f32>,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,

        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Surrogate gradient slope
        #[arg(long)]
        slope: Option<f32>,

        /// Enable weight quantization (3-bit magnitude for current sources)
        #[arg(long)]
        quantize: bool,

        /// Quantization magnitude bits (default: 3 for hardware current sources)
        #[arg(long)]
        quantize_bits: Option<u8>,

        /// Enable noise injection
        #[arg(long)]
        noise: bool,

        /// Weight noise std (default: 0.05)
        #[arg(long)]
        weight_noise: Option<f32>,

        /// Save checkpoint to this path after training
        #[arg(long)]
        save_checkpoint: Option<String>,
    },

    /// Fine-tune using hardware emulator forward pass
    Finetune {
        /// Pre-trained checkpoint
        #[arg(long)]
        checkpoint: String,
        /// Config file
        #[arg(long)]
        config: String,
        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: String,
        /// Output checkpoint path
        #[arg(long, default_value = "models/finetuned.json")]
        output: String,
        /// DAC scale factor (1.16 is optimal for current PCB)
        #[arg(long, default_value = "1.16")]
        dac_scale: f64,
        /// Number of fine-tuning epochs
        #[arg(long, default_value = "3")]
        epochs: usize,
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

        /// Enable noise injection during evaluation (matches hardware conditions)
        #[arg(long)]
        noise: bool,

        /// Config file for noise/quantization parameters (optional, uses defaults if not set)
        #[arg(long)]
        config: Option<String>,
    },

    /// Run a quick test to verify the implementation
    Test {
        /// Use small dataset for quick testing
        #[arg(long)]
        quick: bool,
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

    /// Generate spike raster plots for correctly classified digits
    Raster {
        /// Path to checkpoint file (uses most recent if not specified)
        #[arg(long)]
        checkpoint: Option<String>,

        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Number of correctly classified samples to include
        #[arg(long, default_value = "6")]
        num_samples: usize,

        /// Number of timesteps per inference
        #[arg(long, default_value = "25")]
        num_steps: usize,

        /// Maximum number of test samples to scan while searching for correct predictions
        #[arg(long, default_value = "500")]
        max_search: usize,

        /// Output JSON path for raster data
        #[arg(long, default_value = "./artifacts/spike_raster_data.json")]
        output_json: String,

        /// Output image path for rendered raster plot
        #[arg(long, default_value = "./artifacts/spike_raster_plot.png")]
        output_image: String,

        /// Skip plot rendering (only dump JSON)
        #[arg(long)]
        no_plot: bool,

        /// Optional background image path for the plot
        #[arg(long)]
        background_image: Option<String>,

        /// Background image alpha (0-1)
        #[arg(long, default_value = "0.2")]
        bg_alpha: f32,

        /// Output PNG DPI
        #[arg(long, default_value = "200")]
        dpi: usize,
    },

    /// Run single-neuron test for SPICE comparison
    NeuronTest {
        /// Membrane time constant (seconds)
        #[arg(long, default_value = "0.00396")]
        tau_m: f32,

        /// Integration timestep (seconds)
        #[arg(long, default_value = "2.5e-7")]
        dt: f32,

        /// Spike threshold voltage (V)
        #[arg(long, default_value = "0.8")]
        threshold: f32,

        /// Reference voltage (V)
        #[arg(long, default_value = "0.0")]
        vref: f32,

        /// Input current (A)
        #[arg(long, default_value = "0.000010")]
        input_current: f32,

        /// Simulation duration (seconds)
        #[arg(long, default_value = "0.05")]
        duration: f32,

        /// Pulse stretch time constant (seconds, 0 to disable)
        #[arg(long, default_value = "1.5e-6")]
        tau_pulse: f32,

        /// Peak pulse voltage (V). Default 5.0 = VDD.
        /// Use ~4.44 to match SPICE with diode drop.
        #[arg(long, default_value = "4.44")]
        v_peak: f32,

        /// Comparator propagation delay (seconds, 0 to disable)
        /// Based on NCS2250: ~50ns typical
        #[arg(long, default_value = "0")]
        comparator_delay: f32,

        /// Reset hold period (seconds, 0 to disable)
        /// Time membrane is held at reset after spike (models pulse stretcher)
        #[arg(long, default_value = "3.6e-7")]
        reset_hold: f32,

        /// Membrane capacitance (Farads)
        #[arg(long, default_value = "33e-9")]
        c_mem: f32,

        /// Output CSV file
        #[arg(long, default_value = "neuron_output.csv")]
        output: String,
    },

    /// Launch web-based UI (requires --features web)
    Web {
        /// Port to run web server on
        #[arg(short, long, default_value = "3000")]
        port: u16,

        /// Path to checkpoint file (auto-finds latest if not specified)
        #[arg(short, long)]
        checkpoint: Option<String>,

        /// Data directory for MNIST
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Open browser automatically
        #[arg(long, default_value = "true")]
        open: bool,
    },
}

impl Cli {
    pub(crate) fn run(self) -> Result<()> {
        match self.command {
            Commands::Train {
                config,
                lr,
                epochs,
                batch_size,
                num_steps,
                hidden_size,
                image_size,
                image_width,
                image_height,
                beta,
                seed,
                data_dir,
                slope,
                quantize,
                quantize_bits,
                noise,
                weight_noise,
                save_checkpoint,
            } => {
                let config_path = config.as_deref().unwrap_or(DEFAULT_TRAIN_CONFIG_PATH);
                let mut cfg = Config::load(config_path).with_context(|| {
                    format!("Failed to load training config from {}", config_path)
                })?;

                // Override config with explicitly provided CLI args
                if let Some(val) = lr {
                    cfg.training.lr = val;
                }
                if let Some(val) = epochs {
                    cfg.training.epochs = val;
                }
                if let Some(val) = batch_size {
                    cfg.training.batch_size = val;
                }
                if let Some(val) = num_steps {
                    cfg.training.num_steps = val;
                }
                if let Some(val) = seed {
                    cfg.training.seed = val;
                }
                if let Some(val) = hidden_size {
                    cfg.network.hidden_size = val;
                }
                if let (Some(w), Some(h)) = (image_width, image_height) {
                    cfg.network.image_width = Some(w);
                    cfg.network.image_height = Some(h);
                    cfg.network.input_size = w * h;
                } else if let Some(val) = image_size {
                    cfg.network.image_size = val;
                    cfg.network.input_size = val * val;
                }
                if let Some(val) = beta {
                    cfg.neuron.beta = val;
                }
                if let Some(val) = slope {
                    cfg.neuron.slope = val;
                }
                if quantize {
                    cfg.quantization.enabled = true;
                }
                if let Some(val) = quantize_bits {
                    cfg.quantization.bits = val;
                }
                if noise {
                    cfg.noise.enabled = true;
                }
                if let Some(val) = weight_noise {
                    cfg.noise.weight_std = val;
                }

                commands::train_with_config(&cfg, &data_dir, save_checkpoint)
            }
            Commands::Finetune {
                checkpoint,
                config,
                data_dir,
                output,
                dac_scale,
                epochs,
            } => {
                commands::finetune::run_finetune(
                    &std::path::PathBuf::from(checkpoint),
                    &std::path::PathBuf::from(config),
                    &std::path::PathBuf::from(data_dir),
                    &std::path::PathBuf::from(output),
                    dac_scale,
                    epochs,
                );
                Ok(())
            }
            Commands::Evaluate {
                checkpoint,
                data_dir,
                num_steps,
                batch_size,
                noise,
                config,
            } => commands::evaluate(
                &checkpoint,
                &data_dir,
                num_steps,
                batch_size,
                noise,
                config.as_deref(),
            ),
            Commands::Test { quick } => commands::test_implementation(quick),

            Commands::Inspect {
                checkpoint,
                data_dir,
                num_steps,
                seed,
            } => commands::run_inspector(checkpoint, &data_dir, num_steps, seed),
            Commands::Raster {
                checkpoint,
                data_dir,
                num_samples,
                num_steps,
                max_search,
                output_json,
                output_image,
                no_plot,
                background_image,
                bg_alpha,
                dpi,
            } => commands::run_spike_raster(
                checkpoint,
                &data_dir,
                num_samples,
                num_steps,
                max_search,
                &output_json,
                &output_image,
                no_plot,
                background_image,
                bg_alpha,
                dpi,
            ),
            Commands::NeuronTest {
                tau_m,
                dt,
                threshold,
                vref,
                input_current,
                duration,
                tau_pulse,
                v_peak,
                comparator_delay,
                reset_hold,
                c_mem,
                output,
            } => commands::run_neuron_test(
                tau_m,
                dt,
                threshold,
                vref,
                input_current,
                duration,
                tau_pulse,
                v_peak,
                comparator_delay,
                reset_hold,
                c_mem,
                &output,
            ),
            Commands::Web {
                port,
                checkpoint,
                data_dir,
                open,
            } => commands::run_web_server(port, checkpoint, &data_dir, open),
        }
    }
}
