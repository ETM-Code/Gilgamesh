use anyhow::Result;
use clap::{Parser, Subcommand};
use gilgamesh::config::Config;

pub(crate) mod commands;
pub(crate) mod utils;

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

        /// Hidden layer size (uses config default if not specified)
        #[arg(long)]
        hidden_size: Option<usize>,

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

    /// Minimal SPICE harness (1-2 neurons) with Rust comparison
    SpiceMini {
        /// Output directory for netlist and results
        #[arg(long, default_value = "./spice_output_mini")]
        output_dir: String,

        /// Run ngspice automatically (requires ngspice in PATH)
        #[arg(long)]
        run_ngspice: bool,

        /// Use two neurons (pulse -> synapse -> neuron)
        #[arg(long)]
        two_neurons: bool,

        /// Input current into neuron A (Amps)
        #[arg(long, default_value = "0.000010")]
        input_current: f32,

        /// Synapse gain (A/V) for neuron B (optional)
        #[arg(long)]
        synapse_gain: Option<f32>,

        /// Simulation duration (seconds)
        #[arg(long, default_value = "0.005")]
        duration: f32,

        /// SPICE timestep (seconds)
        #[arg(long, default_value = "2.5e-7")]
        dt: f32,

        /// Peak pulse voltage for Rust comparison (V)
        #[arg(long, default_value = "4.44")]
        v_peak: f32,

        /// Enable adaptive injection (uses R_inject instead of disabling it)
        #[arg(long)]
        enable_inject: bool,
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
                if let Some(config_path) = config {
                    commands::train_from_config(
                        &config_path,
                        &data_dir,
                        visualize,
                        visualize_file,
                        save_checkpoint,
                    )
                } else {
                    let mut cfg = Config::default();
                    cfg.training.lr = lr;
                    cfg.training.epochs = epochs;
                    cfg.training.batch_size = batch_size;
                    cfg.training.num_steps = num_steps;
                    cfg.training.seed = seed;
                    if let Some(h) = hidden_size {
                        cfg.network.hidden_size = h;
                    }
                    cfg.neuron.beta = beta;
                    cfg.neuron.slope = slope;
                    cfg.quantization.enabled = quantize;
                    cfg.quantization.bits = quantize_bits;
                    cfg.noise.enabled = noise;
                    cfg.noise.weight_std = weight_noise;
                    commands::train_with_config(
                        &cfg,
                        &data_dir,
                        visualize,
                        visualize_file,
                        save_checkpoint,
                    )
                }
            }
            Commands::Evaluate {
                checkpoint,
                data_dir,
                num_steps,
                batch_size,
            } => commands::evaluate(&checkpoint, &data_dir, num_steps, batch_size),
            Commands::Test { quick } => commands::test_implementation(quick),
            Commands::Dashboard {
                config,
                epochs,
                data_dir,
            } => commands::run_dashboard(config, epochs, &data_dir),
            Commands::Animate {
                checkpoint,
                data_dir,
                speed,
                sample,
            } => commands::run_animation(checkpoint, &data_dir, speed, sample),
            Commands::Inspect {
                checkpoint,
                data_dir,
                num_steps,
                seed,
            } => commands::run_inspector(checkpoint, &data_dir, num_steps, seed),
            Commands::Spice {
                checkpoint,
                data_dir,
                sample,
                output_dir,
                run_ngspice,
                num_steps,
                analog_output,
                no_pulse_stretch,
            } => commands::run_spice(
                checkpoint,
                &data_dir,
                sample,
                &output_dir,
                run_ngspice,
                num_steps,
                analog_output,
                !no_pulse_stretch,
            ),
            Commands::SpiceMini {
                output_dir,
                run_ngspice,
                two_neurons,
                input_current,
                synapse_gain,
                duration,
                dt,
                v_peak,
                enable_inject,
            } => commands::run_spice_mini(
                &output_dir,
                run_ngspice,
                two_neurons,
                input_current,
                synapse_gain,
                duration,
                dt,
                v_peak,
                enable_inject,
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
