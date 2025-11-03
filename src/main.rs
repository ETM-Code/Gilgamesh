use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};

use gilgamesh::spice_comparator::{run_comparison, ComparisonConfig};

#[derive(Parser, Debug)]
#[command(name = "gilgamesh", version, about = "Neuron SPICE comparator toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Compare {
        #[arg(long)]
        network: PathBuf,
        #[arg(long)]
        spice_csv: PathBuf,
        #[arg(long)]
        neuron: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
        #[arg(long)]
        equivalent_csv: Option<PathBuf>,
        #[arg(long)]
        no_output: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compare {
            network,
            spice_csv,
            neuron,
            json_out,
            equivalent_csv,
            no_output,
        } => {
            let config = ComparisonConfig {
                network_config: network,
                neuron_config: neuron,
                spice_csv,
            };

            let mut result = run_comparison(&config)?;

            let mut serialization_ns: u128 = 0;

            if !no_output {
                if let Some(path) = equivalent_csv {
                    let start = Instant::now();
                    result.equivalent.to_csv(&path)?;
                    serialization_ns += start.elapsed().as_nanos();
                }

                if let Some(path) = json_out {
                    result.timings.rust_serialization_ns = serialization_ns;
                    let start = Instant::now();
                    result.write_json(&path)?;
                    serialization_ns += start.elapsed().as_nanos();
                    result.timings.rust_serialization_ns = serialization_ns;
                    // Rewrite so the emitted JSON captures the updated serialization time.
                    result.write_json(&path)?;
                }
            }

            result.timings.rust_serialization_ns = serialization_ns;

            if no_output {
                println!(
                    "[timings] pre={}ns sim={}ns analysis={}ns serialize={}ns",
                    result.timings.rust_preprocess_ns,
                    result.timings.rust_simulate_ns,
                    result.timings.rust_analysis_ns,
                    result.timings.rust_serialization_ns
                );
            }

            for metric in &result.metrics {
                println!(
                    "{}: rms={:.6} mean_abs={:.6} max_abs={:.6} @ {:.6}s ({} samples)",
                    metric.signal,
                    metric.rms_error,
                    metric.mean_abs_error,
                    metric.max_abs_error,
                    metric.max_abs_error_time,
                    metric.sample_count
                );
            }
        }
    }

    Ok(())
}
