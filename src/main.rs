use std::path::PathBuf;

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
        } => {
            let config = ComparisonConfig {
                network_config: network,
                neuron_config: neuron,
                spice_csv,
            };

            let result = run_comparison(&config)?;

            if let Some(path) = equivalent_csv {
                result.equivalent.to_csv(&path)?;
            }

            if let Some(path) = json_out {
                result.write_json(&path)?;
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
