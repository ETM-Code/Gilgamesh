use anyhow::{Context, Result};
use clap::Parser;
use gilgamesh::checkpoint::Checkpoint;
use gilgamesh::data::MnistDataset;
use gilgamesh::neurons::NeuronMode;
use gilgamesh::spice::{SpiceNetlist, SpiceParams};
use ndarray::Axis;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Parser)]
#[command(name = "network_spice_bench")]
#[command(about = "Benchmark full-network Gilgamesh vs ngspice runtime")]
struct Args {
    /// Path to checkpoint JSON
    #[arg(long, default_value = "./models/physics_6x6_fixed.json")]
    checkpoint: String,

    /// MNIST data directory
    #[arg(long, default_value = "./data")]
    data_dir: String,

    /// Test sample index
    #[arg(long, default_value_t = 42)]
    sample: usize,

    /// Number of timesteps for Gilgamesh forward pass
    #[arg(long, default_value_t = 25)]
    num_steps: usize,

    /// Optional SPICE simulation duration in milliseconds
    #[arg(long)]
    duration_ms: Option<f32>,

    /// Warmup iterations for in-process Gilgamesh benchmark
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Timed iterations for in-process Gilgamesh benchmark
    #[arg(long, default_value_t = 50)]
    gilgamesh_iters: usize,

    /// Optional startup-included CLI runs (gilgamesh spice without ngspice)
    #[arg(long, default_value_t = 3)]
    gilgamesh_cli_runs: usize,

    /// Path to gilgamesh binary for CLI startup benchmark
    #[arg(long, default_value = "./target/release/gilgamesh")]
    gilgamesh_bin: String,

    /// Number of ngspice runs to benchmark
    #[arg(long, default_value_t = 1)]
    spice_runs: usize,

    /// Output directory for benchmark artifacts
    #[arg(long, default_value = "./benchmark_spice")]
    output_dir: String,
}

fn avg_duration(durs: &[Duration]) -> Duration {
    if durs.is_empty() {
        return Duration::ZERO;
    }
    let total: f64 = durs.iter().map(Duration::as_secs_f64).sum();
    Duration::from_secs_f64(total / durs.len() as f64)
}

fn min_duration(durs: &[Duration]) -> Duration {
    durs.iter().copied().min().unwrap_or(Duration::ZERO)
}

fn max_duration(durs: &[Duration]) -> Duration {
    durs.iter().copied().max().unwrap_or(Duration::ZERO)
}

fn run_ngspice_quiet(
    netlist_abs: &Path,
    output_dir_abs: &Path,
    run_idx: usize,
) -> Result<Duration> {
    let raw_path = output_dir_abs.join(format!("gilgamesh_output_run{}.raw", run_idx));
    let log_path = output_dir_abs.join(format!("ngspice_run{}.log", run_idx));

    let start = Instant::now();
    let status = Command::new("ngspice")
        .args(["-b", "-r"])
        .arg(&raw_path)
        .arg("-o")
        .arg(&log_path)
        .arg(netlist_abs)
        .current_dir(output_dir_abs)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| "Failed to run ngspice. Is it installed and in PATH?")?;
    let elapsed = start.elapsed();

    if !status.success() {
        let log = fs::read_to_string(&log_path).unwrap_or_default();
        anyhow::bail!(
            "ngspice failed in run {} (exit code {:?}):\n{}",
            run_idx,
            status.code(),
            log
        );
    }

    Ok(elapsed)
}

fn run_gilgamesh_cli_once(
    bin_path: &Path,
    checkpoint: &str,
    data_dir: &str,
    sample: usize,
    num_steps: usize,
    duration_ms: Option<f32>,
    output_dir: &Path,
    run_idx: usize,
) -> Result<Duration> {
    let run_output = output_dir.join(format!("cli_run_{}", run_idx));
    fs::create_dir_all(&run_output)
        .with_context(|| format!("Failed to create output dir {:?}", run_output))?;

    let mut cmd = Command::new(bin_path);
    cmd.arg("spice")
        .arg("--checkpoint")
        .arg(checkpoint)
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--sample")
        .arg(sample.to_string())
        .arg("--output-dir")
        .arg(run_output.as_os_str())
        .arg("--num-steps")
        .arg(num_steps.to_string())
        // Startup benchmark should only include Rust path; ngspice benchmark is separate.
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    if let Some(ms) = duration_ms {
        cmd.arg("--duration").arg(ms.to_string());
    }

    let start = Instant::now();
    let status = cmd
        .status()
        .with_context(|| format!("Failed to run {:?}", bin_path))?;
    let elapsed = start.elapsed();

    if !status.success() {
        anyhow::bail!(
            "gilgamesh CLI run {} failed (exit code {:?})",
            run_idx,
            status.code()
        );
    }

    Ok(elapsed)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let output_dir = PathBuf::from(&args.output_dir);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create output dir {:?}", output_dir))?;
    let output_dir_abs = output_dir
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize output dir {:?}", output_dir))?;

    println!("=== Full Network Benchmark: Gilgamesh vs SPICE ===");
    println!("Checkpoint: {}", args.checkpoint);
    println!("Data dir:   {}", args.data_dir);
    println!("Sample:     {}", args.sample);
    println!("Steps:      {}", args.num_steps);
    println!();

    // Load model + sample once (setup cost reported separately)
    let setup_start = Instant::now();
    let cp = Checkpoint::load(&args.checkpoint)
        .with_context(|| format!("Failed to load checkpoint {}", args.checkpoint))?;
    let network = cp
        .to_network()
        .context("Failed to reconstruct network from checkpoint")?;
    let dataset = MnistDataset::load(&args.data_dir)
        .with_context(|| format!("Failed to load dataset from {}", args.data_dir))?;

    let sample_idx = args.sample.min(dataset.test_len().saturating_sub(1));
    let (images, labels) = dataset.get_test_batch(&[sample_idx]);
    let input = images.row(0).to_owned();
    let input_batch = input.clone().insert_axis(Axis(0));
    let true_label = labels[0];
    let setup_elapsed = setup_start.elapsed();

    let model_dt = match &network.lif1.mode {
        NeuronMode::Physics { dt, .. } => *dt,
        NeuronMode::Simple => 0.001,
    };
    let duration_s = args
        .duration_ms
        .map(|ms| ms / 1000.0)
        .unwrap_or(args.num_steps as f32 * model_dt);
    let duration_ms = duration_s * 1000.0;

    println!("Loaded sample {} (label={})", sample_idx, true_label);
    println!("Setup time: {:.3} ms", setup_elapsed.as_secs_f64() * 1000.0);
    println!("SPICE duration: {:.3} ms", duration_ms);
    println!();

    // In-process Gilgamesh benchmark (amortized startup)
    for _ in 0..args.warmup {
        let _ = network.forward_traced(&input_batch, args.num_steps);
    }

    let mut gil_durations = Vec::with_capacity(args.gilgamesh_iters);
    let mut last_trace = None;
    for _ in 0..args.gilgamesh_iters {
        let start = Instant::now();
        let trace = network.forward_traced(&input_batch, args.num_steps);
        let elapsed = start.elapsed();
        gil_durations.push(elapsed);
        last_trace = Some(trace);
    }

    let gil_avg = avg_duration(&gil_durations);
    let gil_min = min_duration(&gil_durations);
    let gil_max = max_duration(&gil_durations);
    let gil_total = Duration::from_secs_f64(gil_durations.iter().map(Duration::as_secs_f64).sum());

    let gil_spikes = last_trace
        .as_ref()
        .map(|t| t.output_spike_count.row(0).to_vec())
        .unwrap_or_default();
    let gil_pred = gil_spikes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("Gilgamesh (in-process, amortized startup):");
    println!(
        "  runs={} warmup={} total={:.3} ms avg={:.3} ms min={:.3} ms max={:.3} ms",
        args.gilgamesh_iters,
        args.warmup,
        gil_total.as_secs_f64() * 1000.0,
        gil_avg.as_secs_f64() * 1000.0,
        gil_min.as_secs_f64() * 1000.0,
        gil_max.as_secs_f64() * 1000.0
    );
    println!("  prediction={} spikes={:?}", gil_pred, gil_spikes);
    println!();

    // Optional startup-included CLI benchmark
    let mut cli_durations = Vec::new();
    if args.gilgamesh_cli_runs > 0 {
        let bin_path = PathBuf::from(&args.gilgamesh_bin);
        if !bin_path.exists() {
            println!(
                "Gilgamesh CLI benchmark skipped: binary not found at {}",
                bin_path.display()
            );
            println!("Hint: build it with `cargo build --release --bin gilgamesh`");
            println!();
        } else {
            for i in 0..args.gilgamesh_cli_runs {
                cli_durations.push(run_gilgamesh_cli_once(
                    &bin_path,
                    &args.checkpoint,
                    &args.data_dir,
                    sample_idx,
                    args.num_steps,
                    args.duration_ms,
                    &output_dir_abs,
                    i,
                )?);
            }

            let cli_avg = avg_duration(&cli_durations);
            let cli_min = min_duration(&cli_durations);
            let cli_max = max_duration(&cli_durations);
            println!("Gilgamesh CLI (startup included, no ngspice):");
            println!(
                "  runs={} avg={:.3} ms min={:.3} ms max={:.3} ms",
                args.gilgamesh_cli_runs,
                cli_avg.as_secs_f64() * 1000.0,
                cli_min.as_secs_f64() * 1000.0,
                cli_max.as_secs_f64() * 1000.0
            );
            println!();
        }
    }

    // SPICE benchmark
    let params = SpiceParams::from_network(&network);
    let netlist = SpiceNetlist::from_network(
        &network,
        input
            .as_slice()
            .context("Failed to access input sample as contiguous slice")?,
        duration_s,
        &params,
    );
    let netlist_path = output_dir_abs.join("gilgamesh_bench.cir");
    netlist.write(&netlist_path)?;
    let netlist_abs = netlist_path
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize netlist path {:?}", netlist_path))?;

    let mut spice_durations = Vec::with_capacity(args.spice_runs);
    for i in 0..args.spice_runs {
        spice_durations.push(run_ngspice_quiet(&netlist_abs, &output_dir_abs, i)?);
    }
    let spice_avg = avg_duration(&spice_durations);
    let spice_min = min_duration(&spice_durations);
    let spice_max = max_duration(&spice_durations);

    println!("ngspice (full network):");
    println!(
        "  runs={} avg={:.3} ms min={:.3} ms max={:.3} ms",
        args.spice_runs,
        spice_avg.as_secs_f64() * 1000.0,
        spice_min.as_secs_f64() * 1000.0,
        spice_max.as_secs_f64() * 1000.0
    );
    println!();

    if gil_avg > Duration::ZERO {
        println!(
            "Speed ratio (ngspice avg / gilgamesh in-process avg): {:.1}x",
            spice_avg.as_secs_f64() / gil_avg.as_secs_f64()
        );
    }
    if !cli_durations.is_empty() {
        let cli_avg = avg_duration(&cli_durations);
        if cli_avg > Duration::ZERO {
            println!(
                "Speed ratio (ngspice avg / gilgamesh CLI avg): {:.1}x",
                spice_avg.as_secs_f64() / cli_avg.as_secs_f64()
            );
        }
    }

    Ok(())
}
