use anyhow::{Context, Result};
use gilgamesh::network::Network;
use gilgamesh::neurons::NeuronMode;

use crate::cli::utils::find_latest_checkpoint;

pub(crate) fn run_spice(
    checkpoint: Option<String>,
    data_dir: &str,
    sample: Option<usize>,
    output_dir: &str,
    should_run_ngspice: bool,
    num_steps: usize,
    duration_ms: Option<f32>,
    analog_output: bool,
    pulse_stretch: bool,
) -> Result<()> {
    use gilgamesh::checkpoint::Checkpoint;
    use gilgamesh::data::MnistDataset;
    use gilgamesh::spice::{run_ngspice, ComparisonResult, SpiceNetlist, SpiceParams};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::fs;
    use std::path::Path;

    println!("=== gilgamesh SPICE Comparison (Detailed Pulse-Stretch Model) ===");
    println!();

    let checkpoint_path = match checkpoint {
        Some(path) => Some(path),
        None => find_latest_checkpoint().map(|path| {
            println!("Using most recent checkpoint: {}", path);
            path
        }),
    };

    let network = if let Some(ref path) = checkpoint_path {
        println!("Loading checkpoint: {}", path);
        let cp = Checkpoint::load(path).with_context(|| format!("Failed to load checkpoint from {}", path))?;
        cp.to_network()
            .with_context(|| "Failed to reconstruct network from checkpoint")?
    } else {
        println!("No checkpoint found, using untrained network");
        Network::new(49, 100, 10, 0.9, 42)
    };

    println!("Loading MNIST dataset from {}...", data_dir);
    let dataset = MnistDataset::load(data_dir).context("Failed to load MNIST dataset")?;

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

    let params = SpiceParams::from_network(&network)
        .with_analog_output(analog_output)
        .with_pulse_stretch(pulse_stretch);

    // Compute simulation duration and effective num_steps.
    // If --duration is set, use it. Otherwise, auto-compute from model params,
    // ensuring enough time for the membrane to reach threshold and spike.
    let model_dt = match &network.lif1.mode {
        NeuronMode::Physics { dt, .. } => *dt,
        NeuronMode::Simple => 0.001, // 1ms default for simple mode
    };

    let sim_duration_s = if let Some(dur_ms) = duration_ms {
        dur_ms / 1000.0
    } else {
        let naive_duration = num_steps as f32 * model_dt;
        let min_duration = 10.0 * params.tau_m();
        if naive_duration < min_duration {
            println!(
                "Note: {} steps × {:.3}ms dt = {:.3}ms is short relative to tau_m={:.2}ms.",
                num_steps,
                model_dt * 1000.0,
                naive_duration * 1000.0,
                params.tau_m() * 1000.0,
            );
            println!(
                "  Auto-extending to {:.1}ms (10× tau_m). Use --duration to override.",
                min_duration * 1000.0,
            );
            println!();
            min_duration
        } else {
            naive_duration
        }
    };

    let effective_num_steps = (sim_duration_s / model_dt).round() as usize;

    println!("Circuit parameters:");
    println!(
        "  Supply:        VDD={:.1}V, Vref={:.1}V",
        params.supply.vdd, params.supply.vref
    );
    println!(
        "  Membrane:      C={:.3e}F, R={:.3e}Ω, tau={:.2}ms",
        params.membrane.c_mem,
        params.membrane.r_leak,
        params.tau_m() * 1000.0
    );
    println!(
        "  Threshold:     Vref+{:.2}V, hysteresis={:.3}V",
        params.threshold.over_vref, params.threshold.hysteresis
    );
    println!(
        "  Pulse stretch: {} (tau={:.4}ms)",
        if pulse_stretch { "enabled" } else { "disabled" },
        params.tau_pulse() * 1000.0
    );
    println!(
        "  Analog output: {}",
        if analog_output { "enabled" } else { "disabled" }
    );
    println!(
        "  Duration:      {:.2}ms ({} steps × {:.3}ms dt)",
        sim_duration_s * 1000.0,
        effective_num_steps,
        model_dt * 1000.0,
    );
    println!();

    println!("Running gilgamesh simulation ({} steps)...", effective_num_steps);
    let input_batch = input.clone().insert_axis(ndarray::Axis(0));
    let trace = network.forward_traced(&input_batch, effective_num_steps);

    let gilgamesh_spikes: Vec<f32> = trace.output_spike_count.row(0).to_vec();
    let gilgamesh_pred = gilgamesh_spikes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!(
        "Gilgamesh prediction: {} (correct: {})",
        gilgamesh_pred,
        gilgamesh_pred == label
    );
    println!("Gilgamesh spike counts: {:?}", gilgamesh_spikes);

    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

    println!();
    println!("Generating SPICE netlist (detailed pulse-stretch model)...");
    let netlist = SpiceNetlist::from_network(&network, input.as_slice().unwrap(), sim_duration_s, &params);

    let netlist_path = output_path.join("gilgamesh.cir");
    netlist.write(&netlist_path)?;
    println!("Netlist written to: {:?}", netlist_path);

    if should_run_ngspice {
        println!();
        println!("Running ngspice...");

        let output_size = network.fc2.weight.shape()[1];
        match run_ngspice(&netlist_path, output_path, output_size) {
            Ok(spice_output) => {
                println!("SPICE simulation complete");
                println!();

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
