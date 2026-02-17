use anyhow::{Context, Result};
use clap::Parser;
use gilgamesh::checkpoint::Checkpoint;
use gilgamesh::data::MnistDataset;
use ndarray::Axis;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "spike_raster_dump")]
#[command(about = "Dump per-timestep spike events for correctly classified digits")]
struct Args {
    /// Path to checkpoint JSON
    #[arg(long, default_value = "./models/physics_6x6_fixed.json")]
    checkpoint: String,

    /// MNIST data directory
    #[arg(long, default_value = "./data")]
    data_dir: String,

    /// Number of correctly classified samples to collect
    #[arg(long, default_value_t = 6)]
    num_samples: usize,

    /// Number of timesteps per forward pass
    #[arg(long, default_value_t = 25)]
    num_steps: usize,

    /// Maximum test samples to scan while searching for correct predictions
    #[arg(long, default_value_t = 500)]
    max_search: usize,

    /// Output JSON path
    #[arg(long, default_value = "./spike_raster_data.json")]
    out: String,
}

#[derive(Debug, Serialize)]
struct RasterSample {
    sample_index: usize,
    label: usize,
    prediction: usize,
    output_spike_count: Vec<f32>,
    input_pixels: Vec<f32>,
    hidden_events: Vec<[usize; 2]>,
    output_events: Vec<[usize; 2]>,
}

#[derive(Debug, Serialize)]
struct RasterDump {
    checkpoint: String,
    data_dir: String,
    num_steps: usize,
    hidden_size: usize,
    output_size: usize,
    image_width: usize,
    image_height: usize,
    samples: Vec<RasterSample>,
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn collect_events(history: &[ndarray::Array2<f32>], num_neurons: usize) -> Vec<[usize; 2]> {
    let mut events = Vec::new();
    for (t, spikes_t) in history.iter().enumerate() {
        for n in 0..num_neurons {
            if spikes_t[[0, n]] > 0.5 {
                events.push([t, n]);
            }
        }
    }
    events
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cp = Checkpoint::load(&args.checkpoint)
        .with_context(|| format!("Failed to load checkpoint {}", args.checkpoint))?;
    let network = cp
        .to_network()
        .context("Failed to reconstruct network from checkpoint")?;

    let image_width = cp.architecture.image_width.unwrap_or(6);
    let image_height = cp.architecture.image_height.unwrap_or(6);
    let dataset = MnistDataset::load_with_dimensions(&args.data_dir, image_width, image_height)
        .with_context(|| format!("Failed to load dataset from {}", args.data_dir))?;

    let hidden_size = network.fc1.out_features;
    let output_size = network.fc2.out_features;

    let mut samples = Vec::new();
    let max_idx = args.max_search.min(dataset.test_len());

    for idx in 0..max_idx {
        if samples.len() >= args.num_samples {
            break;
        }

        let (images, labels) = dataset.get_test_batch(&[idx]);
        let input = images.row(0).to_owned();
        let label = labels[0];

        let input_batch = input.clone().insert_axis(Axis(0));
        let trace = network.forward_traced(&input_batch, args.num_steps);
        let spike_count = trace.output_spike_count.row(0).to_vec();
        let prediction = argmax(&spike_count);

        if prediction != label {
            continue;
        }

        let hidden_events = collect_events(&trace.hidden_spike_history, hidden_size);
        let output_events = collect_events(&trace.output_spike_history, output_size);

        samples.push(RasterSample {
            sample_index: idx,
            label,
            prediction,
            output_spike_count: spike_count,
            input_pixels: input.to_vec(),
            hidden_events,
            output_events,
        });
    }

    if samples.is_empty() {
        anyhow::bail!(
            "No correctly classified samples found in first {} test examples",
            max_idx
        );
    }

    let dump = RasterDump {
        checkpoint: args.checkpoint,
        data_dir: args.data_dir,
        num_steps: args.num_steps,
        hidden_size,
        output_size,
        image_width,
        image_height,
        samples,
    };

    let out_path = PathBuf::from(&args.out);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create output directory {:?}", parent))?;
        }
    }

    let json = serde_json::to_string_pretty(&dump).context("Failed to serialize dump JSON")?;
    fs::write(&out_path, json).with_context(|| format!("Failed to write {:?}", out_path))?;

    println!(
        "Wrote {} correctly classified samples to {}",
        dump.samples.len(),
        out_path.display()
    );
    for s in &dump.samples {
        println!(
            "  sample={} label={} pred={} hidden_spikes={} output_spikes={}",
            s.sample_index,
            s.label,
            s.prediction,
            s.hidden_events.len(),
            s.output_events.len()
        );
    }

    Ok(())
}
