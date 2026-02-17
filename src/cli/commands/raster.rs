use anyhow::{Context, Result};
use gilgamesh::checkpoint::Checkpoint;
use gilgamesh::data::MnistDataset;
use ndarray::Axis;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cli::utils::find_latest_checkpoint;

#[derive(Clone, Debug, Serialize)]
struct RasterSample {
    sample_index: usize,
    label: usize,
    prediction: usize,
    output_spike_count: Vec<f32>,
    input_pixels: Vec<f32>,
    hidden_events: Vec<[usize; 2]>,
    output_events: Vec<[usize; 2]>,
}

#[derive(Clone, Debug, Serialize)]
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

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory {:?}", parent))?;
        }
    }
    Ok(())
}

pub(crate) fn run_spike_raster(
    checkpoint: Option<String>,
    data_dir: &str,
    num_samples: usize,
    num_steps: usize,
    max_search: usize,
    output_json: &str,
    output_image: &str,
    no_plot: bool,
    background_image: Option<String>,
    bg_alpha: f32,
    dpi: usize,
) -> Result<()> {
    let checkpoint_path = match checkpoint {
        Some(path) => path,
        None => match find_latest_checkpoint() {
            Some(path) => {
                println!("Using most recent checkpoint: {}", path);
                path
            }
            None => {
                anyhow::bail!(
                    "No checkpoint specified and no checkpoint files found.\n\
                     Train a model first: gilgamesh train --save-checkpoint model.json"
                );
            }
        },
    };

    let cp = Checkpoint::load(&checkpoint_path)
        .with_context(|| format!("Failed to load checkpoint {}", checkpoint_path))?;
    let network = cp
        .to_network()
        .context("Failed to reconstruct network from checkpoint")?;

    let image_width = cp.architecture.image_width.unwrap_or(6);
    let image_height = cp.architecture.image_height.unwrap_or(6);
    let dataset = MnistDataset::load_with_dimensions(data_dir, image_width, image_height)
        .with_context(|| format!("Failed to load dataset from {}", data_dir))?;

    let hidden_size = network.fc1.out_features;
    let output_size = network.fc2.out_features;

    let mut samples = Vec::new();
    let max_idx = max_search.min(dataset.test_len());

    for idx in 0..max_idx {
        if samples.len() >= num_samples {
            break;
        }

        let (images, labels) = dataset.get_test_batch(&[idx]);
        let input = images.row(0).to_owned();
        let label = labels[0];

        let input_batch = input.clone().insert_axis(Axis(0));
        let trace = network.forward_traced(&input_batch, num_steps);
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
        checkpoint: checkpoint_path.clone(),
        data_dir: data_dir.to_string(),
        num_steps,
        hidden_size,
        output_size,
        image_width,
        image_height,
        samples,
    };

    let output_json_path = PathBuf::from(output_json);
    ensure_parent_dir(&output_json_path)?;
    let json = serde_json::to_string_pretty(&dump).context("Failed to serialize dump JSON")?;
    fs::write(&output_json_path, json)
        .with_context(|| format!("Failed to write {}", output_json_path.display()))?;

    println!(
        "Wrote {} correctly classified samples to {}",
        dump.samples.len(),
        output_json_path.display()
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

    if no_plot {
        println!("Plot generation skipped (--no-plot).");
        return Ok(());
    }

    let output_image_path = PathBuf::from(output_image);
    ensure_parent_dir(&output_image_path)?;

    let mut cmd = Command::new("python3");
    cmd.arg("tools/plot_spike_raster.py")
        .arg("--input")
        .arg(&output_json_path)
        .arg("--output")
        .arg(&output_image_path)
        .arg("--bg-alpha")
        .arg(bg_alpha.to_string())
        .arg("--dpi")
        .arg(dpi.to_string());

    if let Some(bg) = background_image {
        cmd.arg("--background").arg(bg);
    }

    let status = cmd
        .status()
        .with_context(|| "Failed to run plot script via python3")?;
    if !status.success() {
        anyhow::bail!(
            "Raster plot script failed with exit code {:?}",
            status.code()
        );
    }

    println!("Wrote raster plot to {}", output_image_path.display());
    Ok(())
}
