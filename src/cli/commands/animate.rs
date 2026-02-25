use anyhow::Result;

#[cfg(feature = "animation")]
use anyhow::Context;

#[cfg(feature = "animation")]
use crate::cli::utils::find_latest_checkpoint;

#[cfg(feature = "animation")]
pub(crate) fn run_animation(
    checkpoint: Option<String>,
    data_dir: &str,
    speed: f32,
    start_sample: usize,
) -> Result<()> {
    use gilgamesh::animation::{create_shared_animation, run_animation as run_nannou};
    use gilgamesh::checkpoint::Checkpoint;
    use std::thread;
    use std::time::Duration;

    println!("=== gilgamesh Interactive Visualizer ===");
    println!();

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

    println!("Loading checkpoint: {}", checkpoint_path);
    let cp = Checkpoint::load(&checkpoint_path)
        .with_context(|| format!("Failed to load checkpoint from {}", checkpoint_path))?;
    let network = cp
        .to_network()
        .with_context(|| "Failed to reconstruct network from checkpoint")?;

    let input_size = cp.architecture.input_size;
    let hidden_size = cp.architecture.hidden_size;
    let output_size = cp.architecture.output_size;

    println!(
        "Network: {} → {} → {} ({})",
        input_size, hidden_size, output_size, cp.architecture.mode
    );
    if let Some(ref meta) = cp.metadata {
        println!(
            "Trained: {} epochs, {:.2}% test accuracy",
            meta.epochs_trained, meta.final_test_accuracy
        );
    }

    println!("Loading MNIST dataset...");
    let dataset =
        gilgamesh::data::MnistDataset::load(data_dir).context("Failed to load MNIST dataset")?;

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

    let animation = create_shared_animation(&[input_size, hidden_size, output_size]);

    let initial_sample = start_sample.min(total_samples - 1);
    {
        let mut anim = animation.lock().unwrap();
        anim.speed = speed;
        anim.viewer.total_samples = total_samples;
        anim.viewer.total_steps = 25;

        let image = test_images.row(initial_sample).to_vec();
        anim.reset_for_sample(initial_sample, test_labels[initial_sample] as u8, &image);
    }

    let fc1_weights: Vec<Vec<f32>> = network
        .fc1
        .weight
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let fc2_weights: Vec<Vec<f32>> = network
        .fc2
        .weight
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();

    {
        let mut anim = animation.lock().unwrap();
        anim.set_weights(&[fc1_weights, fc2_weights]);
    }

    let animation_clone = animation.clone();
    let num_steps = 25usize;

    thread::spawn(move || {
        let mut current_sample = initial_sample;

        loop {
            let (should_change, new_sample) = {
                let mut anim = animation_clone.lock().unwrap();
                if let Some(delta) = anim.viewer.sample_request.take() {
                    let new_idx = if delta == 0 {
                        current_sample
                    } else {
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

                let image = test_images.row(current_sample).to_vec();
                let label = test_labels[current_sample] as u8;

                {
                    let mut anim = animation_clone.lock().unwrap();
                    anim.reset_for_sample(current_sample, label, &image);
                }
            }

            let sample_image = test_images.row(current_sample).to_owned();
            let sample_batch = sample_image.insert_axis(ndarray::Axis(0));

            {
                let mut anim = animation_clone.lock().unwrap();
                anim.viewer.current_step = 0;
                anim.viewer.simulation_complete = false;
            }

            let mut hidden_state = network.lif1.init_state(1);
            let mut output_state = network.lif2.init_state(1);

            for step in 0..num_steps {
                {
                    let anim = animation_clone.lock().unwrap();
                    if anim.viewer.sample_request.is_some() {
                        break;
                    }
                    if anim.paused {
                        drop(anim);
                        thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                }

                let fc1_out = network.fc1.forward(&sample_batch);
                let (hidden_spikes, new_hidden_state, _) =
                    network.lif1.forward(&fc1_out, &hidden_state);
                hidden_state = new_hidden_state;

                let fc2_out = network.fc2.forward(&hidden_spikes);
                let (output_spikes, new_output_state, _) =
                    network.lif2.forward(&fc2_out, &output_state);
                output_state = new_output_state;

                {
                    let mut anim = animation_clone.lock().unwrap();
                    anim.viewer.current_step = step + 1;

                    let input_mem: Vec<f32> = sample_batch.row(0).to_vec();
                    let hidden_mem: Vec<f32> = hidden_state.mem.row(0).to_vec();
                    let output_mem: Vec<f32> = output_state.mem.row(0).to_vec();

                    let input_spikes: Vec<bool> =
                        sample_batch.row(0).iter().map(|&v| v > 0.5).collect();
                    let hidden_spikes_bool: Vec<bool> =
                        hidden_spikes.row(0).iter().map(|&v| v > 0.5).collect();
                    let output_spikes_bool: Vec<bool> =
                        output_spikes.row(0).iter().map(|&v| v > 0.5).collect();

                    anim.update_neurons(
                        &[input_mem, hidden_mem, output_mem],
                        &[input_spikes, hidden_spikes_bool, output_spikes_bool],
                        0.04,
                    );
                }

                thread::sleep(Duration::from_millis((40.0 / speed) as u64));
            }

            {
                let mut anim = animation_clone.lock().unwrap();
                anim.viewer.simulation_complete = true;
            }

            thread::sleep(Duration::from_millis(300));
        }
    });

    println!("Launching visualization window...");
    run_nannou(animation);

    Ok(())
}

#[cfg(not(feature = "animation"))]
pub(crate) fn run_animation(
    _checkpoint: Option<String>,
    _data_dir: &str,
    _speed: f32,
    _sample: usize,
) -> Result<()> {
    println!("Animation feature not enabled. Rebuild with --features animation");
    Ok(())
}
