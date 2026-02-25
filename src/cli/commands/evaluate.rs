use anyhow::{Context, Result};
use gilgamesh::data::MnistDataset;
use gilgamesh::training::{Trainer, TrainingConfig};

/// Find reasonable image dimensions for a given input size.
/// Prefers square, then factors closest to square, constrained to 4-14 range.
fn find_image_dimensions(input_size: usize) -> Result<(usize, usize)> {
    // Try square first
    let side_length = (input_size as f64).sqrt() as usize;
    if side_length * side_length == input_size && side_length >= 4 && side_length <= 14 {
        return Ok((side_length, side_length));
    }

    // Find factors closest to square, within reasonable MNIST downsampling range
    for height_factor in (4..=14).rev() {
        if input_size % height_factor == 0 {
            let width_factor = input_size / height_factor;
            if width_factor >= 4 && width_factor <= 14 {
                return Ok((width_factor, height_factor));
            }
        }
    }

    anyhow::bail!(
        "Cannot find valid image dimensions for input_size {}. \
         Need factors in range 4-14.",
        input_size
    )
}

pub(crate) fn evaluate(
    checkpoint: &str,
    data_dir: &str,
    num_steps: usize,
    batch_size: usize,
) -> Result<()> {
    use gilgamesh::checkpoint::Checkpoint;

    println!("=== gilgamesh Evaluation ===");
    println!();

    println!("Loading checkpoint: {}", checkpoint);
    let loaded_checkpoint = Checkpoint::load(checkpoint)
        .with_context(|| format!("Failed to load checkpoint from {}", checkpoint))?;
    let network = loaded_checkpoint
        .to_network()
        .with_context(|| "Failed to reconstruct network from checkpoint")?;

    println!(
        "Network: {} → {} → {} ({})",
        loaded_checkpoint.architecture.input_size,
        loaded_checkpoint.architecture.hidden_size,
        loaded_checkpoint.architecture.output_size,
        loaded_checkpoint.architecture.mode
    );

    // Get image dimensions from checkpoint if available, otherwise infer
    let input_size = loaded_checkpoint.architecture.input_size;
    let (width, height) = match (
        loaded_checkpoint.architecture.image_width,
        loaded_checkpoint.architecture.image_height,
    ) {
        (Some(width), Some(height)) => {
            println!("Using checkpoint dimensions: {}x{}", width, height);
            (width, height)
        }
        _ => {
            let dims = find_image_dimensions(input_size)?;
            println!("Inferred dimensions from input_size: {}x{}", dims.0, dims.1);
            dims
        }
    };

    println!("Loading MNIST dataset ({}x{})...", width, height);
    let dataset = MnistDataset::load_with_dimensions(data_dir, width, height)
        .context("Failed to load MNIST dataset")?;
    println!("Loaded {} test samples", dataset.test_len());
    println!();

    let config = TrainingConfig {
        lr: 0.0,
        epochs: 0,
        batch_size,
        num_steps,
        seed: 42,
        num_workers: 0,
        bptt_steps: None,
    };
    let mut trainer = Trainer::new(network, config);
    let accuracy = trainer.evaluate(&dataset);

    println!("Test Accuracy: {:.2}%", accuracy);
    Ok(())
}
