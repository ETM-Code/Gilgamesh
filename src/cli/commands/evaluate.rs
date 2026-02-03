use anyhow::{Context, Result};
use gilgamesh::data::MnistDataset;
use gilgamesh::training::{Trainer, TrainingConfig};

/// Find reasonable image dimensions for a given input size.
/// Prefers square, then factors closest to square, constrained to 4-14 range.
fn find_image_dimensions(input_size: usize) -> Result<(usize, usize)> {
    // Try square first
    let sqrt = (input_size as f64).sqrt() as usize;
    if sqrt * sqrt == input_size && sqrt >= 4 && sqrt <= 14 {
        return Ok((sqrt, sqrt));
    }

    // Find factors closest to square, within reasonable MNIST downsampling range
    for h in (4..=14).rev() {
        if input_size % h == 0 {
            let w = input_size / h;
            if w >= 4 && w <= 14 {
                return Ok((w, h));
            }
        }
    }

    anyhow::bail!(
        "Cannot find valid image dimensions for input_size {}. \
         Need factors in range 4-14.",
        input_size
    )
}

pub(crate) fn evaluate(checkpoint: &str, data_dir: &str, num_steps: usize, batch_size: usize) -> Result<()> {
    use gilgamesh::checkpoint::Checkpoint;

    println!("=== gilgamesh Evaluation ===");
    println!();

    println!("Loading checkpoint: {}", checkpoint);
    let cp = Checkpoint::load(checkpoint)
        .with_context(|| format!("Failed to load checkpoint from {}", checkpoint))?;
    let network = cp
        .to_network()
        .with_context(|| "Failed to reconstruct network from checkpoint")?;

    println!(
        "Network: {} → {} → {} ({})",
        cp.architecture.input_size,
        cp.architecture.hidden_size,
        cp.architecture.output_size,
        cp.architecture.mode
    );

    // Get image dimensions from checkpoint if available, otherwise infer
    let input_size = cp.architecture.input_size;
    let (width, height) = match (cp.architecture.image_width, cp.architecture.image_height) {
        (Some(w), Some(h)) => {
            println!("Using checkpoint dimensions: {}x{}", w, h);
            (w, h)
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
    };
    let trainer = Trainer::new(network, config);
    let accuracy = trainer.evaluate(&dataset);

    println!("Test Accuracy: {:.2}%", accuracy);
    Ok(())
}

