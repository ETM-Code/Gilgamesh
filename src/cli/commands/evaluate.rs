use anyhow::{Context, Result};
use gilgamesh::data::MnistDataset;
use gilgamesh::training::{Trainer, TrainingConfig};

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

    println!("Loading MNIST dataset...");
    let dataset = MnistDataset::load(data_dir).context("Failed to load MNIST dataset")?;
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

