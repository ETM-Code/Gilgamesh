use anyhow::{Context, Result};
use gilgamesh::config::Config;
use gilgamesh::data::{AnyDataset, ArrayDataset, Dataset, MnistDataset};
use gilgamesh::training::{NoiseParams, Trainer, TrainingConfig};

/// Find reasonable image dimensions for a given input size.
fn find_image_dimensions(input_size: usize) -> Result<(usize, usize)> {
    let side_length = (input_size as f64).sqrt() as usize;
    if side_length * side_length == input_size && side_length >= 4 && side_length <= 14 {
        return Ok((side_length, side_length));
    }
    for height_factor in (4..=14).rev() {
        if input_size % height_factor == 0 {
            let width_factor = input_size / height_factor;
            if width_factor >= 4 && width_factor <= 14 {
                return Ok((width_factor, height_factor));
            }
        }
    }
    anyhow::bail!(
        "Cannot find valid image dimensions for input_size {}.",
        input_size
    )
}

pub(crate) fn evaluate(
    checkpoint: &str,
    data_dir: &str,
    dataset_kind: &str,
    num_steps: usize,
    batch_size: usize,
    noise: bool,
    config_path: Option<&str>,
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

    let dataset = match dataset_kind {
        "mnist" => {
            println!("Loading MNIST dataset ({}x{})...", width, height);
            let ds = MnistDataset::load_with_dimensions(data_dir, width, height)
                .context("Failed to load MNIST dataset")?;
            AnyDataset::Mnist(ds)
        }
        "ecg" => {
            println!("Loading ECG dataset from .npy files...");
            let ds =
                ArrayDataset::load_npy(data_dir).context("Failed to load ECG dataset (.npy)")?;
            AnyDataset::Array(ds)
        }
        other => anyhow::bail!("Unsupported dataset kind: {other}. Use 'mnist' or 'ecg'."),
    };
    println!("Loaded {} test samples", dataset.test_len());

    let loaded_cfg = if let Some(cfg_path) = config_path {
        Some(
            Config::load(cfg_path)
                .with_context(|| format!("Failed to load config from {}", cfg_path))?,
        )
    } else {
        None
    };

    let mut effective_num_steps = num_steps;
    if let Some(meta) = &loaded_checkpoint.metadata {
        if let Some(saved_steps) = meta.num_steps {
            let cli_is_default = num_steps == TrainingConfig::default().num_steps;
            if cli_is_default && saved_steps != num_steps {
                effective_num_steps = saved_steps;
                println!("Timesteps: using checkpoint metadata value {}", saved_steps);
            }
        }
    }

    let config = TrainingConfig {
        lr: 0.0,
        epochs: 0,
        batch_size,
        num_steps: effective_num_steps,
        seed: 42,
        num_workers: 0,
        bptt_steps: None,
        weight_decay: 0.01,
        max_grad_norm: 1.0,
    };
    let mut trainer = Trainer::new(network, config);

    if let Some(meta) = &loaded_checkpoint.metadata {
        if let Some(bits) = meta.quant_bits {
            trainer.quant_bits = Some(bits);
        }
        if let Some(split_sign) = meta.split_sign_quant {
            trainer.split_sign_quant = split_sign;
        }
        if let Some(input_bits) = meta.input_quant_bits {
            trainer.input_quant_bits = input_bits;
        }
    }

    // Config overrides metadata for eval behavior when provided.
    if let Some(cfg) = &loaded_cfg {
        if cfg.quantization.enabled {
            trainer.quant_bits = Some(cfg.quantization.bits);
        } else {
            trainer.quant_bits = None;
        }
        trainer.split_sign_quant = cfg.quantization.split_sign;
        trainer.input_quant_bits = cfg.quantization.input_bits;
    }

    if let Some(bits) = trainer.quant_bits {
        println!(
            "Eval quantization: {}-bit{}",
            bits,
            if trainer.split_sign_quant {
                ", split-sign"
            } else {
                ""
            }
        );
    }
    if trainer.input_quant_bits > 0 {
        println!("Input quant: {}-bit DAC", trainer.input_quant_bits);
    }

    // Load noise parameters from config file if provided, or use hardware defaults
    if noise {
        let noise_params = if let Some(cfg) = &loaded_cfg {
            NoiseParams {
                weight_std: cfg.noise.weight_std,
                threshold_std: cfg.noise.threshold_std,
                membrane_std: cfg.noise.membrane_std,
                input_std: cfg.noise.input_std,
            }
        } else {
            // Hardware-realistic defaults
            NoiseParams {
                weight_std: 0.05,
                threshold_std: 0.02,
                membrane_std: 0.01,
                input_std: 0.1,
            }
        };

        println!(
            "Noise enabled: weight={:.0}%, threshold={:.0}%, membrane={:.0}%, input={:.0}%",
            noise_params.weight_std * 100.0,
            noise_params.threshold_std * 100.0,
            noise_params.membrane_std * 100.0,
            noise_params.input_std * 100.0,
        );

        trainer.noise = noise_params;
        trainer.noise_during_eval = true;
    }

    println!();
    let accuracy = trainer.evaluate(&dataset);
    println!("Test Accuracy: {:.2}%", accuracy);

    Ok(())
}
