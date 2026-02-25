use anyhow::{Context, Result};
use gilgamesh::config::Config;
use gilgamesh::data::MnistDataset;
use gilgamesh::layers::linear::{DEFAULT_SYNAPSE_NEG_GAIN, DEFAULT_SYNAPSE_POS_GAIN};
use gilgamesh::network::Network;
use gilgamesh::surrogate::SurrogateGradient;
use gilgamesh::training::{NoiseParams, Trainer, TrainingConfig};

pub(crate) fn train_from_config(
    config_path: &str,
    data_dir: &str,
    visualize: bool,
    visualize_file: Option<String>,
    save_checkpoint: Option<String>,
) -> Result<()> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("Failed to load config from {}", config_path))?;
    println!("Loaded config from: {}", config_path);
    train_with_config(&cfg, data_dir, visualize, visualize_file, save_checkpoint)
}

pub(crate) fn train_with_config(
    cfg: &Config,
    data_dir: &str,
    visualize: bool,
    visualize_file: Option<String>,
    save_checkpoint: Option<String>,
) -> Result<()> {
    println!("=== gilgamesh Training ===");
    println!("Mode:           {}", cfg.mode);
    println!("Learning rate:  {}", cfg.training.lr);
    println!("Epochs:         {}", cfg.training.epochs);
    println!("Batch size:     {}", cfg.training.batch_size);
    println!("Timesteps:      {}", cfg.training.num_steps);
    println!("Hidden size:    {}", cfg.network.hidden_size);
    println!("Beta:           {}", cfg.neuron.beta);
    println!("Seed:           {}", cfg.training.seed);
    println!("Slope:          {}", cfg.neuron.slope);
    if cfg.quantization.enabled {
        println!("Quantization:   {}-bit", cfg.quantization.bits);
    }
    if cfg.noise.enabled {
        let scope = if cfg.noise.training_only {
            "train only"
        } else {
            "train+eval"
        };
        println!(
            "Noise:          weight={:.2}% thresh={:.1}% membrane={:.3} input={:.1}% ({})",
            cfg.noise.weight_std * 100.0,
            cfg.noise.threshold_std * 100.0,
            cfg.noise.membrane_std,
            cfg.noise.input_std * 100.0,
            scope
        );
    }
    println!();

    let image_width = cfg.network.get_width();
    let image_height = cfg.network.get_height();
    println!("Loading MNIST dataset...");
    let dataset = MnistDataset::load_with_dimensions(data_dir, image_width, image_height)
        .context("Failed to load MNIST dataset")?;
    println!(
        "Loaded {} training samples, {} test samples",
        dataset.train_len(),
        dataset.test_len()
    );
    println!(
        "Image size: {}x{} = {} features",
        image_width,
        image_height,
        dataset.feature_dim()
    );
    println!();

    use gilgamesh::config::EncodingType;
    use gilgamesh::data::InputEncoder;
    let input_encoder = if cfg.input_encoding.encoding_type == EncodingType::Temporal {
        // Temporal encoding uses image_height for row-by-row presentation
        Some(InputEncoder::temporal(
            image_height,
            cfg.input_encoding.row_spacing,
            cfg.input_encoding.pulse_width,
            cfg.physics.dt,
        ))
    } else {
        None
    };

    let input_size = input_encoder
        .as_ref()
        .map(|e| e.output_dim())
        .unwrap_or(dataset.feature_dim());
    let output_size = cfg.network.output_size;
    let hidden_size = cfg.network.hidden_size;
    let beta = cfg.neuron.beta;
    let slope = cfg.neuron.slope;
    let seed = cfg.training.seed;

    let mut network = if cfg.is_physics_mode() && cfg.physics.enabled {
        let tau_m = cfg.physics.tau_m;
        let dt = cfg.physics.dt;
        if cfg.physics.adaptation_enabled {
            println!(
                "Using Physics mode with adaptation: tau_m={:.4}s, dt={:.4}s, tau_theta={:.4}s",
                tau_m, dt, cfg.physics.tau_theta
            );
            Network::new_physics_with_adaptation(
                input_size,
                hidden_size,
                output_size,
                tau_m,
                dt,
                cfg.physics.tau_theta,
                cfg.physics.theta_low,
                cfg.physics.theta_high,
                seed,
            )
        } else {
            let tau_pulse = cfg.physics.tau_pulse;
            let v_peak = cfg.hardware.pulse_peak();
            println!(
                "Using Physics mode: tau_m={:.4}s, dt={:.4}s, tau_pulse={:.4e}s, v_peak={:.2}V",
                tau_m, dt, tau_pulse, v_peak
            );
            Network::new_physics_with_pulse(
                input_size,
                hidden_size,
                output_size,
                tau_m,
                dt,
                tau_pulse,
                v_peak,
                seed,
            )
        }
    } else {
        Network::new(input_size, hidden_size, output_size, beta, seed)
    };

    network.lif1.spike_grad = SurrogateGradient::fast_sigmoid(slope);
    network.lif2.spike_grad = SurrogateGradient::fast_sigmoid(slope);

    if cfg.hardware.enable_current_caps {
        let syn_scale = cfg.hardware.synapse_scale_from_baseline();
        let pos_gain = DEFAULT_SYNAPSE_POS_GAIN * syn_scale;
        let neg_gain = DEFAULT_SYNAPSE_NEG_GAIN * syn_scale;
        let total_cap = cfg.hardware.total_cap_units();

        network.fc1.synapse_pos_gain = pos_gain;
        network.fc1.synapse_neg_gain = neg_gain;
        network.fc2.synapse_pos_gain = pos_gain;
        network.fc2.synapse_neg_gain = neg_gain;
        network.fc1.total_current_cap = total_cap;
        network.fc2.total_current_cap = total_cap;

        let total_cap_text = total_cap
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "none".to_string());
        println!(
            "Current caps:   enabled (I_syn_max={:.2}uA, I_total_max={:.2}uA, syn_scale={:.3}, total_cap_units={})",
            cfg.hardware.synapse_current_max_ua,
            cfg.hardware.total_current_max_ua,
            syn_scale,
            total_cap_text
        );
    }

    if cfg.input_encoding.encoding_type == EncodingType::Spiking {
        network.spiking_input = true;
        println!("Input encoding: spiking (deterministic accumulator)");
    }

    println!("Network architecture:");
    println!("  Input:  {}", input_size);
    if network.is_physics_mode() {
        println!("  Hidden: {} (LIF, physics mode)", hidden_size);
        println!("  Output: {} (LIF, physics mode)", output_size);
    } else {
        println!("  Hidden: {} (LIF, beta={})", hidden_size, beta);
        println!("  Output: {} (LIF, beta={})", output_size, beta);
    }
    println!("  Total parameters: {}", network.num_parameters());
    println!();

    let train_config = TrainingConfig {
        lr: cfg.training.lr,
        epochs: cfg.training.epochs,
        batch_size: cfg.training.batch_size,
        num_steps: cfg.training.num_steps,
        seed: cfg.training.seed,
        num_workers: cfg.training.num_workers,
        bptt_steps: cfg.training.bptt_steps,
    };
    let mut trainer = Trainer::new(network, train_config);

    if cfg.quantization.enabled {
        trainer.quant_bits = Some(cfg.quantization.bits);
    }
    if cfg.quantization.split_sign {
        trainer.split_sign_quant = true;
    }
    if cfg.quantization.input_bits > 0 {
        trainer.input_quant_bits = cfg.quantization.input_bits;
        println!("Input quant:    {}-bit DAC", cfg.quantization.input_bits);
    }

    if cfg.noise.enabled {
        trainer.noise = NoiseParams {
            weight_std: cfg.noise.weight_std,
            threshold_std: cfg.noise.threshold_std,
            membrane_std: cfg.noise.membrane_std,
            input_std: cfg.noise.input_std,
        };
        if !cfg.noise.training_only {
            trainer.noise_during_eval = true;
        }
    }

    if cfg.physics.adaptation_enabled {
        trainer.adaptation_enabled = true;
        trainer.dt = cfg.physics.dt;
    }

    if cfg.output.analog_gain > 0.0 {
        trainer.analog_gain = cfg.output.analog_gain;
        println!("Analog gain:    {}", cfg.output.analog_gain);
    }

    if let Some(encoder) = input_encoder {
        println!(
            "Input encoding: temporal (row_spacing={:.4}s, pulse_width={:.1}%, input_dim={})",
            cfg.input_encoding.row_spacing,
            cfg.input_encoding.pulse_width * 100.0,
            encoder.output_dim()
        );
        trainer.input_encoder = Some(encoder);
    }

    if let Some(bptt) = cfg.training.bptt_steps {
        if bptt > 0 {
            println!("BPTT truncation: {} steps", bptt);
        }
    }

    let use_lr_schedule = cfg.training.epochs > 1;
    if use_lr_schedule {
        trainer.lr_scheduler = Some(gilgamesh::training::LRScheduler::new(
            cfg.training.lr,
            cfg.training.epochs,
        ));
        println!(
            "LR schedule:    cosine annealing ({:.6} -> {:.6})",
            cfg.training.lr,
            cfg.training.lr * 0.01
        );
    }

    trainer.max_grad_norm = Some(1.0);
    println!("Grad clipping:  max_norm=1.0");

    trainer.optimizer.set_weight_decay(0.01);
    println!("Weight decay:   0.01 (AdamW)");

    #[cfg(feature = "visualization")]
    let mut recorder = if visualize {
        let rec = if let Some(ref path) = visualize_file {
            println!("Visualization:  saving to {}", path);
            gilgamesh::TrainingRecorder::to_file("gilgamesh", path, false, 1)?
        } else {
            println!("Visualization:  spawning Rerun viewer");
            gilgamesh::TrainingRecorder::new("gilgamesh", false, 1)?
        };
        rec.log_architecture(input_size, hidden_size, output_size, &cfg.mode)?;
        Some(rec)
    } else {
        None
    };

    #[cfg(not(feature = "visualization"))]
    let mut recorder: Option<gilgamesh::TrainingRecorder> = {
        let _ = &visualize_file;
        if visualize {
            println!(
                "Warning: visualization requested but feature not enabled. Rebuild with --features visualization"
            );
        }
        None
    };

    println!("Training...");
    println!("{:-<60}", "");

    let mut best_test_acc = 0.0f32;

    for epoch in 1..=cfg.training.epochs {
        let lr = trainer
            .update_lr_for_epoch(epoch)
            .unwrap_or(cfg.training.lr);

        let (train_loss, train_acc) = trainer.train_epoch(&dataset);
        let test_acc = trainer.evaluate(&dataset);

        if test_acc > best_test_acc {
            best_test_acc = test_acc;
        }

        println!(
            "Epoch {:3} | Loss: {:.4} | Train Acc: {:5.2}% | Test Acc: {:5.2}% | LR: {:.6}",
            epoch, train_loss, train_acc, test_acc, lr
        );

        if let Some(ref mut rec) = recorder {
            rec.set_epoch(epoch);
            let _ = rec.log_epoch_metrics(train_loss, train_acc, test_acc, lr);
            if epoch % 5 == 0 || epoch == 1 {
                let _ = rec.log_weights("fc1", &trainer.network.fc1.weight);
                let _ = rec.log_weights("fc2", &trainer.network.fc2.weight);
            }
        }
    }

    println!("{:-<60}", "");
    println!();
    println!("=== Training Complete ===");
    println!("Best Test Accuracy: {:.2}%", best_test_acc);
    let final_test_acc = trainer.evaluate(&dataset);
    println!("Final Test Accuracy: {:.2}%", final_test_acc);

    if let Some(ref checkpoint_path) = save_checkpoint {
        use gilgamesh::checkpoint::{Checkpoint, TrainingMetadata};

        let metadata = TrainingMetadata {
            epochs_trained: cfg.training.epochs,
            final_train_accuracy: best_test_acc,
            final_test_accuracy: final_test_acc,
            final_loss: None,
            config_file: None,
        };

        // Include quantized weights for hardware deployment
        // Use configured bits if quantization enabled, otherwise default to 4-bit
        let quant_bits = Some(cfg.quantization.bits);

        // Store image dimensions for correct evaluation later
        let image_dims = Some((image_width, image_height));

        let checkpoint = Checkpoint::from_network_quantized(
            &trainer.network,
            Some(metadata),
            quant_bits,
            image_dims,
        );
        checkpoint
            .save(checkpoint_path)
            .with_context(|| format!("Failed to save checkpoint to {}", checkpoint_path))?;

        println!("Checkpoint saved to: {}", checkpoint_path);
        println!(
            "Quantized weights included: {}-bit magnitude integers for hardware",
            cfg.quantization.bits
        );
    }

    Ok(())
}
