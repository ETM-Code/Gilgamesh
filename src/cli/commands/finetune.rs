//! Fine-tune a trained model using the hardware emulator forward pass.
//!
//! Strategy: perturbation-based optimization on fc1 weights.
//! For each weight, try ±delta, keep the change that improves hardware accuracy.
//! This is simple, gradient-free, and directly optimizes board accuracy.

use ndarray::Array2;
use std::path::PathBuf;

use gilgamesh::checkpoint::Checkpoint;
use gilgamesh::config::Config;
use gilgamesh::data::{AnyDataset, ArrayDataset, Dataset, MnistDataset};
use gilgamesh::hw_forward::{
    hw_forward_batch_cfg, theta_from_r_bottom, HwForwardConfig, R_BOTTOM_NOMINAL,
    R_BOTTOM_OUTPUT_FAITHFUL,
};
use gilgamesh::layers::linear::quantize_input;

#[derive(Clone, Copy, Debug, Default)]
struct FineTunePreprocess {
    quant_bits: Option<u8>,
    input_quant_bits: u8,
}

pub fn run_finetune(
    checkpoint_path: &PathBuf,
    config_path: &PathBuf,
    data_dir: &PathBuf,
    dataset_kind: &str,
    output_path: &PathBuf,
    dac_scale: f64,
    epochs: usize,
    train_limit: Option<usize>,
    test_limit: Option<usize>,
    hidden_theta_opt: Option<f64>,
    output_theta_opt: Option<f64>,
) {
    let cfg = Config::load(config_path.to_str().unwrap()).expect("Failed to load config");

    // Per-layer absolute membrane thresholds (GND-referenced divider voltages).
    // Defaults are the FAITHFUL as-built board ([`HwForwardConfig::default`]):
    // hidden at the nominal 1.0577 V (220k R_bottom), output at the real ~0.543 V
    // (150k‖300k = 100k) — the output membrane only reaches ~0.5 V on the weak
    // synapse mirror, so the nominal 1.058 V is unreachable. Both still overridable.
    let hidden_theta = hidden_theta_opt.unwrap_or_else(|| theta_from_r_bottom(R_BOTTOM_NOMINAL));
    let output_theta =
        output_theta_opt.unwrap_or_else(|| theta_from_r_bottom(R_BOTTOM_OUTPUT_FAITHFUL));

    println!("=== Hardware-in-the-Loop Fine-Tuning ===\n");

    let cp = Checkpoint::load(checkpoint_path).expect("Failed to load checkpoint");
    let mut network = cp.to_network().expect("Failed to build network");

    // Get quantized fc2 weights from checkpoint JSON directly
    let fc2_quantized: Vec<Vec<i8>> = {
        let json_str = std::fs::read_to_string(checkpoint_path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        v["quantized"]["fc2_weight"]
            .as_array()
            .expect("Checkpoint needs quantized fc2_weight")
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i8)
                    .collect()
            })
            .collect()
    };

    let num_steps = cfg.training.num_steps;
    let preprocess = FineTunePreprocess {
        quant_bits: cp.metadata.as_ref().and_then(|m| m.quant_bits),
        input_quant_bits: cp
            .metadata
            .as_ref()
            .and_then(|m| m.input_quant_bits)
            .unwrap_or(0),
    };

    println!("Loaded: {}", checkpoint_path.display());
    println!("Dataset: {}", dataset_kind);
    println!("DAC scale: {:.2}×", dac_scale);
    println!(
        "Thresholds (V): hidden={:.4}  output={:.4}",
        hidden_theta, output_theta
    );
    println!(
        "Masked outputs (faithful as-built): {:?}",
        HwForwardConfig::default().masked_outputs
    );
    println!("Epochs: {}", epochs);
    if let Some(bits) = preprocess.quant_bits {
        println!("Weight quant: {}-bit", bits);
    }
    if preprocess.input_quant_bits > 0 {
        println!("Input quant: {}-bit DAC", preprocess.input_quant_bits);
    }

    let dataset = match dataset_kind {
        "mnist" => AnyDataset::Mnist(
            MnistDataset::load_with_dimensions(
                data_dir.to_str().unwrap(),
                cfg.network.get_width(),
                cfg.network.get_height(),
            )
            .expect("Failed to load MNIST"),
        ),
        "ecg" => AnyDataset::Array(
            ArrayDataset::load_npy(data_dir.to_str().unwrap())
                .expect("Failed to load ECG dataset (.npy)"),
        ),
        other => panic!("Unsupported dataset kind: {other}. Use 'mnist' or 'ecg'."),
    };

    let default_test = if dataset_kind == "ecg" { 5000 } else { 2000 };
    let n_test = dataset.test_len().min(test_limit.unwrap_or(default_test));
    let test_indices: Vec<usize> = (0..n_test).collect();
    let (test_images, test_labels) = dataset.get_test_batch(&test_indices);

    let default_train = if dataset_kind == "ecg" { 20000 } else { 5000 };
    let n_train = dataset.train_len().min(train_limit.unwrap_or(default_train));
    let train_indices: Vec<usize> = (0..n_train).collect();
    let (train_images, train_labels) = dataset.get_train_batch(&train_indices);

    // Evaluate initial hardware accuracy
    let initial_acc = hw_accuracy(
        &network,
        &test_images,
        &test_labels,
        &fc2_quantized,
        dac_scale,
        num_steps,
        preprocess,
        hidden_theta,
        output_theta,
    );
    println!(
        "Initial HW accuracy: {:.1}% ({} test samples)\n",
        initial_acc, n_test
    );

    let mut best_acc = initial_acc;
    let (n_inputs, n_hidden) = (network.fc1.weight.nrows(), network.fc1.weight.ncols());

    let mut best_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(checkpoint_path).unwrap()).unwrap();

    for epoch in 0..epochs {
        let mut improvements = 0;
        // Perturbation size decreases over epochs
        let delta = 0.02 * (0.7f32).powi(epoch as i32);

        for i in 0..n_inputs {
            for j in 0..n_hidden {
                let original = network.fc1.weight[[i, j]];
                // Local baseline at current point for this coordinate.
                let base_acc = hw_accuracy(
                    &network,
                    &train_images,
                    &train_labels,
                    &fc2_quantized,
                    dac_scale,
                    num_steps,
                    preprocess,
                    hidden_theta,
                    output_theta,
                );

                // Try +delta
                network.fc1.weight[[i, j]] = original + delta;
                let acc_plus = hw_accuracy(
                    &network,
                    &train_images,
                    &train_labels,
                    &fc2_quantized,
                    dac_scale,
                    num_steps,
                    preprocess,
                    hidden_theta,
                    output_theta,
                );

                // Try -delta
                network.fc1.weight[[i, j]] = original - delta;
                let acc_minus = hw_accuracy(
                    &network,
                    &train_images,
                    &train_labels,
                    &fc2_quantized,
                    dac_scale,
                    num_steps,
                    preprocess,
                    hidden_theta,
                    output_theta,
                );

                // Keep the best
                if acc_plus >= base_acc && acc_plus >= acc_minus {
                    network.fc1.weight[[i, j]] = original + delta;
                    if acc_plus > base_acc {
                        improvements += 1;
                    }
                } else if acc_minus >= base_acc {
                    network.fc1.weight[[i, j]] = original - delta;
                    if acc_minus > base_acc {
                        improvements += 1;
                    }
                } else {
                    network.fc1.weight[[i, j]] = original;
                }
            }
        }

        let test_acc = hw_accuracy(
            &network,
            &test_images,
            &test_labels,
            &fc2_quantized,
            dac_scale,
            num_steps,
            preprocess,
            hidden_theta,
            output_theta,
        );
        println!(
            "Epoch {:>2} | delta={:.4} | improvements={}/{} | HW Test: {:.1}%",
            epoch + 1,
            delta,
            improvements,
            n_inputs * n_hidden,
            test_acc
        );

        if test_acc > best_acc {
            best_acc = test_acc;
            // Save: update fc1 weights in the original checkpoint JSON
            let mut v = best_json.clone();
            let fc1_vec: Vec<Vec<f32>> = network
                .fc1
                .weight
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();
            v["weights"]["fc1_weight"] = serde_json::to_value(&fc1_vec).unwrap();
            std::fs::write(output_path, serde_json::to_string_pretty(&v).unwrap())
                .expect("Failed to save");
            best_json = v;
        }
    }

    // Ensure output exists even when no epoch improves test accuracy.
    if !output_path.exists() {
        std::fs::write(
            output_path,
            serde_json::to_string_pretty(&best_json).unwrap(),
        )
        .expect("Failed to save");
    }

    println!(
        "\nBest HW test accuracy: {:.1}% (was {:.1}%)",
        best_acc, initial_acc
    );
    println!("Saved to: {}", output_path.display());
}

fn fc1_for_hw(
    network: &gilgamesh::network::Network,
    images: &Array2<f32>,
    preprocess: FineTunePreprocess,
) -> Array2<f32> {
    let mut input = images.clone();
    if preprocess.input_quant_bits > 0 {
        input = quantize_input(&input, preprocess.input_quant_bits);
    }

    let mut fc1 = match preprocess.quant_bits {
        Some(bits) => network.fc1.forward_quantized(&input, bits),
        None => network.fc1.forward(&input),
    };

    if network.dac_max.is_finite() {
        fc1.mapv_inplace(|v| v.clamp(0.0, network.dac_max));
    }
    fc1
}

fn hw_accuracy(
    network: &gilgamesh::network::Network,
    images: &Array2<f32>,
    labels: &[usize],
    fc2_quantized: &[Vec<i8>],
    dac_scale: f64,
    num_steps: usize,
    preprocess: FineTunePreprocess,
    hidden_theta: f64,
    output_theta: f64,
) -> f64 {
    let fc1_out = fc1_for_hw(network, images, preprocess);
    // Faithful as-built board: the matched-mirror default (which masks O2/O10),
    // with the explicit per-layer thresholds / scale / steps for this run.
    let cfg = HwForwardConfig {
        hidden_theta,
        output_theta,
        dac_scale,
        num_steps,
        ..HwForwardConfig::default()
    };
    let hw_counts = hw_forward_batch_cfg(&fc1_out, fc2_quantized, &cfg);

    let mut correct = 0;
    for i in 0..images.nrows() {
        let pred = hw_counts
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        if pred == labels[i] {
            correct += 1;
        }
    }
    correct as f64 / images.nrows() as f64 * 100.0
}
