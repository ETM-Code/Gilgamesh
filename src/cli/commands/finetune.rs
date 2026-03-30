//! Fine-tune a trained model using the hardware emulator forward pass.
//!
//! Strategy: perturbation-based optimization on fc1 weights.
//! For each weight, try ±delta, keep the change that improves hardware accuracy.
//! This is simple, gradient-free, and directly optimizes board accuracy.

use ndarray::Array2;
use std::path::PathBuf;

use gilgamesh::checkpoint::Checkpoint;
use gilgamesh::config::Config;
use gilgamesh::data::MnistDataset;
use gilgamesh::hw_forward::hw_forward_batch;
pub fn run_finetune(
    checkpoint_path: &PathBuf,
    config_path: &PathBuf,
    data_dir: &PathBuf,
    output_path: &PathBuf,
    dac_scale: f64,
    epochs: usize,
) {
    let cfg = Config::load(config_path.to_str().unwrap()).expect("Failed to load config");

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

    println!("Loaded: {}", checkpoint_path.display());
    println!("DAC scale: {:.2}×", dac_scale);
    println!("Epochs: {}", epochs);

    // Load MNIST
    let dataset = MnistDataset::load_with_size(data_dir.to_str().unwrap(), cfg.network.image_size)
        .expect("Failed to load MNIST");

    let n_test = dataset.test_len().min(2000);
    let test_indices: Vec<usize> = (0..n_test).collect();
    let (test_images, test_labels) = dataset.get_test_batch(&test_indices);

    let n_train = dataset.train_len().min(5000); // use subset for speed
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

fn hw_accuracy(
    network: &gilgamesh::network::Network,
    images: &Array2<f32>,
    labels: &[usize],
    fc2_quantized: &[Vec<i8>],
    dac_scale: f64,
    num_steps: usize,
) -> f64 {
    let fc1_out = network.fc1.forward(images);
    let hw_counts = hw_forward_batch(&fc1_out, fc2_quantized, dac_scale, num_steps);

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
