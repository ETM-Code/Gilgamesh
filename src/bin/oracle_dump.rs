//! Authoritative gilgamesh HW oracle dump (read-only verification tool).
//!
//! Mirrors src/cli/commands/finetune.rs `fc1_for_hw` + `hw_accuracy` EXACTLY:
//! load checkpoint -> to_network -> fc2_quantized from JSON ->
//! preprocess(quant_bits, input_quant_bits) -> MNIST 6x6 ->
//! fc1_for_hw -> hw_forward_batch(dac_scale, num_steps).
//!
//! Prints, for the first N test digits: true label, 10-element output spike
//! count vector, hw argmax, and whether argmax==label. Reports sample accuracy.
//!
//! Usage: cargo run --release --bin oracle_dump -- <checkpoint.json> <data_dir> <n> [dac_scale] [num_steps] [output_theta_V] [hidden_theta_V]
//!
//! `output_theta_V` / `hidden_theta_V` are optional absolute membrane
//! thresholds (GND-referenced divider voltages). When omitted, both default to
//! the legacy shared nominal theta_0 = theta_from_r_bottom(220k) = 1.0577 V,
//! exactly reproducing the original behaviour. The Tarski HW reference config
//! uses output_theta = 0.62 V (output R_bottom ≈ 116 kΩ).

use gilgamesh::checkpoint::Checkpoint;
use gilgamesh::data::MnistDataset;
use gilgamesh::hw_forward::{hw_forward_batch_with_thresholds, theta_from_r_bottom};
use gilgamesh::layers::linear::quantize_input;
use ndarray::Array2;

fn fc1_for_hw(
    network: &gilgamesh::network::Network,
    images: &Array2<f32>,
    quant_bits: Option<u8>,
    input_quant_bits: u8,
) -> Array2<f32> {
    let mut input = images.clone();
    if input_quant_bits > 0 {
        input = quantize_input(&input, input_quant_bits);
    }
    let mut fc1 = match quant_bits {
        Some(bits) => network.fc1.forward_quantized(&input, bits),
        None => network.fc1.forward(&input),
    };
    if network.dac_max.is_finite() {
        fc1.mapv_inplace(|v| v.clamp(0.0, network.dac_max));
    }
    fc1
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = &args[1];
    let data_dir = &args[2];
    let n: usize = args[3].parse().unwrap();
    let dac_scale: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1.16);
    let num_steps: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(25);
    let nominal_theta = theta_from_r_bottom(220e3);
    let output_theta: f64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(nominal_theta);
    let hidden_theta: f64 = args.get(7).map(|s| s.parse().unwrap()).unwrap_or(nominal_theta);

    let cp = Checkpoint::load(ckpt).expect("load ckpt");
    let network = cp.to_network().expect("to_network");
    let quant_bits: Option<u8> = cp.metadata.as_ref().and_then(|m| m.quant_bits);
    let input_quant_bits = cp
        .metadata
        .as_ref()
        .and_then(|m| m.input_quant_bits)
        .unwrap_or(0);

    let fc2_quantized: Vec<Vec<i8>> = {
        let s = std::fs::read_to_string(ckpt).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        v["quantized"]["fc2_weight"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap() as i8)
                    .collect()
            })
            .collect()
    };

    let w = cp.architecture.image_width.unwrap_or(6);
    let h = cp.architecture.image_height.unwrap_or(6);
    let ds = MnistDataset::load_with_dimensions(data_dir, w, h).expect("load mnist");
    let idx: Vec<usize> = (0..n).collect();
    let (imgs, labels) = ds.get_test_batch(&idx);

    let fc1 = fc1_for_hw(&network, &imgs, quant_bits, input_quant_bits);
    let counts = hw_forward_batch_with_thresholds(
        &fc1,
        &fc2_quantized,
        dac_scale,
        num_steps,
        hidden_theta,
        output_theta,
    );

    println!(
        "# oracle_dump ckpt={ckpt} dac_scale={dac_scale} num_steps={num_steps} hidden_theta={hidden_theta:.4}V output_theta={output_theta:.4}V quant_bits={:?} input_quant_bits={input_quant_bits} dac_max={} img={}x{}",
        quant_bits, network.dac_max, w, h
    );
    println!("# columns: idx,label,argmax,match,counts[0..10],fc1");
    let mut correct = 0usize;
    let mut nonzero = 0usize;
    for i in 0..n {
        let row: Vec<f32> = counts.row(i).to_vec();
        let total: f32 = row.iter().sum();
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(j, _)| j)
            .unwrap();
        let lbl = labels[i];
        let m = argmax == lbl;
        if m {
            correct += 1;
        }
        if total > 0.0 {
            nonzero += 1;
        }
        let fc1row: Vec<f32> = fc1.row(i).iter().map(|v| (v * 1000.0).round() / 1000.0).collect();
        let crow: Vec<i32> = row.iter().map(|v| *v as i32).collect();
        println!(
            "{i},{lbl},{argmax},{},{crow:?},{fc1row:?}",
            if m { "Y" } else { "N" }
        );
    }
    println!(
        "# SAMPLE_ACC={:.2}% ({}/{}) nonzero_count_rows={}/{}",
        correct as f64 / n as f64 * 100.0,
        correct,
        n,
        nonzero,
        n
    );
}
