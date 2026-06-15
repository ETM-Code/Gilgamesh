//! Threshold-scan diagnostic for the HW forward path (read-only).
//!
//! Reuses the EXACT fc1_for_hw preprocessing from finetune.rs/oracle_dump.rs,
//! then sweeps per-layer (hidden_theta, output_theta) absolute membrane
//! thresholds through `hw_forward_batch_with_thresholds` and reports test
//! accuracy on the full requested sample. Used to find a physically-grounded
//! output threshold (i.e. a smaller R_bottom divider on the output layer)
//! under which outputs actually fire and classify.
//!
//! Usage:
//!   cargo run --release --bin theta_scan -- <ckpt.json> <data_dir> <n> [dac_scale] [num_steps]

use gilgamesh::checkpoint::Checkpoint;
use gilgamesh::data::MnistDataset;
use gilgamesh::hw_forward::{hw_forward_batch_with_thresholds, r_bottom_for_theta, theta_from_r_bottom};
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

fn accuracy(
    fc1: &Array2<f32>,
    fc2: &[Vec<i8>],
    labels: &[usize],
    dac_scale: f64,
    num_steps: usize,
    h_theta: f64,
    o_theta: f64,
) -> (f64, usize) {
    let counts = hw_forward_batch_with_thresholds(fc1, fc2, dac_scale, num_steps, h_theta, o_theta);
    let mut correct = 0usize;
    let mut nonzero = 0usize;
    for i in 0..fc1.nrows() {
        let row = counts.row(i);
        let total: f32 = row.iter().sum();
        if total > 0.0 {
            nonzero += 1;
        }
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(j, _)| j)
            .unwrap();
        if argmax == labels[i] {
            correct += 1;
        }
    }
    (correct as f64 / fc1.nrows() as f64 * 100.0, nonzero)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = &args[1];
    let data_dir = &args[2];
    let n: usize = args[3].parse().unwrap();
    let dac_scale: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1.16);
    let num_steps: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(25);

    let cp = Checkpoint::load(ckpt).expect("load ckpt");
    let network = cp.to_network().expect("to_network");
    let quant_bits: Option<u8> = cp.metadata.as_ref().and_then(|m| m.quant_bits);
    let input_quant_bits = cp
        .metadata
        .as_ref()
        .and_then(|m| m.input_quant_bits)
        .unwrap_or(0);

    let fc2: Vec<Vec<i8>> = {
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

    let h_nom = theta_from_r_bottom(220e3);
    println!(
        "# theta_scan ckpt={ckpt} dac_scale={dac_scale} num_steps={num_steps} n={n} hidden_theta(nom 220k)={h_nom:.4}V"
    );
    println!("# col: hidden_theta_V, output_theta_V, output_Rbottom_ohm, acc%, nonzero_rows");

    // 2D refine: co-vary hidden and output thresholds over physically-realisable
    // divider voltages. Track and report the best (h_theta, o_theta).
    let h_thetas = [1.0577, 0.95, 0.85, 0.75, 0.65, 0.55];
    let out_thetas = [
        0.70, 0.65, 0.62, 0.60, 0.58, 0.55, 0.50, 0.45, 0.40, 0.30, 0.25, 0.22,
    ];
    let mut best = (0.0f64, h_nom, h_nom, 0usize);
    for &ht in &h_thetas {
        for &ot in &out_thetas {
            let (acc, nz) = accuracy(&fc1, &fc2, &labels, dac_scale, num_steps, ht, ot);
            let rb = r_bottom_for_theta(ot);
            println!("{ht:.4},{ot:.4},{rb:.0},{acc:.2},{nz}");
            if acc > best.0 {
                best = (acc, ht, ot, nz);
            }
        }
    }
    println!(
        "# BEST acc={:.2}% at hidden_theta={:.4}V output_theta={:.4}V (out_Rbottom={:.0}ohm) nonzero={}",
        best.0,
        best.1,
        best.2,
        r_bottom_for_theta(best.2),
        best.3
    );
}
