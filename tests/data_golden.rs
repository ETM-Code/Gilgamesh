//! Characterization tests for src/data/mod.rs

mod common;
use common::*;

use gilgamesh::data::{ArrayDataset, BatchIterator, Dataset, InputEncoder, MnistDataset};
use ndarray::array;
use std::path::Path;

#[test]
fn input_encoder_rate_coded_passthrough() {
    let enc = InputEncoder::rate_coded(3);
    assert_eq!(enc.output_dim(), 9);
    assert!(!enc.is_temporal());
    let input = array![[10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0]];
    // Rate coded returns input clone at any timestep.
    let out = enc.encode_timestep(&input, 5);
    close_arr(&out, &[10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0], 0.0);
}

#[test]
fn input_encoder_temporal_row_selection_golden() {
    // temporal(3, row_spacing=0.002, pulse_width=0.9, dt=0.001).
    let enc = InputEncoder::temporal(3, 0.002, 0.9, 0.001);
    assert_eq!(enc.output_dim(), 3); // one row of 3 pixels
    assert!(enc.is_temporal());
    let input = array![[10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0]];

    // Each row is active for two consecutive timesteps; after the last row -> zeros.
    let expected = [
        [10.0, 11.0, 12.0], // t=0
        [10.0, 11.0, 12.0], // t=1
        [20.0, 21.0, 22.0], // t=2
        [20.0, 21.0, 22.0], // t=3
        [30.0, 31.0, 32.0], // t=4
        [30.0, 31.0, 32.0], // t=5
        [0.0, 0.0, 0.0],    // t=6 (past last row)
    ];
    for (t, exp) in expected.iter().enumerate() {
        let out = enc.encode_timestep(&input, t);
        close_arr(&out, exp, 0.0);
    }
}

#[test]
fn input_encoder_timesteps_needed_temporal() {
    // temporal(7, 0.0015, ...): total=0.0105, /dt(0.001)=10.5 -> ceil 11.
    let enc = InputEncoder::temporal(7, 0.0015, 0.9, 0.001);
    assert_eq!(enc.timesteps_needed(0), 11);
    // requested larger wins
    assert_eq!(enc.timesteps_needed(20), 20);
}

fn make_dataset() -> ArrayDataset {
    // 5 train rows, 3 test rows, 2 features. Deterministic content.
    let train_images = array![
        [0.0, 0.1],
        [1.0, 1.1],
        [2.0, 2.1],
        [3.0, 3.1],
        [4.0, 4.1]
    ];
    let test_images = array![[10.0, 10.1], [11.0, 11.1], [12.0, 12.1]];
    ArrayDataset {
        train_images,
        train_labels: vec![0, 1, 0, 1, 0],
        test_images,
        test_labels: vec![1, 0, 1],
    }
}

#[test]
fn array_dataset_dims() {
    let ds = make_dataset();
    assert_eq!(ds.train_len(), 5);
    assert_eq!(ds.test_len(), 3);
    assert_eq!(ds.feature_dim(), 2);
}

#[test]
fn batch_iterator_no_shuffle_order_and_boundaries() {
    let ds = make_dataset();
    let it = BatchIterator::new(&ds, 2, false, true);
    // ceil(5/2) = 3 batches
    assert_eq!(it.num_batches(), 3);

    let batches: Vec<_> = it.collect();
    assert_eq!(batches.len(), 3);

    // Batch 0: rows 0,1
    let (imgs0, labels0) = &batches[0];
    assert_eq!(imgs0.shape(), &[2, 2]);
    assert_eq!(labels0, &vec![0, 1]);
    close_arr(imgs0, &[0.0, 0.1, 1.0, 1.1], 1e-9);

    // Batch 1: rows 2,3
    let (imgs1, labels1) = &batches[1];
    assert_eq!(labels1, &vec![0, 1]);
    close_arr(imgs1, &[2.0, 2.1, 3.0, 3.1], 1e-9);

    // Batch 2: row 4 (remainder)
    let (imgs2, labels2) = &batches[2];
    assert_eq!(imgs2.shape(), &[1, 2]);
    assert_eq!(labels2, &vec![0]);
    close_arr(imgs2, &[4.0, 4.1], 1e-9);
}

#[test]
fn mnist_downsample_normalize_golden_path_guarded() {
    // Hermetic only if ./data/MNIST is present; otherwise skip.
    if !Path::new("./data").join("MNIST").exists() && !Path::new("./data/train-images-idx3-ubyte").exists()
    {
        eprintln!("Skipping MNIST test: ./data not present");
        return;
    }
    let ds = match MnistDataset::load("./data") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping MNIST test: load failed: {e}");
            return;
        }
    };
    assert_eq!(ds.train_len(), 60_000);
    assert_eq!(ds.test_len(), 10_000);
    assert_eq!(ds.feature_dim(), 36); // default 6x6 downsample
    assert_eq!(ds.image_size(), 6);

    // Sample 0 top-left corner: averaging-downsample then (x/255-0.1307)/0.3081.
    // The digit is centered so the corner pixels are background -> a fixed negative value.
    let (imgs, _labels) = ds.get_train_batch(&[0]);
    common::close32(imgs[[0, 0]], -0.42421296, 1e-5);
    for c in 0..6 {
        common::close32(imgs[[0, c]], -0.42421296, 1e-5);
    }
}

#[test]
fn batch_iterator_test_split_no_shuffle() {
    let ds = make_dataset();
    let it = BatchIterator::new(&ds, 2, false, false);
    assert_eq!(it.num_batches(), 2); // ceil(3/2)
    let batches: Vec<_> = it.collect();
    let (imgs0, labels0) = &batches[0];
    assert_eq!(labels0, &vec![1, 0]);
    close_arr(imgs0, &[10.0, 10.1, 11.0, 11.1], 1e-9);
}
