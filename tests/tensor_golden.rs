//! Characterization tests for src/tensor.rs

mod common;
use common::*;

use gilgamesh::tensor::{
    cross_entropy_loss, cross_entropy_loss_weighted, heaviside, heaviside_array, linear_forward,
    softmax,
};
use ndarray::array;

#[test]
fn softmax_golden_and_row_sums() {
    let out = softmax(&array![[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]);
    close_arr(
        &out,
        &[
            0.09003057, 0.24472848, 0.66524094, // exp-normalized
            0.33333334, 0.33333334, 0.33333334, // uniform
        ],
        1e-6,
    );
    // rows sum to 1
    for row in out.rows() {
        close32(row.sum(), 1.0, 1e-6);
    }
}

#[test]
fn cross_entropy_loss_golden() {
    let (loss, grad) = cross_entropy_loss(&array![[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]], &[0, 2]);
    close32(loss, 0.41703004, 1e-6);
    close_arr(
        &grad,
        &[
            -0.17049941, 0.1212165, 0.04928295, 0.049282946, 0.12121649, -0.17049944,
        ],
        1e-6,
    );
}

#[test]
fn cross_entropy_loss_weighted_golden() {
    let (loss, grad) =
        cross_entropy_loss_weighted(&array![[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]], &[0, 2], &[1.0, 1.0, 2.0]);
    close32(loss, 0.625545, 1e-6);
    // Row 1 (target class 2, weight 2.0) is scaled by the class weight.
    close_arr(
        &grad,
        &[
            -0.17049941, 0.1212165, 0.04928295, // row0 weight 1
            0.09856589, 0.24243298, -0.3409989, // row1 weight 2
        ],
        1e-6,
    );
}

#[test]
fn heaviside_strict_greater_than_zero() {
    assert_eq!(heaviside(0.0), 0.0);
    assert_eq!(heaviside(-0.0), 0.0);
    assert_eq!(heaviside(1e-30), 1.0);
    assert_eq!(heaviside(-1e-30), 0.0);
}

#[test]
fn heaviside_array_matches_scalar() {
    let out = heaviside_array(&array![[-1.0, 0.0, 0.5], [1e-30, -1e-30, 2.0]]);
    eq_arr(&out, &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
}

#[test]
fn linear_forward_with_bias_golden() {
    let out = linear_forward(
        &array![[1.0, 2.0]],
        &array![[1.0, 0.5, 2.0], [-1.0, 3.0, 0.5]],
        Some(&array![0.1, 0.2, 0.3]),
    );
    close_arr(&out, &[-0.9, 6.7, 3.3], 1e-6);
}
