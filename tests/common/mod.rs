//! Shared float-tolerance helpers for the gilgamesh characterization suite.
//!
//! Every golden constant in this suite was produced by RUNNING the current
//! code (see `examples/golden_capture.rs`) and pasting the observed value.
//! The purpose is to lock current behavior so any future change is detected.

#![allow(dead_code)]

use ndarray::Array2;

/// Assert two f32 scalars are within `tol`.
pub fn close32(a: f32, b: f32, tol: f32) {
    assert!(
        (a - b).abs() <= tol,
        "expected {b}, got {a} (diff {}, tol {tol})",
        (a - b).abs()
    );
}

/// Assert two f64 scalars are within `tol`.
pub fn close64(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "expected {b}, got {a} (diff {}, tol {tol})",
        (a - b).abs()
    );
}

/// Assert a flattened Array2<f32> matches an expected slice within `tol`.
pub fn close_arr(actual: &Array2<f32>, expected: &[f32], tol: f32) {
    let flat: Vec<f32> = actual.iter().cloned().collect();
    assert_eq!(
        flat.len(),
        expected.len(),
        "length mismatch: got {} expected {}",
        flat.len(),
        expected.len()
    );
    for (i, (a, e)) in flat.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: expected {e}, got {a} (diff {}, tol {tol})",
            (a - e).abs()
        );
    }
}

/// Assert a flattened Array2<f32> equals an expected slice exactly (for integer
/// spike counts which are exact f32 values).
pub fn eq_arr(actual: &Array2<f32>, expected: &[f32]) {
    let flat: Vec<f32> = actual.iter().cloned().collect();
    assert_eq!(flat, expected, "array mismatch");
}

/// Assert a Vec<f32> matches expected within tol.
pub fn close_vec(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: expected {e}, got {a}"
        );
    }
}
