//! Characterization tests for the vectorized batch wrappers in src/surrogate.rs
//! (forward_batch / backward_batch / SpikeFunction::apply_batch), which the
//! existing surrogate_golden.rs leaves untested.

mod common;
use common::*;

use gilgamesh::surrogate::{SpikeFunction, SurrogateGradient};

#[test]
fn forward_batch_equals_scalar_map() {
    let sg = SurrogateGradient::fast_sigmoid(25.0);
    let xs = [-1.0f32, -0.1, 0.0, 0.1, 1.0];
    let batch = sg.forward_batch(&xs);
    let scalar: Vec<f32> = xs.iter().map(|&v| sg.forward(v)).collect();
    assert_eq!(batch, scalar);
    // Heaviside (strict > 0) golden.
    assert_eq!(batch, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn backward_batch_equals_scalar_map() {
    let sg = SurrogateGradient::fast_sigmoid(25.0);
    let xs = [-1.0f32, -0.1, 0.0, 0.1, 1.0];
    let batch = sg.backward_batch(&xs);
    let scalar: Vec<f32> = xs.iter().map(|&v| sg.backward(v)).collect();
    close_vec(&batch, &scalar, 0.0);
    close_vec(
        &batch,
        &[0.00147929, 0.08163265, 1.0, 0.08163265, 0.00147929],
        1e-6,
    );
}

#[test]
fn backward_batch_matches_scalar_for_all_variants() {
    let variants = [
        SurrogateGradient::fast_sigmoid(25.0),
        SurrogateGradient::atan(2.0),
        SurrogateGradient::sigmoid(25.0),
        SurrogateGradient::StraightThrough,
        SurrogateGradient::Triangular { threshold: 0.5 },
    ];
    let xs = [-0.7f32, -0.05, 0.0, 0.05, 0.7];
    for v in variants {
        let batch = v.backward_batch(&xs);
        let scalar: Vec<f32> = xs.iter().map(|&x| v.backward(x)).collect();
        close_vec(&batch, &scalar, 0.0);
    }
}

#[test]
fn spike_function_apply_batch_equals_loop() {
    let sf = SpikeFunction::new(SurrogateGradient::fast_sigmoid(25.0), 1.0);
    let mem = [0.5f32, 1.0, 1.5, 2.0];
    let (spikes, grads) = sf.apply_batch(&mem);

    // Compare to scalar apply over the same vector.
    let mut exp_s = Vec::new();
    let mut exp_g = Vec::new();
    for &m in &mem {
        let (s, g) = sf.apply(m);
        exp_s.push(s);
        exp_g.push(g);
    }
    assert_eq!(spikes, exp_s);
    close_vec(&grads, &exp_g, 0.0);

    // Golden values.
    assert_eq!(spikes, vec![0.0, 0.0, 1.0, 1.0]);
    close_vec(
        &grads,
        &[0.0054869684, 1.0, 0.0054869684, 0.00147929],
        1e-6,
    );
}
