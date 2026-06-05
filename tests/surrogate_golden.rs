//! Characterization tests for src/surrogate.rs

mod common;
use common::*;

use gilgamesh::surrogate::{SpikeFunction, SurrogateGradient};

#[test]
fn backward_fast_sigmoid_golden() {
    let sg = SurrogateGradient::fast_sigmoid(25.0);
    close32(sg.backward(-1.0), 0.00147929, 1e-6);
    close32(sg.backward(-0.1), 0.08163265, 1e-6);
    close32(sg.backward(0.0), 1.0, 1e-6);
    close32(sg.backward(0.1), 0.08163265, 1e-6);
    close32(sg.backward(1.0), 0.00147929, 1e-6);
}

#[test]
fn backward_atan_golden() {
    let sg = SurrogateGradient::atan(2.0);
    close32(sg.backward(-1.0), 0.091999665, 1e-6);
    close32(sg.backward(-0.1), 0.9101699, 1e-6);
    close32(sg.backward(0.0), 1.0, 1e-6);
    close32(sg.backward(0.1), 0.9101699, 1e-6);
    close32(sg.backward(1.0), 0.091999665, 1e-6);
}

#[test]
fn backward_sigmoid_golden() {
    let sg = SurrogateGradient::sigmoid(25.0);
    close32(sg.backward(-1.0), 3.4719858e-10, 1e-15);
    close32(sg.backward(-0.1), 1.7525928, 1e-5);
    close32(sg.backward(0.0), 6.25, 1e-5);
    close32(sg.backward(0.1), 1.7525929, 1e-5);
    close32(sg.backward(1.0), 3.471986e-10, 1e-15);
}

#[test]
fn backward_straight_through_is_one() {
    let sg = SurrogateGradient::StraightThrough;
    for v in [-1.0, -0.1, 0.0, 0.1, 1.0] {
        assert_eq!(sg.backward(v), 1.0);
    }
}

#[test]
fn backward_triangular_sign_flip_at_zero() {
    // CHARACTERIZATION: current behavior, possibly a bug, locked to detect change.
    // Triangular returns +threshold for x<0 and -threshold for x>=0 (flips at 0).
    let sg = SurrogateGradient::Triangular { threshold: 0.5 };
    close32(sg.backward(-1.0), 0.5, 1e-9);
    close32(sg.backward(-0.1), 0.5, 1e-9);
    close32(sg.backward(0.0), -0.5, 1e-9); // x >= 0 branch
    close32(sg.backward(0.1), -0.5, 1e-9);
    close32(sg.backward(1.0), -0.5, 1e-9);
}

#[test]
fn forward_heaviside_strict_for_all_variants() {
    let variants = [
        SurrogateGradient::fast_sigmoid(25.0),
        SurrogateGradient::atan(2.0),
        SurrogateGradient::sigmoid(25.0),
        SurrogateGradient::StraightThrough,
        SurrogateGradient::Triangular { threshold: 0.5 },
    ];
    for v in variants {
        assert_eq!(v.forward(0.0), 0.0, "forward(0) must be 0 (strict >0)");
        assert_eq!(v.forward(1e-6), 1.0);
        assert_eq!(v.forward(-1e-6), 0.0);
    }
}

#[test]
fn slope_accessor_per_variant() {
    assert_eq!(SurrogateGradient::fast_sigmoid(25.0).slope(), 25.0);
    assert_eq!(SurrogateGradient::atan(2.0).slope(), 2.0);
    assert_eq!(SurrogateGradient::sigmoid(13.0).slope(), 13.0);
    assert_eq!(SurrogateGradient::StraightThrough.slope(), 1.0);
    assert_eq!(SurrogateGradient::Triangular { threshold: 0.5 }.slope(), 0.5);
}

#[test]
fn default_is_atan_alpha_2() {
    match SurrogateGradient::default() {
        SurrogateGradient::ATan { alpha } => assert_eq!(alpha, 2.0),
        other => panic!("expected ATan{{2.0}}, got {other:?}"),
    }
}

#[test]
fn spike_function_apply_golden() {
    let sf = SpikeFunction::new(SurrogateGradient::fast_sigmoid(25.0), 1.0);
    // mem=0.5 -> shifted=-0.5 -> no spike
    let (s, g) = sf.apply(0.5);
    assert_eq!(s, 0.0);
    close32(g, 1.0 / (25.0 * 0.5 + 1.0_f32).powi(2), 1e-6);
    // mem==threshold -> shifted==0 -> >0 false -> spike 0.0
    let (s, g) = sf.apply(1.0);
    assert_eq!(s, 0.0);
    close32(g, 1.0, 1e-6);
    // mem=1.5 -> shifted=0.5 -> spike
    let (s, g) = sf.apply(1.5);
    assert_eq!(s, 1.0);
    close32(g, 1.0 / (25.0 * 0.5 + 1.0_f32).powi(2), 1e-6);
}

#[test]
fn serde_round_trip_all_variants() {
    let variants = [
        SurrogateGradient::fast_sigmoid(25.0),
        SurrogateGradient::atan(2.0),
        SurrogateGradient::sigmoid(25.0),
        SurrogateGradient::StraightThrough,
        SurrogateGradient::Triangular { threshold: 0.5 },
    ];
    for v in variants {
        let json = serde_json::to_string(&v).unwrap();
        let back: SurrogateGradient = serde_json::from_str(&json).unwrap();
        // structural equality via slope + discriminant behavior
        assert_eq!(back.slope(), v.slope());
        assert_eq!(back.backward(0.3), v.backward(0.3));
    }
    // Lock the exact JSON for one variant.
    assert_eq!(
        serde_json::to_string(&SurrogateGradient::fast_sigmoid(25.0)).unwrap(),
        r#"{"FastSigmoid":{"slope":25.0}}"#
    );
}
