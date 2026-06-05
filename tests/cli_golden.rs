//! Characterization test for the `gilgamesh test --quick` CLI path.
//!
//! Hermetic: no MNIST, no network checkpoint, no external services.
//! Uses the binary path Cargo injects via CARGO_BIN_EXE_gilgamesh.

use std::process::Command;

#[test]
fn cli_test_quick_smoke() {
    let bin = env!("CARGO_BIN_EXE_gilgamesh");
    let output = Command::new(bin)
        .args(["test", "--quick"])
        .output()
        .expect("failed to run gilgamesh test --quick");

    assert!(
        output.status.success(),
        "exit code was {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Fixed milestones.
    assert!(stdout.contains("All Tests Passed"), "missing pass banner");
    assert!(stdout.contains("Output spike count shape: [4, 10]"), "missing [4,10] shape");
    assert!(stdout.contains("Output membrane shape: [4, 10]"), "missing membrane shape");
    assert!(stdout.contains("FC1 weight grad shape: [49, 100]"), "missing FC1 grad shape");
    assert!(stdout.contains("FC2 weight grad shape: [100, 10]"), "missing FC2 grad shape");

    // CHARACTERIZATION: Physics-default neuron produces no spikes for the Test 2
    // probe input — locked as current (possibly surprising) behavior.
    assert!(
        stdout.contains("Spikes: [[0.0, 0.0, 0.0]]"),
        "expected the all-zero Physics-default spikes line, got:\n{stdout}"
    );
}
