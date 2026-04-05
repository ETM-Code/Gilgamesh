# Gilgamesh Project Notes

**Always reference this document when working on this codebase.**

## Overview

Gilgamesh is a Rust-based implementation of hardware-accurate spiking neural networks (SNNs) designed for embedded deployment and chip simulation. It provides dual-mode operation supporting both simple beta-decay models (snnTorch-compatible) and physics-accurate RC membrane models. Achieves 96%+ accuracy on MNIST with minimal parameters for embedded systems.

## Directory Structure

```
gilgamesh/
├── Cargo.toml                 # Rust package manifest with feature flags
├── src/                       # Rust implementation (~5600 LOC)
│   ├── main.rs               # CLI entry point
│   ├── lib.rs                # Library exports
│   ├── cli/                  # Command-line interface
│   │   ├── mod.rs            # CLI parser and command routing
│   │   └── commands/         # Command implementations
│   │       ├── train.rs      # Training command
│   │       ├── animate.rs    # Interactive visualizer (nannou)
│   │       ├── dashboard.rs  # Training dashboard (egui)
│   │       ├── inspect.rs    # Single sample inspector
│   │       ├── evaluate.rs   # Model evaluation
│   │       ├── spice.rs      # SPICE netlist generation
│   │       ├── spice_mini.rs # Minimal SPICE harness
│   │       ├── neuron_test.rs# Neuron testing
│   │       └── web.rs        # Web server
│   │
│   ├── config.rs             # JSON configuration system (605 LOC)
│   ├── network/              # Network architecture
│   │   ├── network.rs        # Network struct, forward/backward (42 KB)
│   │   ├── cache.rs          # Computation caching
│   │   ├── gradients.rs      # Gradient computation
│   │   ├── state.rs          # Network state management
│   │   └── trace.rs          # Simulation trace recording
│   │
│   ├── neurons/              # LIF neuron implementations (~1560 LOC)
│   │   └── lif/
│   │       ├── leaky.rs      # Main neuron interface
│   │       ├── forward.rs    # Forward pass (550 LOC)
│   │       ├── backward.rs   # Backward pass
│   │       ├── mode.rs       # Physics/Simple mode selection
│   │       └── state.rs      # Neuron state struct
│   │
│   ├── layers/linear.rs      # Dense layer with forward/backward
│   ├── training.rs           # Trainer, optimizer, loss functions
│   ├── data.rs               # MNIST loading and input encoding
│   ├── checkpoint.rs         # Model serialization
│   ├── surrogate.rs          # Surrogate gradient functions
│   ├── spice.rs              # SPICE netlist generation (1207 LOC)
│   ├── hardware.rs           # Hardware configuration
│   ├── visualization.rs      # Rerun.io visualization
│   ├── dashboard.rs          # egui training dashboard
│   ├── animation.rs          # nannou network animation
│   ├── inspector.rs          # egui visual test runner
│   └── web/                  # Web server (Axum)
│
├── frontend/                 # React TypeScript web UI
│   ├── src/
│   │   ├── App.tsx           # Main app component
│   │   └── components/       # React components
│   ├── package.json
│   └── vite.config.ts
│
├── configs/                  # JSON configuration files (25 files)
│   ├── simple.json           # Simple mode config
│   ├── physics.json          # Physics mode config
│   └── temporal_*.json       # Temporal encoding variants
│
├── models/                   # Trained model checkpoints
│   ├── model.json            # Default trained model
│   └── physics_h*.json       # Models with varying hidden sizes
│
├── data/                     # MNIST dataset (symlinked)
├── data_6x6/                 # Downsampled 6x6 MNIST
├── test_images_6x6/          # Test samples (PNG + JSON)
│
├── tools/                    # Analysis utilities
│   ├── synapse_search.py     # Binary search for minimum synapses
│   └── neuron_compare.py     # Neuron comparison script
│
├── scripts/                  # Data generation scripts
├── spice_output*/            # SPICE simulation outputs
└── docs/                     # Documentation
```

## Core Architecture

### Neuron Model (LIF - Leaky Integrate-and-Fire)

**Simple Mode (snnTorch-compatible):**
```
mem[t+1] = beta * mem[t] + input - spike * threshold
```

**Physics Mode (RC Membrane Dynamics):**
```
decay = exp(-dt / tau_m)
mem[t+1] = input * tau_m + (mem[t] - input * tau_m) * decay
```

Physics mode supports pulse stretching (`tau_pulse`), threshold adaptation (`tau_theta`), and hardware voltage clamping (0-5V).

### Default Network Architecture

```
Input (36) → Linear → LIF (12) → Linear → LIF (10) → Spike Count → Prediction
```

- 36 input features (6x6 downsampled MNIST)
- 12 hidden LIF neurons
- 10 output LIF neurons (one per digit)
- Total parameters: 552 synapses
- Surrogate gradient: Fast sigmoid with slope 25.0

### Input Encoding

**Rate-Coded (default):** All 36 pixels presented simultaneously at every timestep.

**Temporal Encoding:** Rows presented sequentially (6 rows over time), mimicking hardware scanning. Row spacing: 1.5ms default.

## Known Issue: evaluate requires --noise for noise-trained models

Models trained with `noise.enabled: true` MUST be evaluated with `--noise --config <config>`.
Without this flag, evaluate uses a different forward path (`forward_quantized` instead of
`forward_noisy`) and produces ~10% accuracy (random chance). This is because the noisy
forward path applies synapse drive model scaling inline, which differs from the standard path.

```bash
# CORRECT:
gilgamesh evaluate --checkpoint model.json --data-dir ./data --noise --config configs/tarski_pcb.json

# WRONG (gives 10%):
gilgamesh evaluate --checkpoint model.json --data-dir ./data
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `train` | Train a new SNN with configurable parameters |
| `evaluate` | Evaluate a trained model on test set (**use --noise for noise-trained models**) |
| `dashboard` | Interactive egui training dashboard |
| `animate` | Real-time network visualizer (nannou) |
| `inspect` | Single sample visual inspector |
| `spice` | Generate ngspice-compatible netlists |
| `spice_mini` | Minimal SPICE harness for 1-2 neurons |
| `neuron_test` | Single neuron test with hardware params |
| `web` | Launch web-based UI with WebSocket |

## Feature Flags

```toml
[features]
blas-accelerate   # BLAS with Apple Accelerate (macOS)
blas-openblas     # BLAS with OpenBLAS (Linux)
visualization     # Rerun.io integration
dashboard         # egui training dashboard
animation         # nannou network animation
web               # Web server with React frontend
```

## Configuration System

JSON-configurable parameters in `configs/`:

| Category | Parameters |
|----------|------------|
| Network | input_size, hidden_size, output_size, image_size |
| Neuron | beta, threshold, reset_mechanism, slope |
| Physics | tau_m, tau_pulse, tau_theta, dt |
| Hardware | v_rail, comp_rail_drop, diode_drop, v_min, v_max |
| Training | lr, epochs, batch_size, num_steps, seed |
| Input | encoding_type, row_spacing, pulse_width |
| Quantization | enabled, bits |
| Noise | enabled, weight_std |
| Output | mode (spike_count or analog) |

## Training System

- **Optimizer**: Adam with AdamW-style weight decay (0.01)
- **Learning Rate**: Cosine annealing from 0.001 to 0.00001
- **Gradient Clipping**: max_norm = 1.0
- **Loss Function**: Cross-entropy on spike counts
- **Surrogate Gradient**: Fast sigmoid with slope = 25

## Performance Benchmarks

| Architecture | Image Size | Parameters | Test Accuracy |
|--------------|------------|------------|---------------|
| 36-6-10 | 6x6 | 276 | 85.05% |
| 36-12-10 | 6x6 | 552 | 91.38% |
| 49-9-10 | 7x7 | 531 | 90.14% |
| 36-12-10 (physics) | 6x6 | 552 | **96.43%** |

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/network/network.rs` | Core network implementation (largest module) |
| `src/neurons/lif/forward.rs` | Neuron forward pass with physics/simple modes |
| `src/config.rs` | Complete configuration system |
| `src/spice.rs` | SPICE netlist generation for hardware validation |
| `src/training.rs` | Training loop, Adam optimizer, loss functions |
| `../arduino-mnist/training/comparison/snntorch_comparison.py` | PyTorch baseline for hardware-oriented comparison |
| `../arduino-mnist/training/comparison/EMBEDDED_GUIDE.md` | Quantization and embedded deployment guide |

## Dependencies

**Rust:** ndarray, rayon, serde, clap, mnist, rerun, eframe/egui, nannou, axum/tokio

**Frontend:** React 18, Vite, TypeScript, Tailwind CSS, Recharts

**Python Comparison:** moved under `arduino-mnist/training/comparison` (torch, snntorch, matplotlib)

## Development Notes

- Focus on minimal parameter networks for chip deployment
- Extensive SPICE validation for hardware accuracy
- Supports 8-bit and 16-bit quantization for embedded systems
- Arduino-focused comparison/training code now lives in `arduino-mnist/training/comparison`
- Temporal encoding has known overfitting issues (needs recurrent connections)
