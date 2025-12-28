# gilgamesh

A Rust implementation of hardware-accurate spiking neural networks (SNNs). Features dual-mode operation: a simple beta-decay model (snnTorch-compatible) for fast prototyping and a physics-accurate RC membrane model for chip deployment.

## Features

- **Dual-mode neurons**: Simple (snnTorch-compatible) or Physics-accurate RC dynamics
- **Leaky Integrate-and-Fire (LIF)** neurons with surrogate gradient learning
- **Temporal input encoding**: Row-by-row presentation for hardware-like input
- **Analog output mode**: Membrane voltage signals alongside discrete spikes
- **Physics constraints**: Voltage clamping to hardware rails (0-5V)
- **Training features**: Adam optimizer, cosine LR schedule, gradient clipping, weight decay
- **Configurable via JSON**: All parameters controllable through config files

## Building

### Standard build (portable)
```bash
cargo build --release
```

### With BLAS acceleration (recommended, 2-5x faster)

**macOS** (Apple Accelerate):
```bash
cargo build --release --features blas-accelerate
```

**Linux** (OpenBLAS):
```bash
# Install: sudo apt install libopenblas-dev
cargo build --release --features blas-openblas
```

## Quick Start

```bash
# Simple training with CLI args
./target/release/gilgamesh train --epochs 15 --data-dir ./data

# Training with config file
./target/release/gilgamesh train --config configs/test1_physics.json --data-dir ./data

# Run unit tests
./target/release/gilgamesh test
```

## Network Architecture

Default architecture for MNIST classification:

```
Input (49) → Linear → LIF (100) → Linear → LIF (10) → Spike Count → Prediction
     │              └─ hidden layer ─┘           └─ output layer ─┘
     └─ 7x7 downsampled MNIST images
```

- **Input**: 49 features (7x7 pixels)
- **Hidden**: 100 LIF neurons with surrogate gradient
- **Output**: 10 LIF neurons (one per digit class)
- **Parameters**: 6,010 total (49×100 + 100 + 100×10 + 10)

## Modes of Operation

### Simple Mode (snnTorch-compatible)

Uses discrete beta-decay membrane dynamics:
```
mem[t+1] = beta * mem[t] + input - spike * threshold
```

Where beta = 0.9 provides exponential decay. This matches snnTorch's Leaky neuron.

### Physics Mode

Uses continuous RC membrane dynamics derived from circuit physics:
```
decay = exp(-dt / tau_m)
mem[t+1] = input * tau_m + (mem[t] - input * tau_m) * decay
```

Parameters:
- `tau_m`: Membrane time constant (~9.5ms)
- `dt`: Integration timestep (1ms default)
- `v_min/v_max`: Hardware voltage rails (0-5V)

The equivalence: `beta = exp(-dt / tau_m)`, so `tau_m = -dt / ln(beta)`

## Input Encoding Modes

### Rate-Coded (default)
All 49 pixels presented simultaneously at every timestep. The pixel intensity determines spike probability or input current.

### Temporal Encoding
Rows presented sequentially, mimicking hardware scanning:
- 7 rows presented over time with configurable spacing
- `row_spacing`: Time between rows (default: 1.5ms)
- `pulse_width`: Fraction of row spacing for pulse (default: 90%)

## Analog Output Mode

When `analog_gain > 0`, membrane voltage is added to the spike signal:
```
output = spikes + analog_gain * membrane_voltage
```

This provides richer inter-layer communication at the cost of potentially reduced sparsity.

## Configuration

All parameters are configurable via JSON:

```json
{
  "mode": "physics",
  "network": {
    "input_size": 49,
    "hidden_size": 100,
    "output_size": 10
  },
  "neuron": {
    "beta": 0.9,
    "threshold": 1.0,
    "slope": 25.0
  },
  "physics": {
    "enabled": true,
    "tau_m": 0.00949,
    "tau_pulse": 0.00167,
    "dt": 0.001
  },
  "input_encoding": {
    "encoding_type": "temporal",
    "row_spacing": 0.0015,
    "pulse_width": 0.9
  },
  "output": {
    "analog_gain": 0.1
  },
  "training": {
    "lr": 0.001,
    "epochs": 15,
    "batch_size": 128,
    "num_steps": 25,
    "seed": 42
  }
}
```

## Benchmark Results

All tests on MNIST (60k train, 10k test, 7x7 downsampled), 15 epochs:

| Test | Mode | Input Encoding | Analog | Best Test Acc | Notes |
|------|------|----------------|--------|---------------|-------|
| 1 | Physics | Rate-coded | No | **96.43%** | Best overall |
| 2 | Physics | Rate-coded | 0.1 | **96.03%** | Minimal impact |
| 3 | Physics | Temporal | No | 38.72% | Severe overfitting |
| 4 | Physics | Temporal | 0.1 | 35.32% | Analog hurts here |

### Key Findings

1. **Physics mode with rate-coded input achieves 96%+ accuracy**, matching snnTorch performance
2. **Analog output has minimal impact** on rate-coded input (+/- 0.4%)
3. **Temporal encoding causes severe overfitting**: 89% train vs 37% test accuracy
4. **Analog output slightly hurts temporal encoding** (35% vs 39% test)

### Temporal Encoding Analysis

The large train-test gap with temporal encoding suggests:
- The network memorizes training patterns rather than learning generalizable features
- Row-by-row presentation requires different architecture (e.g., recurrent connections, more hidden neurons)
- May need regularization tuned for temporal dynamics

## Training Details

- **Optimizer**: Adam with AdamW-style weight decay (0.01)
- **Learning rate**: Cosine annealing from 0.001 to 0.00001
- **Gradient clipping**: max_norm = 1.0
- **Surrogate gradient**: Fast sigmoid with slope = 25
- **Loss function**: Cross-entropy on spike counts

## CLI Options

```
gilgamesh train [OPTIONS]

Options:
  --config <FILE>       JSON config file (overrides other args)
  --lr <RATE>           Learning rate [default: 0.001]
  --epochs <N>          Number of epochs [default: 15]
  --batch-size <N>      Batch size [default: 128]
  --num-steps <N>       Timesteps per sample [default: 25]
  --hidden-size <N>     Hidden layer neurons [default: 100]
  --beta <VALUE>        Membrane decay factor [default: 0.9]
  --seed <N>            Random seed [default: 42]
  --data-dir <PATH>     MNIST data directory [default: ./data]
  --slope <VALUE>       Surrogate gradient slope [default: 25.0]
  --quantize            Enable 8-bit weight quantization
  --noise               Enable noise injection
```

## Project Structure

```
src/
├── main.rs          # CLI entry point
├── lib.rs           # Library exports
├── config.rs        # JSON configuration
├── network.rs       # Network architecture, forward/backward
├── training.rs      # Trainer, optimizer, loss functions
├── data.rs          # MNIST loading, input encoding
├── neurons/
│   └── leaky.rs     # LIF neuron (simple + physics modes)
├── layers/
│   └── linear.rs    # Dense layer with forward/backward
├── surrogate.rs     # Surrogate gradient functions
└── tensor.rs        # Loss functions, utilities
```

## License

MIT
