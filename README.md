# gilgamesh
![gilgamesh logo](logo.png)
A Rust implementation of hardware-accurate spiking neural networks (SNNs). Features dual-mode operation: a simple beta-decay model (snnTorch-compatible) for fast prototyping and a physics-accurate RC membrane model for chip deployment.

## Features

- **Dual-mode neurons**: Simple (snnTorch-compatible) or Physics-accurate RC dynamics
- **Leaky Integrate-and-Fire (LIF)** neurons with surrogate gradient learning
- **Temporal input encoding**: Row-by-row presentation for hardware-like input
- **Analog output mode**: Membrane voltage signals alongside discrete spikes
- **Physics constraints**: Voltage clamping to hardware rails (0-5V)
- **Training features**: Adam optimizer, cosine LR schedule, gradient clipping, weight decay
- **Configurable via JSON**: All parameters controllable through config files
- **Interactive visualizer**: Real-time network activity with sample navigation (nannou)
- **Training visualization**: Loss curves, accuracy, weight heatmaps in Rerun.io
- **Checkpoint system**: Save/load trained networks for inference and visualization
- **Synapse search**: Binary search tool to find minimum synapses for target accuracies

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
gilgamesh train --epochs 15 --data-dir ./data

# Training with config file
gilgamesh train --config configs/test1_physics.json --data-dir ./data

# Train and save checkpoint
gilgamesh train --epochs 10 --save-checkpoint model.json

# Run unit tests
gilgamesh test
```

### Interactive Network Visualizer

Watch your trained network process MNIST samples in real-time:

```bash
# Build with animation support
cargo build --release --features animation

# Launch visualizer (auto-finds most recent checkpoint)
./target/release/gilgamesh animate

# Or specify a checkpoint and starting sample
./target/release/gilgamesh animate --checkpoint model.json --sample 0
```

**Controls:**
- `←` / `→` : Previous / Next sample
- `Space` : Pause / Resume
- `R` : Restart current sample

The visualizer shows:
- MNIST image on the left with true label and prediction
- Network activity with neurons colored by membrane potential
- Spike propagation through synapses
- Output probability bars with spike counts

### Training Visualization with Rerun

Monitor training metrics in real-time using Rerun.io:

```bash
# Build with visualization support
cargo build --release --features visualization

# Train with live Rerun viewer
./target/release/gilgamesh train --visualize --save-checkpoint model.json

# Or save to .rrd file for later viewing
./target/release/gilgamesh train --visualize --visualize-file training.rrd
rerun training.rrd
```

Logged data includes:
- `training/loss`, `training/train_accuracy`, `training/test_accuracy`
- `training/learning_rate` schedule
- `weights/fc1`, `weights/fc2` as heatmaps (every 5 epochs)

**Tip:** In Rerun, select the "epoch" timeline at the bottom and expand the entity tree on the left to see all metrics.

### With Dashboard Features

```bash
# Build with dashboard support
cargo build --release --features dashboard

# Visual inspector (auto-detects most recent checkpoint)
./target/release/gilgamesh inspect

# Or specify a checkpoint
./target/release/gilgamesh inspect --checkpoint model.json
```

### Web UI

Interactive browser-based interface for exploring network activity:

```bash
# Build with web support
cargo build --release --features web

# Launch web server (auto-opens browser)
./target/release/gilgamesh web

# Or specify options
./target/release/gilgamesh web --port 8080 --checkpoint model.json
```

The web UI connects to the Rust backend via WebSocket for real-time simulation.

Requires: `--features web`

### SPICE Hardware Validation

```bash
# Generate netlist for a specific sample
gilgamesh spice --sample 42

# Run ngspice automatically and compare results
gilgamesh spice --run-ngspice
```

## Network Architecture

Default architecture for MNIST classification:

```
Input (36) → Linear → LIF (12) → Linear → LIF (10) → Spike Count → Prediction
     │              └─ hidden layer ─┘           └─ output layer ─┘
     └─ 6x6 downsampled MNIST images
```

- **Input**: 36 features (6x6 pixels)
- **Hidden**: 12 LIF neurons with surrogate gradient
- **Output**: 10 LIF neurons (one per digit class)
- **Parameters**: 552 total (36×12 + 12×10)

### Architecture Comparison

| Architecture | Image | Synapses | Test Accuracy |
|--------------|-------|----------|---------------|
| 24-6-10 | 3x8 | 204 | 80.06% |
| 25-9-10 | 5x5 | 315 | 85.33% |
| 36-6-10 | 6x6 | 276 | 85.05% |
| 36-9-10 | 6x6 | 414 | 89.34% |
| **36-12-10** | **6x6** | **552** | **91.38%** |
| 49-9-10 | 7x7 | 531 | 90.14% |

### Minimum Synapses by Target Accuracy

Results from automated synapse search across non-square image dimensions:

| Target | Architecture | Image | Synapses | Achieved |
|--------|-------------|-------|----------|----------|
| 85% | 25-9-10 | 5x5 | 315 | 85.33% |
| 80% | 24-6-10 | 3x8 | 204 | 80.06% |

Non-square images (e.g. 3x8) can outperform square ones by capturing more vertical structure from MNIST digits with fewer total pixels.

### Quantization-Aware Training (QAT) for Hardware

Weights are quantized during training using split-sign scaling: 3-bit magnitude (0-7) + 1-bit sign, with independent current scales for excitatory and inhibitory sources per layer. This matches the hardware model where separate current DACs drive positive and negative synapses.

Input pixels are quantized to 12-bit DAC resolution during training.

**QAT results without noise** (rate-coded, 6x6, physics mode, 3-bit split-sign weights, 12-bit input):

| Architecture | Synapses | Best Accuracy | Notes |
|--------------|----------|---------------|-------|
| 36-5-10 | 230 | 75.6% avg | High seed variance (±3%) |
| 36-6-10 | 276 | 77.2% avg | High seed variance |
| 36-7-10 | 322 | 83.2% avg | ±2.2% across seeds |
| **36-8-10** | **368** | **84.6% avg** | **±1.0%, most efficient for ~85%** |
| 36-9-10 | 414 | 85.7% avg | Stable across seeds |
| 36-12-10 | 552 | 87.0% | 3-bit ceiling |

**With noise injection** (weight=5%, threshold=2%, membrane=0.01, input=10%, enabled by default):

Noise improves robustness to hardware variation. 7x7 input captures more spatial structure and raises the 3-bit accuracy ceiling from ~87% to ~90%.

| Architecture | Image | Synapses | Accuracy | Notes |
|--------------|-------|----------|----------|-------|
| 49-5-10 | 7x7 | 295 | 83.66% | |
| 49-6-10 | 7x7 | 354 | 84.75% | |
| 49-7-10 | 7x7 | 413 | 84.78% | |
| **49-8-10** | **7x7** | **472** | **87.47%** | |
| 49-9-10 | 7x7 | 531 | 89.83% | Near 3-bit ceiling |
| 36-7-10 | 6x6 | 322 | 84.66% | |
| 36-8-10 | 6x6 | 368 | 84.07% | |
| 36-9-10 | 6x6 | 414 | 87.39% | |

**4-bit QAT results** (rate-coded, physics mode, 4-bit split-sign weights, 12-bit input, noise enabled):

| Architecture | Image | Synapses | Accuracy |
|--------------|-------|----------|----------|
| 64-5-10 | 8x8 | 370 | 75.63% |
| 64-6-10 | 8x8 | 444 | 85.81% |
| 64-7-10 | 8x8 | 518 | 86.36% |
| 64-8-10 | 8x8 | 592 | 88.39% |
| 64-9-10 | 8x8 | 666 | 89.24% |
| 64-10-10 | 8x8 | 740 | 89.72% |
| 81-5-10 | 9x9 | 455 | 84.72% |
| 81-6-10 | 9x9 | 546 | 87.61% |
| 81-7-10 | 9x9 | 637 | 89.36% |
| 81-8-10 | 9x9 | 728 | 89.47% |
| **81-9-10** | **9x9** | **819** | **91.32%** |
| 100-5-10 | 10x10 | 550 | 78.03% |
| 100-6-10 | 10x10 | 660 | 87.65% |
| 100-7-10 | 10x10 | 770 | 88.55% |
| 100-8-10 | 10x10 | 880 | 90.45% |
| **100-9-10** | **10x10** | **990** | **92.03%** |

**Key findings:**
- **4-bit weights raise the accuracy ceiling** to ~92% (vs ~90% for 3-bit at 7x7)
- With 3-bit weights, accuracy saturates around **87% for 6x6** and **90% for 7x7** — weight precision is the bottleneck, not network capacity
- **9x9 is the most efficient** for 90%+ accuracy: 81-9-10 hits 91.32% at 819 synapses vs 100-9-10 at 92.03% with 990 synapses
- Larger images help but with diminishing returns — 9x9 and 10x10 are within ~1% at same hidden sizes, while 8x8 lags ~2%
- Noise is enabled by default for all training

Saved model: `models/rate_quant_3bit_6x6_h8.json`

**Post-training quantization** (for comparison — quantizing after full-precision training):

| Network | Bits | Original | Quantized | Accuracy Drop |
|---------|------|----------|-----------|---------------|
| 36-6-10 | 3 | 85.68% | 78.32% | 7.4% |
| 36-12-10 | 3 | 91.80% | 82.82% | 9.0% |
| **36-12-10** | **4** | **92.05%** | **91.11%** | **0.9%** |

QAT with 3-bit weights significantly closes the gap vs post-training quantization (84.6% QAT vs 78.3% post-training at similar network size).

Checkpoint format includes both f32 weights and quantized integer weights:
```json
{
  "quantized": {
    "magnitude_bits": 4,
    "max_magnitude": 15,
    "fc1_pos_scale": 0.0197,
    "fc1_neg_scale": 0.0315,
    "fc1_weight": [[0, 2, -10, 1, ...], ...],
    "fc2_pos_scale": 0.0436,
    "fc2_neg_scale": 0.0627,
    "fc2_weight": [[-4, 15, 1, 7, -11, ...], ...]
  }
}
```

To reconstruct analog weights: `weight = magnitude × (pos_scale if positive else neg_scale)`

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
All 36 pixels presented simultaneously at every timestep. The pixel intensity determines spike probability or input current.

### Spiking Input Encoding
Converts pixel values to spike trains via deterministic accumulator, making all connections uniform spiking synapses (matching hardware where every connection is a spike-triggered current source).

Features:
- **Burst spikes**: `floor(accumulator)` allows multi-level encoding (like Loihi 2 graded spikes)
- **Dithered initialization**: Random accumulator offsets in [0,1) break quantization patterns
- **Residual injection**: Final timestep adds remaining accumulator value as fractional spike
- **Normalization fix**: Undoes MNIST normalization to get [0,1] pixel values for the accumulator

Results with noise injection (weight=5%, threshold=2%, membrane=0.01, input=10%):

| Architecture | Image | Synapses | Accuracy |
|--------------|-------|----------|----------|
| 36-12-10 | 6x6 | 552 | 86.69% |
| 40-7-10 | 5x8 | 350 | 82.95% |
| 36-8-10 | 4x9 | 368 | 82.67% |
| 25-9-10 | 5x5 | 315 | 81.89% |

### Temporal Encoding
Rows presented sequentially, mimicking hardware scanning:
- 6 rows presented over time with configurable spacing
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
    "input_size": 36,
    "hidden_size": 12,
    "output_size": 10,
    "image_size": 6
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

All tests on MNIST (60k train, 10k test, 6x6 downsampled), 15 epochs:

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

## PyTorch/snntorch Comparison

A Python comparison script is included to benchmark gilgamesh against standard PyTorch implementations.

### Setup

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv --system-site-packages comparison/venv

# Install dependencies (if not using system packages)
source comparison/venv/bin/activate
pip install torch torchvision snntorch matplotlib numpy
```

### Running

```bash
# Run all models (SNNs + ANNs)
./comparison/run_comparison.sh

# Or run specific models
comparison/venv/bin/python comparison/snntorch_comparison.py --models baseline ann

# Custom training
comparison/venv/bin/python comparison/snntorch_comparison.py --epochs 20 --lr 0.0005
```

### Models Compared

| Model | Type | Architecture |
|-------|------|--------------|
| `GilgameshSNN` | SNN | 36 → LIF(12) → LIF(10) — matches gilgamesh exactly |
| `GilgameshSNN_Synaptic` | SNN | Dual-exponential synaptic dynamics |
| `GilgameshSNN_Recurrent` | SNN | Recurrent connections in hidden layer |
| `GilgameshSNN_3Layer` | SNN | 36 → LIF(12) → LIF(6) → LIF(10) |
| `StandardANN` | ANN | 36 → ReLU(12) → 10 — non-spiking baseline |
| `StandardANN_3Layer` | ANN | 36 → ReLU(12) → ReLU(6) → 10 |

### Output

Results saved to `comparison/results/`:
- `comparison_report.md` — Accuracy comparison table
- `training_curves.png` — Training/test accuracy plots
- `*_weights.pt` — PyTorch state dicts
- `*_weights.json` — JSON weights (gilgamesh-compatible format)

## Training Details

- **Optimizer**: Adam with AdamW-style weight decay (0.01)
- **Learning rate**: Cosine annealing from 0.001 to 0.00001
- **Gradient clipping**: max_norm = 1.0
- **Surrogate gradient**: Fast sigmoid with slope = 25
- **Loss function**: Cross-entropy on spike counts

## CLI Options

### Training
```
gilgamesh train [OPTIONS]

Options:
  --config <FILE>           JSON config file (overrides other args)
  --lr <RATE>               Learning rate [default: 0.001]
  --epochs <N>              Number of epochs [default: 15]
  --batch-size <N>          Batch size [default: 128]
  --num-steps <N>           Timesteps per sample [default: 25]
  --hidden-size <N>         Hidden layer neurons [default: 12]
  --beta <VALUE>            Membrane decay factor [default: 0.9]
  --seed <N>                Random seed [default: 42]
  --data-dir <PATH>         MNIST data directory [default: ./data]
  --slope <VALUE>           Surrogate gradient slope [default: 25.0]
  --quantize                Enable 8-bit weight quantization
  --noise                   Enable noise injection
  --save-checkpoint <PATH>  Save trained model to checkpoint file
  --visualize               Enable Rerun visualization (requires --features visualization)
  --visualize-file <PATH>   Save visualization to .rrd file instead of spawning viewer
```

### Interactive Network Visualizer

Real-time visualization of network activity with sample navigation.

```
gilgamesh animate [OPTIONS]

Options:
  --checkpoint <PATH>   Path to checkpoint file (auto-detects most recent if omitted)
  --data-dir <PATH>     MNIST data directory [default: ./data]
  --speed <MULT>        Animation speed multiplier [default: 1.0]
  --sample <N>          Starting sample index [default: 0]
```

Features:
- MNIST image display with label/prediction
- Neuron membrane potentials visualized as color intensity
- Spike propagation pulses along synapses
- Output probability bars with spike counts
- Arrow key navigation between test samples

Requires: `--features animation`

### Visual Inspector

Interactively inspect individual test samples with a visual egui interface.

```
gilgamesh inspect [OPTIONS]

Options:
  --checkpoint <PATH>   Path to checkpoint file (auto-detects most recent if omitted)
  --data-dir <PATH>     MNIST data directory [default: ./data]
  --num-steps <N>       Timesteps for inference [default: 25]
  --seed <N>            Random seed for sample selection [default: 42]
```

The inspector shows:
- 6x6 input image (scaled up for visibility)
- Bar chart of output neuron spike counts
- Color-coded predictions: green=correct, red=wrong, blue=true label
- Click "Next Sample" to cycle through random test samples

Requires: `--features dashboard`

### Web UI

Browser-based interactive network explorer with real-time WebSocket connection.

```
gilgamesh web [OPTIONS]

Options:
  -p, --port <PORT>     Server port [default: 3000]
  -c, --checkpoint <PATH>   Path to checkpoint file (auto-detects most recent if omitted)
  --data-dir <PATH>     MNIST data directory [default: ./data]
  --open <BOOL>         Open browser automatically [default: true]
```

Requires: `--features web`

### SPICE Comparison

Generate ngspice-compatible netlists and compare simulation results for hardware validation.

```
gilgamesh spice [OPTIONS]

Options:
  --checkpoint <PATH>   Path to checkpoint file (auto-detects most recent if omitted)
  --data-dir <PATH>     MNIST data directory [default: ./data]
  --sample <N>          Test sample index (random if omitted)
  --output-dir <PATH>   Directory for netlists and results [default: ./spice_output]
  --run-ngspice         Run ngspice automatically (requires ngspice in PATH)
  --num-steps <N>       Timesteps for simulation [default: 25]
```

Features:
- LIF neuron subcircuit with RC membrane, comparator, reset switch, and pulse shaping
- Full 36→12→10 network with VCCS (voltage-controlled current sources) for weights
- Compares gilgamesh spike counts against ngspice simulation
- Outputs comparison table with predictions from both simulators

Performance benchmark (full network, sample 42, physics_6x6_fixed checkpoint):
- **gilgamesh is 12005x faster than ngspice** for the timed simulation kernel in this setup.
- Measured with `network_spice_bench` using `--duration-ms 1.0`, where:
  - gilgamesh in-process average: **0.027 ms**
  - ngspice average: **327.149 ms**
- Startup-inclusive CLI comparison in the same run: ngspice was **3.5x** slower than `gilgamesh spice` CLI average.

```bash
cargo build --release --bin network_spice_bench --bin gilgamesh
./target/release/network_spice_bench \
  --checkpoint ./models/physics_6x6_fixed.json \
  --data-dir ./data \
  --sample 42 \
  --num-steps 25 \
  --duration-ms 1.0 \
  --gilgamesh-iters 200 \
  --warmup 10 \
  --gilgamesh-cli-runs 5 \
  --spice-runs 1 \
  --output-dir ./benchmark_spice_latest
```

Latency/accuracy tradeoff (Physics 6x6, `dt=1ms`):
- Baseline model (`models/physics_6x6_fixed.json`) was trained at 25 steps.
- Since each step is 1ms of simulated time, `25 steps = 25ms`, `10 steps = 10ms`.

Baseline checkpoint accuracy vs steps:

| Steps | Simulated time | Test accuracy |
|-------|----------------|---------------|
| 5 | 5ms | 66.84% |
| 10 | 10ms | 82.75% |
| 15 | 15ms | 85.20% |
| 20 | 20ms | 86.58% |
| 25 | 25ms | 87.20% |

Retraining for 10-step inference:
- Trained a dedicated 10-step model (`configs/physics_6x6_10steps.json`).
- Output checkpoint: `models/physics_6x6_10steps_retrained.json`.
- Results:
  - retrained model @10 steps: **85.69%**
  - retrained model @25 steps: **86.40%**
  - baseline model @25 steps: **87.20%**

So retraining for 10ms recovered accuracy substantially compared to truncating the 25-step-trained model at 10 steps (82.75% → 85.69%).

Model topology used in these tests:
- Input: 36 features (6x6)
- Hidden: 12 LIF neurons (physics mode)
- Output: 10 LIF neurons (physics mode)
- Total spiking neurons: 22
- Synapses: 552 (`36×12 + 12×10`)

### Training Speed Benchmark (Physics Mode vs snnTorch vs PyTorch)

Measured on this machine:
- macOS arm64 (Darwin 25.2.0)
- Apple M3
- 16GB RAM
- CPU-only benchmark (`--device cpu` for Python runs)

Matched benchmark spec:
- Dataset: MNIST downsampled to 6x6
- Architecture: `36 -> 12 -> 10`
- Timesteps: `25`
- Epochs: `15`
- Batch size: `128`
- Seed: `42`

Commands used:

```bash
# gilgamesh (physics mode)
/usr/bin/time -p ./target/release/gilgamesh train \
  --config ./configs/physics_6x6.json \
  --data-dir ./data

# snnTorch baseline SNN (GilgameshSNN model in comparison script)
/usr/bin/time -p ./comparison/.venv/bin/python ./comparison/snntorch_comparison.py \
  --data-dir ./comparison/data \
  --output-dir ./artifacts/training_bench_20260211/snntorch_baseline \
  --models baseline \
  --epochs 15 \
  --batch-size 128 \
  --num-steps 25 \
  --hidden-size 12 \
  --seed 42 \
  --device cpu

# PyTorch ANN baseline (StandardANN model in comparison script)
/usr/bin/time -p ./comparison/.venv/bin/python ./comparison/snntorch_comparison.py \
  --data-dir ./comparison/data \
  --output-dir ./artifacts/training_bench_20260211/pytorch_ann \
  --models ann \
  --epochs 15 \
  --batch-size 128 \
  --num-steps 25 \
  --hidden-size 12 \
  --seed 42 \
  --device cpu
```

Results (wall-clock `real` time):

| Stack | Time | Best test accuracy | Relative to gilgamesh |
|-------|------|--------------------|-----------------------|
| gilgamesh (physics mode) | **4.87s** | 86.99% | 1.00x |
| snnTorch SNN (`baseline`) | 76.74s | 90.56% | **15.76x slower** |
| PyTorch ANN (`ann`) | 40.83s | 88.51% | **8.38x slower** |

For this workload, gilgamesh is absolutely goated on training speed.

Why the gap is large on this benchmark:
- gilgamesh uses a fixed, compiled Rust training path with custom forward/backward and optimizer updates.
- No dynamic autograd graph construction overhead.
- Tiny model + many timesteps amplifies per-op framework overhead in Python stacks.
- gilgamesh batches preloaded contiguous arrays directly.

Raw logs for this run:
- `artifacts/training_bench_20260211/gilgamesh_physics.log`
- `artifacts/training_bench_20260211/snntorch_baseline.log`
- `artifacts/training_bench_20260211/pytorch_ann.log`

### Synapse Search

Multi-phase intelligent search for minimum synapses to achieve target accuracies. Explores non-square image dimensions (w×h) since orientation affects how much spatial info is captured from MNIST.

```bash
# Full search (all targets: 95%, 90%, 85%, 80%, 70%)
python3 tools/synapse_search.py

# Quick mode (3-epoch probes, 8-epoch full training)
python3 tools/synapse_search.py --quick

# Custom targets and feature range
python3 tools/synapse_search.py --targets 85 80 --min-features 9 --max-features 49

# Specify output file
python3 tools/synapse_search.py --output results.json
```

Options:
- `--targets`: Target accuracies to search for (default: 95 90 85 80 70)
- `--probe-epochs`: Epochs for Phase 1 landscape probing (default: 5)
- `--full-epochs`: Epochs for Phase 2/3 full training (default: 15)
- `--min-features` / `--max-features`: Input feature range (default: 9-120)
- `--quick`: Quick mode (3-epoch probes, 8-epoch full)
- `--output`: Output JSON file for results

The search has three phases:
1. **Landscape scan**: Probe each (w,h) dimension at small and large hidden sizes
2. **Binary search**: For promising dimensions, binary search on hidden size
3. **Refinement**: Multi-seed testing and nearby dimension exploration

**Synapse formula (weights only, no biases):**
```
Total = hidden × (input + 10)
      = hidden × (width × height + 10)
```

Example: 24-6-10 architecture (3x8 image) = 6 × (24 + 10) = **204 synapses**

### Checkpoint Auto-Detection

Both `inspect` and `spice` commands automatically find and use the most recently modified checkpoint file when `--checkpoint` is not specified. It searches:
- Current directory (`*.json`)
- `./checkpoints/*.json`
- `./models/*.json`

Files are validated to ensure they contain checkpoint data (architecture and weights).

## Project Structure

```
src/
├── main.rs          # CLI entry point
├── lib.rs           # Library exports
├── config.rs        # JSON configuration
├── network.rs       # Network architecture, forward/backward
├── training.rs      # Trainer, optimizer, loss functions
├── data.rs          # MNIST loading, input encoding
├── checkpoint.rs    # Model serialization/deserialization
├── inspector.rs     # Visual test runner (egui)
├── spice.rs         # SPICE netlist generation and comparison
├── neurons/
│   └── leaky.rs     # LIF neuron (simple + physics modes)
├── layers/
│   └── linear.rs    # Dense layer with forward/backward
├── surrogate.rs     # Surrogate gradient functions
├── tensor.rs        # Loss functions, utilities
├── dashboard.rs     # Training dashboard (egui)
├── animation.rs     # Network animation (nannou)
└── visualization.rs # Rerun.io visualization

tools/
└── synapse_search.py  # Binary search for minimum synapses

comparison/
├── snntorch_comparison.py  # PyTorch/snntorch training script
├── requirements.txt        # Python dependencies
├── run_comparison.sh       # Run script (uses venv)
└── results/                # Output directory (generated)
```

## License

MIT
