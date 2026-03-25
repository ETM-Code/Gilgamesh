# Gilgamesh Configuration Files

## Hardware Deployment

### `tarski_pcb.json` — USE THIS for the Tarski PCB

This is the hardware-matched configuration for the Tarski InputSystem PCB.
All values derived from the final KiCad schematic and validated by the emulator.

**Train:**
```bash
gilgamesh train --config configs/tarski_pcb.json --data-dir ./data --save-checkpoint models/my_model.json
```

**Evaluate (MUST use --noise for noise-trained models):**
```bash
gilgamesh evaluate --checkpoint models/my_model.json --data-dir ./data --noise --config configs/tarski_pcb.json
```

> **WARNING:** Running `evaluate` without `--noise` on a noise-trained checkpoint
> gives ~10% accuracy (random chance). This is a known issue — the training forward
> path (`forward_noisy`) differs from the default evaluate path (`forward_quantized`).
> Always use `--noise --config configs/tarski_pcb.json` when evaluating.

### What it configures

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | 36→9→10 | 6×6 MNIST, 9 hidden, 10 output |
| τ_m | 1.2ms | C_mem=10nF × R_leak=120kΩ |
| τ_pulse | 870µs | R_stretch=150kΩ × C_stretch=5.8nF (**requires PCB fix**) |
| τ_θ | 0.596ms | C_thresh=4.7nF / G_theta |
| spike_scale | 0.5 | τ_pulse effective duty cycle within 1ms timestep |
| Quantization | 3-bit magnitude, split-sign | 1+2+4 binary mirror encoding per polarity |
| fixed_fc2_scale | 0.1349 | I_unit × duty × R_leak / (θ_hw × spike_scale) |
| Noise | 5% weight, 2% threshold, 1% membrane, 10% input | Manufacturing variation |
| Input | 12-bit DAC | MCP4728 resolution |
| Current caps | 3µA/synapse, 50µA/neuron | 10MΩ R_set × 7 mirror units |

### Why fixed_fc2_scale matters

Without it, gilgamesh's quantization picks a weight scale that maximizes accuracy
but doesn't match the hardware's current level. Each mirror unit on the PCB delivers
exactly 0.435µA (set by the 10MΩ R_set). The fixed scale ensures every integer weight
step in gilgamesh corresponds to exactly one mirror unit on hardware.

With adaptive scale: gilgamesh and hardware spike counts differ by ~50%.
With fixed scale: **identical spike patterns** on every output neuron.

### PCB fix required

See `URGENT_CIRCUIT_SURGERY.md` at the repo root. The pulse stretcher capacitor
C_stretch must be changed from 10pF to ~5.6nF on the 9 hidden layer neurons.

## Other Configs

| Config | Architecture | Use |
|--------|-------------|-----|
| `tarski_pcb.json` | 36-9-10, physics | **Hardware deployment (use this)** |
| `physics_6x6.json` | 36-12-10, physics | Old component values (τ_m=3.96ms). Outdated. |
| `simple.json` | Various, simple mode | snnTorch-compatible, not hardware-accurate |
| `temporal_*.json` | Various, temporal | Row-by-row encoding variants |
