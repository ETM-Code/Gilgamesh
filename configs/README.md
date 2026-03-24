# Gilgamesh Configuration Files

## Hardware Deployment

### `tarski_pcb.json` — USE THIS for the Tarski PCB

This is the hardware-matched configuration for the Tarski InputSystem PCB.
All values come from the final KiCad schematic (2026-03-24).

**Train:**
```bash
gilgamesh train --config configs/tarski_pcb.json --data-dir ./data --save-checkpoint models/tarski_pcb_v1.json
```

**Evaluate (must use --noise since training uses noise):**
```bash
gilgamesh evaluate --checkpoint models/tarski_pcb_v1.json --data-dir ./data --noise --config configs/tarski_pcb.json
```

**What it configures:**
- **Architecture:** 36→9→10 (6×6 MNIST, 9 hidden neurons, 10 output classes)
- **Physics:** τ_m=1.2ms (C_mem=10nF, R_leak=120kΩ), τ_pulse=1.5µs, τ_θ=0.596ms
- **Quantization:** 2-bit magnitude (0-3) with split-sign — matches hardware's 3 exc + 3 inh mirror enables per synapse (max ±3 units, no weight clamping)
- **Noise:** 5% weight, 2% threshold, 1% membrane, 10% input — models hardware manufacturing variation
- **Input:** 12-bit DAC quantization matching MCP4728 resolution

**Why 2-bit, not 3-bit:**
Each synapse on the PCB has 3 excitatory mirror enables (1+1+2 units) and 3 inhibitory (1+1+2 units). The maximum weight per polarity is 4 units. With 2-bit quantization, magnitude range is 0-3, which fits within the hardware limit. 3-bit (magnitude 0-7) would produce weights that exceed what the hardware can represent.

**Accuracy:** ~83% on 6×6 MNIST (comparable to 3-bit at ~84%).

## Other Configs

- `physics_6x6.json` — 36-12-10 with earlier component values (τ_m=3.96ms). **Outdated for hardware deployment.**
- `simple.json` — Simple mode (snnTorch-compatible), not hardware-accurate.
- `temporal_*.json` — Temporal encoding variants.
