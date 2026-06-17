# Tarski PCB MNIST network (36→9→10, 3-bit quantized)

Trained networks for the **Tarski analog neuromorphic PCB** — a 6×6→36→9→10 analog
spiking MNIST classifier. Architecture and quantization match the board exactly
(`configs/tarski_pcb_v7.json`): 36 inputs (6×6), 9 hidden, 10 outputs, 3-bit-magnitude
split-sign synapse weights (−7…+7), `broken_inhibitory_lsb` modelled.

## Files

| File | What it is | Accuracy |
|---|---|---|
| `tarski_mnist_q3_v7_base.json` | Base SNN, surrogate-gradient trained on 6×6 MNIST, 3-bit quantized fc2 (hardware-ready). | **84.21%** best test (software/idealized forward) |
| `tarski_mnist_q3_v7_hwaware.json` | The base, then **hardware-in-the-loop fine-tuned** against gilgamesh's *faithful as-built board* `hw_forward` (real output θ ≈ 0.543 V, **O2/O10 masked**, matched mirrors). Coordinate-ascent on fc1 to maximize accuracy *as the real board reads it*. | **56.3%** on the faithful HW forward (was 55.1%) |
| `train_base.log`, `finetune_hwaware.log` | Full run logs. | |

## Why the two numbers differ (this is expected, not a bug)

`*_base.json` scores ~84% under the **idealized** forward. `*_hwaware.json` is evaluated
through the **faithful** board model, which is honest about the analog reality:

- **O2/O10 are physically masked** on the as-built board → two digit classes are unreadable
  → ~80% hard ceiling before anything else.
- The weak ~0.43 µA synapse current + 120 kΩ leak plateau the output membranes at ~0.5 V,
  compressing all ten columns into a ~0.05 V band → near-tie digits tip the wrong way.

So ~56% is the *faithful predicted board accuracy* on this sample, consistent with the
in-emulation decode (`tarski-works/`). See `tarski-works/OUTPUT_THRESHOLD_AND_CURRENT_MATH.md`
and `LESSONS_AND_HAUKSBEE_IMPROVEMENTS.md` for the full analysis, and
`BOARD_CHANGES.md` for the hardware levers (un-mask O2/O10, stronger synapse current) that
would raise that ceiling.

## Reproduce

```bash
# base
gilgamesh train   --dataset mnist --config configs/tarski_pcb_v7.json --data-dir data \
                  --save-checkpoint models/tarski_pcb_mnist_36-9-10/tarski_mnist_q3_v7_base.json
# hardware-aware fine-tune (faithful board: output θ 0.543 V, O2/O10 masked — the config default)
gilgamesh finetune --checkpoint models/tarski_pcb_mnist_36-9-10/tarski_mnist_q3_v7_base.json \
                   --config configs/tarski_pcb_v7.json --data-dir data \
                   --output models/tarski_pcb_mnist_36-9-10/tarski_mnist_q3_v7_hwaware.json \
                   --epochs 3 --train-samples 800 --test-samples 1500
```

Note: `--data-dir data` points at the **28×28** MNIST idx files; gilgamesh downsamples to 6×6
internally. (Pre-downsampled `data_6x6/*` are rejected by the loader, which expects 28-row idx.)

The faithful-board behaviour (output threshold, O2/O10 mask, optional mirror-mismatch) lives in
`HwForwardConfig::default()` in `src/hw_forward.rs`.
