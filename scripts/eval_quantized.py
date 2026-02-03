#!/usr/bin/env python3
"""Evaluate quantized weights accuracy with separate pos/neg scales.

This script reconstructs f32 weights from quantized integer weights
using the per-layer, per-sign scales and evaluates with gilgamesh.
"""

import json
import subprocess
import tempfile
from pathlib import Path

def main():
    checkpoint_path = "models/model_36_12_10_q4.json"

    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    print(f"Original checkpoint: {checkpoint_path}")
    arch = checkpoint["architecture"]
    print(f"Architecture: {arch['input_size']}-{arch['hidden_size']}-{arch['output_size']}")
    print()

    # Evaluate original f32 weights
    print("Evaluating original f32 weights...")
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "gilgamesh", "--",
         "evaluate", "--checkpoint", checkpoint_path],
        capture_output=True, text=True
    )
    print(result.stdout)

    # Check if we have the new quantized format with separate pos/neg scales
    quant = checkpoint.get("quantized")
    if not quant:
        print("No quantized weights found in checkpoint")
        return

    # Check for new format (has fc1_pos_scale) vs old format (has fc1_weight_per_bit)
    if "fc1_pos_scale" in quant:
        print("Using new format with separate pos/neg scales per layer")
        fc1_pos_scale = quant["fc1_pos_scale"]
        fc1_neg_scale = quant["fc1_neg_scale"]
        fc2_pos_scale = quant["fc2_pos_scale"]
        fc2_neg_scale = quant["fc2_neg_scale"]

        # Reconstruct fc1: positive uses pos_scale, negative uses neg_scale
        fc1_quant = []
        for row in quant["fc1_weight"]:
            fc1_row = []
            for w in row:
                w = int(w)
                if w >= 0:
                    fc1_row.append(w * fc1_pos_scale)
                else:
                    fc1_row.append(w * fc1_neg_scale)  # w is negative, so this gives correct sign
            fc1_quant.append(fc1_row)

        # Reconstruct fc2
        fc2_quant = []
        for row in quant["fc2_weight"]:
            fc2_row = []
            for w in row:
                w = int(w)
                if w >= 0:
                    fc2_row.append(w * fc2_pos_scale)
                else:
                    fc2_row.append(w * fc2_neg_scale)
            fc2_quant.append(fc2_row)

        print(f"  FC1 scales: pos={fc1_pos_scale:.6f}, neg={fc1_neg_scale:.6f}")
        print(f"  FC2 scales: pos={fc2_pos_scale:.6f}, neg={fc2_neg_scale:.6f}")

    elif "fc1_weight_per_bit" in quant:
        print("Using old format with per-layer scales")
        fc1_scale = quant["fc1_weight_per_bit"]
        fc2_scale = quant["fc2_weight_per_bit"]

        fc1_quant = [[int(w) * fc1_scale for w in row]
                     for row in quant["fc1_weight"]]
        fc2_quant = [[int(w) * fc2_scale for w in row]
                     for row in quant["fc2_weight"]]

        print(f"  FC1 scale: {fc1_scale:.6f}")
        print(f"  FC2 scale: {fc2_scale:.6f}")
    else:
        print("Unknown quantization format")
        return

    # Create new checkpoint with reconstructed weights
    quant_checkpoint = checkpoint.copy()
    quant_checkpoint["weights"] = {
        "fc1_weight": fc1_quant,
        "fc2_weight": fc2_quant,
    }
    del quant_checkpoint["quantized"]  # Remove to avoid confusion

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(quant_checkpoint, f, indent=2)
        temp_path = f.name

    bits = quant.get("magnitude_bits", quant.get("bits", 4))
    print(f"\nEvaluating reconstructed {bits}-bit weights...")
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "gilgamesh", "--",
         "evaluate", "--checkpoint", temp_path],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Clean up
    Path(temp_path).unlink()

if __name__ == "__main__":
    main()
