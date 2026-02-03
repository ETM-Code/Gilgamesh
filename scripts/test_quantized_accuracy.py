#!/usr/bin/env python3
"""Test accuracy using quantized integer weights vs original f32 weights."""

import json
import numpy as np
from pathlib import Path

def load_checkpoint(path):
    with open(path) as f:
        return json.load(f)

def reconstruct_weights(quantized):
    """Reconstruct f32 weights from quantized integers."""
    scale = quantized["weight_per_bit"]
    fc1 = np.array(quantized["fc1_weight"], dtype=np.float32) * scale
    fc2 = np.array(quantized["fc2_weight"], dtype=np.float32) * scale
    return fc1, fc2

def compare_weights(checkpoint):
    """Compare original f32 weights with reconstructed quantized weights."""
    # Original f32 weights
    fc1_orig = np.array(checkpoint["weights"]["fc1_weight"], dtype=np.float32)
    fc2_orig = np.array(checkpoint["weights"]["fc2_weight"], dtype=np.float32)

    # Reconstructed from quantized
    fc1_quant, fc2_quant = reconstruct_weights(checkpoint["quantized"])

    # Calculate error
    fc1_error = np.abs(fc1_orig - fc1_quant)
    fc2_error = np.abs(fc2_orig - fc2_quant)

    print(f"FC1 shape: {fc1_orig.shape}")
    print(f"FC2 shape: {fc2_orig.shape}")
    print()
    print(f"Quantization info:")
    print(f"  bits: {checkpoint['quantized']['bits']}")
    print(f"  weight_per_bit: {checkpoint['quantized']['weight_per_bit']:.6f}")
    print(f"  max_int: {checkpoint['quantized']['max_int']}")
    print()
    print(f"FC1 weight stats:")
    print(f"  original range: [{fc1_orig.min():.4f}, {fc1_orig.max():.4f}]")
    print(f"  quantized range: [{fc1_quant.min():.4f}, {fc1_quant.max():.4f}]")
    print(f"  mean abs error: {fc1_error.mean():.6f}")
    print(f"  max abs error: {fc1_error.max():.6f}")
    print()
    print(f"FC2 weight stats:")
    print(f"  original range: [{fc2_orig.min():.4f}, {fc2_orig.max():.4f}]")
    print(f"  quantized range: [{fc2_quant.min():.4f}, {fc2_quant.max():.4f}]")
    print(f"  mean abs error: {fc2_error.mean():.6f}")
    print(f"  max abs error: {fc2_error.max():.6f}")

    # Show distribution of quantized values
    fc1_int = np.array(checkpoint["quantized"]["fc1_weight"])
    fc2_int = np.array(checkpoint["quantized"]["fc2_weight"])

    print()
    print("Quantized integer distribution:")
    for name, arr in [("FC1", fc1_int), ("FC2", fc2_int)]:
        unique, counts = np.unique(arr, return_counts=True)
        print(f"  {name}: ", end="")
        for v, c in zip(unique, counts):
            print(f"{v:+d}:{c}", end=" ")
        print()

def main():
    checkpoint_path = Path("models/model_36_6_10_q4.json")

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = load_checkpoint(checkpoint_path)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Architecture: {checkpoint['architecture']['input_size']}-"
          f"{checkpoint['architecture']['hidden_size']}-"
          f"{checkpoint['architecture']['output_size']}")
    print()

    compare_weights(checkpoint)

if __name__ == "__main__":
    main()
