#!/usr/bin/env python3
"""Search for minimum synapses with 3-bit quantization for target accuracies."""

import json
import subprocess
import tempfile
from pathlib import Path

def train_and_eval(image_size, hidden_size, epochs=35):
    """Train a model and return both original and quantized accuracy."""
    input_size = image_size * image_size
    output_size = 10
    synapses = input_size * hidden_size + hidden_size * output_size

    # Create temp checkpoint path
    checkpoint_path = f"models/search_{input_size}_{hidden_size}_10_q3.json"

    # Determine data directory based on image size
    if image_size == 6:
        data_dir = "./data"  # Uses default 6x6
    else:
        data_dir = "./data"  # Will need to handle other sizes

    print(f"\n{'='*60}")
    print(f"Training {input_size}-{hidden_size}-10 ({synapses} synapses)")
    print(f"{'='*60}")

    # Train
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "gilgamesh", "--",
         "train", "--hidden-size", str(hidden_size), "--epochs", str(epochs),
         "--quantize-bits", "3", "--save-checkpoint", checkpoint_path],
        capture_output=True, text=True
    )

    # Extract original accuracy from training output
    original_acc = None
    for line in result.stdout.split('\n'):
        if "Final Test Accuracy:" in line:
            original_acc = float(line.split(':')[1].strip().replace('%', ''))
            break

    if original_acc is None:
        print(f"Training failed: {result.stderr}")
        return None, None, synapses

    # Now evaluate quantized
    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        return original_acc, None, synapses

    quant = checkpoint.get("quantized")
    if not quant:
        print("No quantized weights in checkpoint")
        return original_acc, None, synapses

    # Reconstruct quantized weights
    fc1_pos_scale = quant["fc1_pos_scale"]
    fc1_neg_scale = quant["fc1_neg_scale"]
    fc2_pos_scale = quant["fc2_pos_scale"]
    fc2_neg_scale = quant["fc2_neg_scale"]

    fc1_quant = []
    for row in quant["fc1_weight"]:
        fc1_row = []
        for w in row:
            w = int(w)
            if w >= 0:
                fc1_row.append(w * fc1_pos_scale)
            else:
                fc1_row.append(w * fc1_neg_scale)
        fc1_quant.append(fc1_row)

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

    # Create quantized checkpoint
    quant_checkpoint = checkpoint.copy()
    quant_checkpoint["weights"] = {
        "fc1_weight": fc1_quant,
        "fc2_weight": fc2_quant,
    }
    del quant_checkpoint["quantized"]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(quant_checkpoint, f, indent=2)
        temp_path = f.name

    # Evaluate quantized
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "gilgamesh", "--",
         "evaluate", "--checkpoint", temp_path],
        capture_output=True, text=True
    )

    quant_acc = None
    for line in result.stdout.split('\n'):
        if "Test Accuracy:" in line:
            quant_acc = float(line.split(':')[1].strip().replace('%', ''))
            break

    Path(temp_path).unlink()

    return original_acc, quant_acc, synapses


def main():
    results = []

    # Test various architectures with 6x6 input (36 features)
    # Start with configurations around what we know works
    configs = [
        # (image_size, hidden_size)
        (6, 6),   # 276 synapses - baseline
        (6, 8),   # 368 synapses
        (6, 10),  # 460 synapses
        (6, 12),  # 552 synapses
        (6, 14),  # 644 synapses
        (6, 16),  # 736 synapses
        (6, 18),  # 828 synapses
        (6, 20),  # 920 synapses
    ]

    for image_size, hidden_size in configs:
        original, quantized, synapses = train_and_eval(image_size, hidden_size)
        if original is not None and quantized is not None:
            results.append({
                'arch': f"{image_size*image_size}-{hidden_size}-10",
                'synapses': synapses,
                'original': original,
                'quantized': quantized,
                'drop': original - quantized
            })
            print(f"\nResult: {results[-1]['arch']} | {synapses} syn | "
                  f"orig: {original:.2f}% | quant: {quantized:.2f}% | drop: {original-quantized:.2f}%")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - 3-bit Quantization Results")
    print("="*70)
    print(f"{'Architecture':<15} {'Synapses':<10} {'Original':<10} {'Quantized':<10} {'Drop':<10}")
    print("-"*70)

    for r in sorted(results, key=lambda x: x['synapses']):
        print(f"{r['arch']:<15} {r['synapses']:<10} {r['original']:<10.2f} {r['quantized']:<10.2f} {r['drop']:<10.2f}")

    # Find minimums for targets
    print("\n" + "-"*70)
    for target in [80, 85]:
        matching = [r for r in results if r['quantized'] >= target]
        if matching:
            best = min(matching, key=lambda x: x['synapses'])
            print(f"Minimum for {target}% quantized: {best['arch']} with {best['synapses']} synapses ({best['quantized']:.2f}%)")
        else:
            print(f"No architecture achieved {target}% quantized accuracy")


if __name__ == "__main__":
    main()
