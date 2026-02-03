#!/usr/bin/env python3
"""Comprehensive search for minimum synapses with 3-bit quantization."""

import json
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def eval_quantized(checkpoint_path):
    """Evaluate quantized accuracy for a checkpoint."""
    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
    except FileNotFoundError:
        return None

    quant = checkpoint.get("quantized")
    if not quant:
        return None

    fc1_pos_scale = quant["fc1_pos_scale"]
    fc1_neg_scale = quant["fc1_neg_scale"]
    fc2_pos_scale = quant["fc2_pos_scale"]
    fc2_neg_scale = quant["fc2_neg_scale"]

    fc1_quant = [[w * fc1_pos_scale if w >= 0 else w * fc1_neg_scale for w in row]
                 for row in quant["fc1_weight"]]
    fc2_quant = [[w * fc2_pos_scale if w >= 0 else w * fc2_neg_scale for w in row]
                 for row in quant["fc2_weight"]]

    quant_checkpoint = checkpoint.copy()
    quant_checkpoint["weights"] = {"fc1_weight": fc1_quant, "fc2_weight": fc2_quant}
    del quant_checkpoint["quantized"]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(quant_checkpoint, f)
        temp_path = f.name

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
    return quant_acc


def train_model(image_size, hidden_size, seed=42, epochs=35):
    """Train a model and return results."""
    input_size = image_size * image_size
    synapses = input_size * hidden_size + hidden_size * 10
    checkpoint_path = f"models/search_{input_size}_{hidden_size}_10_s{seed}.json"

    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "gilgamesh", "--",
         "train", "--image-size", str(image_size), "--hidden-size", str(hidden_size),
         "--epochs", str(epochs), "--seed", str(seed), "--quantize-bits", "3",
         "--save-checkpoint", checkpoint_path],
        capture_output=True, text=True
    )

    original_acc = None
    for line in result.stdout.split('\n'):
        if "Final Test Accuracy:" in line:
            original_acc = float(line.split(':')[1].strip().replace('%', ''))
            break

    if original_acc is None:
        return None

    quant_acc = eval_quantized(checkpoint_path)

    return {
        'image': image_size,
        'hidden': hidden_size,
        'seed': seed,
        'input': input_size,
        'synapses': synapses,
        'original': original_acc,
        'quantized': quant_acc,
        'drop': original_acc - quant_acc if quant_acc else None
    }


def main():
    # Test configurations: (image_size, hidden_size, seed)
    configs = []

    # 6x6 (36 inputs)
    for h in [6, 7, 8, 9, 10, 12, 14, 16]:
        for seed in [42, 123]:
            configs.append((6, h, seed))

    # 7x7 (49 inputs)
    for h in [6, 7, 8, 9, 10, 12]:
        for seed in [42, 123]:
            configs.append((7, h, seed))

    # 8x8 (64 inputs)
    for h in [6, 8, 10, 12]:
        for seed in [42]:
            configs.append((8, h, seed))

    results = []
    total = len(configs)

    for i, (img, h, seed) in enumerate(configs):
        print(f"\n[{i+1}/{total}] Training {img*img}-{h}-10 (seed={seed})...")
        r = train_model(img, h, seed)
        if r:
            results.append(r)
            print(f"  Original: {r['original']:.2f}% | Quantized: {r['quantized']:.2f}% | "
                  f"Synapses: {r['synapses']}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - 3-bit Quantization Results (sorted by synapses)")
    print("="*80)
    print(f"{'Arch':<12} {'Syn':<8} {'Seed':<6} {'Orig':<8} {'Quant':<8} {'Drop':<8}")
    print("-"*80)

    for r in sorted(results, key=lambda x: (x['synapses'], -x['quantized'])):
        arch = f"{r['input']}-{r['hidden']}-10"
        print(f"{arch:<12} {r['synapses']:<8} {r['seed']:<6} {r['original']:<8.2f} "
              f"{r['quantized']:<8.2f} {r['drop']:<8.2f}")

    # Best by synapse count
    print("\n" + "-"*80)
    print("Best quantized accuracy per synapse count:")
    synapse_counts = sorted(set(r['synapses'] for r in results))
    for syn in synapse_counts:
        matches = [r for r in results if r['synapses'] == syn]
        best = max(matches, key=lambda x: x['quantized'])
        arch = f"{best['input']}-{best['hidden']}-10"
        print(f"  {syn} synapses: {arch} -> {best['quantized']:.2f}% (seed {best['seed']})")

    # Minimum for targets
    print("\n" + "-"*80)
    for target in [80, 85]:
        matching = [r for r in results if r['quantized'] >= target]
        if matching:
            best = min(matching, key=lambda x: x['synapses'])
            arch = f"{best['input']}-{best['hidden']}-10"
            print(f"Minimum for {target}% quantized: {arch} with {best['synapses']} synapses "
                  f"({best['quantized']:.2f}%, seed {best['seed']})")
        else:
            print(f"No architecture achieved {target}% quantized accuracy")


if __name__ == "__main__":
    main()
