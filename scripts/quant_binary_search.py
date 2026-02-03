#!/usr/bin/env python3
"""Binary search for minimum synapses with 3-bit quantization."""

import json
import subprocess
import tempfile
from pathlib import Path

def eval_quantized(checkpoint_path):
    """Evaluate quantized accuracy for a checkpoint."""
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    quant = checkpoint["quantized"]
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

    Path(temp_path).unlink()

    for line in result.stdout.split('\n'):
        if "Test Accuracy:" in line:
            return float(line.split(':')[1].strip().replace('%', ''))
    return None


def train_and_eval(input_size, hidden_size, epochs=35):
    """Train and return both original and quantized accuracy."""
    synapses = input_size * hidden_size + hidden_size * 10
    checkpoint_path = f"models/q3_{input_size}_{hidden_size}.json"

    # Determine image dimensions (prefer square, fallback to rectangular)
    import math
    sqrt = int(math.sqrt(input_size))
    if sqrt * sqrt == input_size:
        img_args = ["--image-size", str(sqrt)]
    else:
        # Non-square: find factors closest to square
        for h in range(int(math.sqrt(input_size)), 0, -1):
            if input_size % h == 0:
                w = input_size // h
                # For now, skip non-square (would need CLI support)
                print(f"  Skipping non-square {w}x{h}")
                return None, None, synapses

    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "gilgamesh", "--",
         "train"] + img_args + ["--hidden-size", str(hidden_size),
         "--epochs", str(epochs), "--quantize-bits", "3",
         "--save-checkpoint", checkpoint_path],
        capture_output=True, text=True
    )

    original_acc = None
    for line in result.stdout.split('\n'):
        if "Final Test Accuracy:" in line:
            original_acc = float(line.split(':')[1].strip().replace('%', ''))
            break

    if original_acc is None:
        print(f"  Training failed")
        return None, None, synapses

    quant_acc = eval_quantized(checkpoint_path)
    return original_acc, quant_acc, synapses


def binary_search_hidden(input_size, target_acc, low=4, high=30):
    """Binary search for minimum hidden size to achieve target quantized accuracy."""
    best_hidden = None
    best_quant = 0
    results = {}

    while low <= high:
        mid = (low + high) // 2
        synapses = input_size * mid + mid * 10

        print(f"  Testing hidden={mid} ({synapses} synapses)...")
        orig, quant, _ = train_and_eval(input_size, mid)

        if quant is None:
            low = mid + 1
            continue

        results[mid] = (orig, quant, synapses)
        print(f"    -> orig={orig:.2f}%, quant={quant:.2f}%")

        if quant >= target_acc:
            best_hidden = mid
            best_quant = quant
            high = mid - 1  # Try smaller
        else:
            low = mid + 1  # Need bigger

    return best_hidden, best_quant, results


def main():
    all_results = []

    # Test different input sizes
    input_sizes = [
        36,   # 6x6
        49,   # 7x7
        64,   # 8x8
        25,   # 5x5
    ]

    for target in [80, 85]:
        print(f"\n{'='*60}")
        print(f"Finding minimum for {target}% quantized accuracy")
        print(f"{'='*60}")

        for input_size in input_sizes:
            print(f"\nInput size {input_size}:")
            best_h, best_q, results = binary_search_hidden(input_size, target)

            if best_h:
                synapses = input_size * best_h + best_h * 10
                all_results.append({
                    'target': target,
                    'input': input_size,
                    'hidden': best_h,
                    'synapses': synapses,
                    'quantized': best_q
                })
                print(f"  -> Minimum: {input_size}-{best_h}-10 = {synapses} synapses ({best_q:.2f}%)")
            else:
                print(f"  -> Could not find solution")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for target in [80, 85]:
        matches = [r for r in all_results if r['target'] == target]
        if matches:
            best = min(matches, key=lambda x: x['synapses'])
            print(f"\nMinimum for {target}% quantized:")
            print(f"  {best['input']}-{best['hidden']}-10 = {best['synapses']} synapses ({best['quantized']:.2f}%)")

            print(f"\nAll solutions for {target}%:")
            for r in sorted(matches, key=lambda x: x['synapses']):
                print(f"  {r['input']}-{r['hidden']}-10 = {r['synapses']} synapses ({r['quantized']:.2f}%)")


if __name__ == "__main__":
    main()
