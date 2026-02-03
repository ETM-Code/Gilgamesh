#!/usr/bin/env python3
"""Find minimum synapses for 80% and 85% with 3-bit quantization."""

import json
import subprocess
import tempfile
from pathlib import Path

TARGETS = [80, 85]
EPOCHS = 35

def eval_quantized(checkpoint_path):
    """Return quantized accuracy for a checkpoint."""
    with open(checkpoint_path) as f:
        cp = json.load(f)
    q = cp["quantized"]

    fc1 = [[w * q["fc1_pos_scale"] if w >= 0 else w * q["fc1_neg_scale"] for w in row]
           for row in q["fc1_weight"]]
    fc2 = [[w * q["fc2_pos_scale"] if w >= 0 else w * q["fc2_neg_scale"] for w in row]
           for row in q["fc2_weight"]]

    cp2 = cp.copy()
    cp2["weights"] = {"fc1_weight": fc1, "fc2_weight": fc2}
    del cp2["quantized"]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cp2, f)
        tmp = f.name

    r = subprocess.run(
        ["cargo", "run", "--release", "--bin", "gilgamesh", "--", "evaluate", "--checkpoint", tmp],
        capture_output=True, text=True)
    Path(tmp).unlink()

    for line in r.stdout.split('\n'):
        if "Test Accuracy:" in line:
            return float(line.split(':')[1].strip().replace('%', ''))
    return 0


def train(width, height, hidden):
    """Train and return (original_acc, quantized_acc)."""
    inp = width * height
    syn = inp * hidden + hidden * 10
    path = f"models/q3_{width}x{height}_{hidden}.json"

    cmd = ["cargo", "run", "--release", "--bin", "gilgamesh", "--", "train",
           "--image-width", str(width), "--image-height", str(height),
           "--hidden-size", str(hidden), "--epochs", str(EPOCHS),
           "--quantize-bits", "3", "--save-checkpoint", path]

    r = subprocess.run(cmd, capture_output=True, text=True)

    orig = None
    for line in r.stdout.split('\n'):
        if "Final Test Accuracy:" in line:
            orig = float(line.split(':')[1].strip().replace('%', ''))

    if orig is None:
        return None, None

    quant = eval_quantized(path)
    return orig, quant


def find_min_hidden(width, height, target):
    """Binary search for minimum hidden neurons to hit target quantized accuracy."""
    inp = width * height
    lo, hi = 4, 25
    best = None
    cache = {}

    while lo <= hi:
        mid = (lo + hi) // 2
        syn = inp * mid + mid * 10

        if mid in cache:
            orig, quant = cache[mid]
        else:
            print(f"    {width}x{height}-{mid}-10 ({syn} syn)...", end=" ", flush=True)
            orig, quant = train(width, height, mid)
            if quant is None:
                print("failed")
                lo = mid + 1
                continue
            cache[mid] = (orig, quant)
            print(f"orig={orig:.1f}% quant={quant:.1f}%")

        if quant >= target:
            best = (mid, quant, syn)
            hi = mid - 1
        else:
            lo = mid + 1

    return best, cache


def main():
    results = {80: [], 85: []}

    # Image configurations: (width, height, input_size)
    # Square and non-square options
    configs = [
        (5, 5, 25),   # 5x5
        (6, 5, 30),   # 6x5
        (5, 7, 35),   # 5x7
        (6, 6, 36),   # 6x6
        (7, 5, 35),   # 7x5 (same inputs as 5x7, different aspect)
        (6, 7, 42),   # 6x7
        (7, 6, 42),   # 7x6
        (7, 7, 49),   # 7x7
        (8, 6, 48),   # 8x6
        (8, 8, 64),   # 8x8
    ]

    for w, h, inp in configs:
        print(f"\n{'='*50}")
        print(f"Image {w}x{h} ({inp} inputs)")
        print(f"{'='*50}")

        for target in TARGETS:
            print(f"\n  Target: {target}%")
            best, cache = find_min_hidden(w, h, target)

            if best:
                hidden, q, syn = best
                results[target].append((w, h, inp, hidden, syn, q))
                print(f"  -> Minimum: {w}x{h}-{hidden}-10 = {syn} synapses ({q:.1f}%)")
            else:
                print(f"  -> Not achieved")

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    for target in TARGETS:
        print(f"\n{target}% quantized accuracy:")
        if results[target]:
            for w, h, inp, hidden, syn, q in sorted(results[target], key=lambda x: x[4]):
                print(f"  {w}x{h} ({inp}-{hidden}-10): {syn} synapses -> {q:.1f}%")

            best = min(results[target], key=lambda x: x[4])
            w, h, inp, hidden, syn, q = best
            print(f"\n  ** MINIMUM: {w}x{h} ({inp}-{hidden}-10) = {syn} synapses **")
        else:
            print("  No solution found")


if __name__ == "__main__":
    main()
