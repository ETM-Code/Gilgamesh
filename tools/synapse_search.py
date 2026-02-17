#!/usr/bin/env python3
"""
Intelligent search for minimum synapses to achieve target accuracies.

Uses a multi-phase approach:
  Phase 1: Coarse landscape scan with short training to map accuracy vs architecture
  Phase 2: Focus on promising regions, binary search on synapse count
  Phase 3: Refine with full training and multiple seeds near the boundary

Supports non-square image dimensions (w×h) for more granular input sizes.

Synapse formula (weights only, no biases):
  Total = hidden × (input + 10)
  where input = image_width × image_height
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


TARGET_ACCURACIES = [95.0, 90.0, 85.0, 80.0, 70.0]

MAX_HIDDEN = 30


@dataclass
class TrainingResult:
    """Result from a single training run."""
    image_width: int
    image_height: int
    hidden_size: int
    output_size: int = 10
    best_accuracy: float = 0.0
    final_accuracy: float = 0.0
    total_synapses: int = 0
    success: bool = False
    error: Optional[str] = None
    epochs: int = 15
    seed: int = 42
    input_size: int = field(init=False)

    def __post_init__(self):
        self.input_size = self.image_width * self.image_height
        self.total_synapses = self.hidden_size * (self.input_size + self.output_size)

    @property
    def architecture(self) -> str:
        return f"{self.input_size}-{self.hidden_size}-{self.output_size}"

    @property
    def dims(self) -> str:
        if self.image_width == self.image_height:
            return f"{self.image_width}x{self.image_height}"
        return f"{self.image_width}x{self.image_height}"


def generate_input_dimensions(min_features=9, max_features=120):
    """Generate (width, height) pairs covering a range of input feature counts.

    Includes both orientations for non-square (e.g. 3x12 AND 12x3) since
    width vs height emphasis captures different spatial info from MNIST.
    """
    dims = set()
    for w in range(3, 15):
        for h in range(3, 15):
            features = w * h
            if min_features <= features <= max_features:
                dims.add((w, h))
    # Sort by feature count, then by squareness (prefer square)
    return sorted(dims, key=lambda d: (d[0] * d[1], abs(d[0] - d[1])))


def create_config(width, height, hidden_size, epochs=15, seed=42,
                   spiking=False, noise=False, tau_m=None, num_steps=None):
    """Create a physics-mode training config with given dimensions."""
    input_size = width * height

    # Spiking mode uses slower tau_m and more timesteps by default
    if tau_m is None:
        tau_m = 0.02 if spiking else 0.00396
    if num_steps is None:
        num_steps = 50 if spiking else 25

    config = {
        "mode": "physics",
        "network": {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": 10,
            "image_size": width,  # fallback for square
        },
        "neuron": {
            "beta": 0.9,
            "threshold": 1.0,
            "slope": 25.0,
            "reset_mechanism": "subtract",
        },
        "physics": {
            "enabled": True,
            "tau_m": tau_m,
            "dt": 0.001,
            "adaptation_enabled": False,
        },
        "training": {
            "lr": 0.001,
            "epochs": epochs,
            "batch_size": 128,
            "num_steps": num_steps,
            "seed": seed,
        },
        "input_encoding": {"encoding_type": "spiking" if spiking else "rate_coded"},
        "quantization": {"enabled": False},
        "noise": {
            "enabled": noise,
            "training_only": False,
            "weight_std": 0.05,
            "threshold_std": 0.02,
            "membrane_std": 0.01,
            "input_std": 0.1,
        } if noise else {"enabled": False},
        "output": {"mode": "spike_count", "analog_gain": 0.0},
    }
    if width != height:
        config["network"]["image_width"] = width
        config["network"]["image_height"] = height
    return config


def run_training(config, data_dir="./data", gilgamesh_bin="./target/release/gilgamesh"):
    """Run a single training experiment and return the result."""
    net = config["network"]
    w = net.get("image_width", net.get("image_size", 6))
    h = net.get("image_height", net.get("image_size", w))
    hidden = net["hidden_size"]
    epochs = config["training"]["epochs"]
    seed = config["training"]["seed"]

    result = TrainingResult(image_width=w, image_height=h, hidden_size=hidden,
                            epochs=epochs, seed=seed)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        proc = subprocess.run(
            [gilgamesh_bin, "train", "--config", config_path, "--data-dir", data_dir],
            capture_output=True, text=True, timeout=600,
        )
        output = proc.stdout + proc.stderr

        if proc.returncode != 0:
            result.error = f"Exit code {proc.returncode}"
        else:
            best = re.search(r"Best Test Accuracy:\s*([\d.]+)%", output)
            final = re.search(r"Final Test Accuracy:\s*([\d.]+)%", output)
            if best:
                result.best_accuracy = float(best.group(1))
                result.success = True
            if final:
                result.final_accuracy = float(final.group(1))

    except subprocess.TimeoutExpired:
        result.error = "Timeout"
    except Exception as e:
        result.error = str(e)
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass

    return result


class SynapseSearch:
    """Multi-phase intelligent search for minimum synapses."""

    def __init__(self, data_dir="./data", gilgamesh_bin="./target/release/gilgamesh",
                 probe_epochs=5, full_epochs=15, verbose=True,
                 spiking=False, noise=False):
        self.data_dir = data_dir
        self.gilgamesh_bin = gilgamesh_bin
        self.probe_epochs = probe_epochs
        self.full_epochs = full_epochs
        self.verbose = verbose
        self.spiking = spiking
        self.noise = noise
        # Cache: (w, h, hidden, epochs, seed) -> TrainingResult
        self.cache = {}
        self.all_results = []

    def _run(self, w, h, hidden, epochs=None, seed=42):
        """Run training with caching."""
        if epochs is None:
            epochs = self.full_epochs
        key = (w, h, hidden, epochs, seed)
        if key in self.cache:
            return self.cache[key]

        config = create_config(w, h, hidden, epochs=epochs, seed=seed,
                               spiking=self.spiking, noise=self.noise)
        result = run_training(config, self.data_dir, self.gilgamesh_bin)
        self.cache[key] = result
        self.all_results.append(result)
        return result

    def _log(self, msg, indent=0):
        if self.verbose:
            print("  " * indent + msg, flush=True)

    def phase1_landscape(self, dims_list):
        """Phase 1: Coarse scan to understand what accuracy each input size can achieve.

        For each input dimension, test a small and large hidden size to get
        a rough accuracy range. Uses short training (probe_epochs).
        """
        self._log(f"\n{'='*70}")
        self._log("PHASE 1: Landscape scan (coarse, {}-epoch probes)".format(self.probe_epochs))
        self._log(f"{'='*70}")

        # Map: (w, h) -> (max_accuracy_seen, best_hidden)
        landscape = {}

        for w, h in dims_list:
            features = w * h
            # Test a small hidden (3) and a generous hidden
            test_hidden = min(MAX_HIDDEN, max(8, features // 4))

            r_small = self._run(w, h, 3, epochs=self.probe_epochs)
            r_large = self._run(w, h, test_hidden, epochs=self.probe_epochs)

            best_acc = max(r_small.best_accuracy, r_large.best_accuracy)
            landscape[(w, h)] = {
                "max_acc": best_acc,
                "small_acc": r_small.best_accuracy,
                "large_acc": r_large.best_accuracy,
                "test_hidden": test_hidden,
            }

            self._log(f"  {w}x{h} (input={features:3d}): "
                       f"h=3 -> {r_small.best_accuracy:5.1f}%, "
                       f"h={test_hidden} -> {r_large.best_accuracy:5.1f}%")

        return landscape

    def phase2_find_minimum(self, target_acc, dims_list, landscape):
        """Phase 2: For each input dimension that can plausibly reach the target,
        binary search on hidden size to find the minimum, then pick the
        dimension with fewest total synapses.

        Uses full_epochs for precision.
        """
        self._log(f"\n{'='*70}")
        self._log(f"PHASE 2: Finding minimum synapses for {target_acc}%")
        self._log(f"{'='*70}")

        best_result = None
        best_synapses = float("inf")

        # Sort dims by most promising first: highest accuracy in landscape probe
        ranked_dims = sorted(
            dims_list,
            key=lambda d: landscape.get(d, {}).get("max_acc", 0),
            reverse=True,
        )

        for w, h in ranked_dims:
            features = w * h
            info = landscape.get((w, h), {})
            probe_max = info.get("max_acc", 0)

            # Skip if even generous hidden in probe couldn't approach target
            # Allow some margin since full training will do better
            if probe_max < target_acc - 15:
                self._log(f"  Skipping {w}x{h}: probe max only {probe_max:.1f}%", 1)
                continue

            # Can we beat current best? Minimum synapses for this input = 1 * (features + 10)
            min_possible = features + 10
            if min_possible >= best_synapses:
                self._log(f"  Skipping {w}x{h}: min possible ({min_possible}) >= best ({best_synapses})", 1)
                continue

            # Binary search on hidden size
            self._log(f"\n  Searching {w}x{h} (input={features}):", 1)

            # First confirm max hidden can actually reach target with full training
            max_h = min(MAX_HIDDEN, best_synapses // (features + 10))
            if max_h < 1:
                self._log(f"    Max hidden would be 0, skipping", 1)
                continue

            r_max = self._run(w, h, max_h, epochs=self.full_epochs)
            self._log(f"    h={max_h}: {r_max.best_accuracy:.2f}%", 1)

            if r_max.best_accuracy < target_acc:
                self._log(f"    Can't reach {target_acc}% (max={r_max.best_accuracy:.2f}%)", 1)
                continue

            # Binary search for minimum hidden
            low, high = 1, max_h
            best_for_dim = None

            while low <= high:
                mid = (low + high) // 2
                synapses = mid * (features + 10)

                # Don't bother if this can't beat current best
                if synapses >= best_synapses:
                    high = mid - 1
                    continue

                r = self._run(w, h, mid, epochs=self.full_epochs)
                self._log(f"    h={mid}: {r.best_accuracy:.2f}% ({synapses} syn)", 1)

                if r.best_accuracy >= target_acc:
                    best_for_dim = r
                    high = mid - 1
                else:
                    low = mid + 1

            if best_for_dim and best_for_dim.total_synapses < best_synapses:
                best_synapses = best_for_dim.total_synapses
                best_result = best_for_dim
                self._log(f"  ** New best: {best_for_dim.architecture} ({best_for_dim.dims}) "
                          f"= {best_synapses} synapses ({best_for_dim.best_accuracy:.2f}%)", 1)

        return best_result

    def phase3_refine(self, target_acc, result, dims_list):
        """Phase 3: Refine around the boundary with multiple seeds and nearby configs.

        Tries to squeeze out a smaller architecture by:
        - Testing multiple seeds (training is noisy, maybe we got unlucky/lucky)
        - Trying nearby input dimensions at the same hidden size
        - Trying hidden-1 with multiple seeds
        """
        if result is None:
            return None

        self._log(f"\n{'='*70}")
        self._log(f"PHASE 3: Refining around {result.architecture} ({result.dims})")
        self._log(f"{'='*70}")

        best = result
        w, h, hidden = result.image_width, result.image_height, result.hidden_size
        features = w * h

        # Try hidden - 1 with multiple seeds (maybe we can get away with less)
        if hidden > 1:
            self._log(f"\n  Testing h={hidden - 1} with multiple seeds:")
            for seed in [42, 123, 7, 2024, 999]:
                r = self._run(w, h, hidden - 1, epochs=self.full_epochs, seed=seed)
                self._log(f"    seed={seed}: {r.best_accuracy:.2f}% ({r.total_synapses} syn)")
                if r.best_accuracy >= target_acc and r.total_synapses < best.total_synapses:
                    best = r
                    self._log(f"    ** Improved! {r.architecture}")

        # Try nearby input dimensions at current hidden and hidden-1
        for test_h in [best.hidden_size, best.hidden_size - 1]:
            if test_h < 1:
                continue
            self._log(f"\n  Nearby dims at h={test_h}:")
            target_features = best.input_size
            for dw, dh in dims_list:
                df = dw * dh
                # Only look at nearby input sizes
                if abs(df - target_features) > 10:
                    continue
                syn = test_h * (df + 10)
                if syn >= best.total_synapses:
                    continue
                r = self._run(dw, dh, test_h, epochs=self.full_epochs)
                self._log(f"    {dw}x{dh} (input={df}): {r.best_accuracy:.2f}% ({syn} syn)")
                if r.best_accuracy >= target_acc and r.total_synapses < best.total_synapses:
                    best = r
                    self._log(f"    ** Improved! {r.architecture} ({r.dims})")

        # If we improved, try one more step down
        if best is not result and best.hidden_size > 1:
            self._log(f"\n  Got improvement, trying h={best.hidden_size - 1} multi-seed:")
            for seed in [42, 123, 7]:
                r = self._run(best.image_width, best.image_height,
                              best.hidden_size - 1, epochs=self.full_epochs, seed=seed)
                self._log(f"    seed={seed}: {r.best_accuracy:.2f}% ({r.total_synapses} syn)")
                if r.best_accuracy >= target_acc and r.total_synapses < best.total_synapses:
                    best = r

        return best

    def run(self, targets=None, min_features=9, max_features=120):
        """Run the full multi-phase search."""
        if targets is None:
            targets = TARGET_ACCURACIES

        dims = generate_input_dimensions(min_features=min_features, max_features=max_features)
        mode_desc = "spiking" if self.spiking else "rate-coded"
        if self.noise:
            mode_desc += " + noise"
        self._log(f"Mode: {mode_desc}")
        self._log(f"Generated {len(dims)} input dimensions to explore")
        self._log(f"Range: {dims[0][0]}x{dims[0][1]} ({dims[0][0]*dims[0][1]} features) "
                  f"to {dims[-1][0]}x{dims[-1][1]} ({dims[-1][0]*dims[-1][1]} features)")

        # Phase 1: landscape
        landscape = self.phase1_landscape(dims)

        # Process targets from hardest to easiest
        results = {}
        sorted_targets = sorted(targets, reverse=True)

        for target in sorted_targets:
            # Phase 2: find minimum
            best = self.phase2_find_minimum(target, dims, landscape)

            # Phase 3: refine
            best = self.phase3_refine(target, best, dims)

            results[target] = best

        # Summary
        self._log(f"\n{'='*70}")
        self._log("FINAL RESULTS: Minimum Synapses for Target Accuracies")
        self._log(f"{'='*70}")
        self._log(f"{'Target':>8} | {'Arch':>14} | {'Dims':>7} | {'Synapses':>8} | {'Accuracy':>8}")
        self._log(f"{'-'*8}-+-{'-'*14}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

        for target in sorted(results.keys(), reverse=True):
            r = results[target]
            if r:
                self._log(f"{target:>7.1f}% | {r.architecture:>14} | {r.dims:>7} | "
                          f"{r.total_synapses:>8} | {r.best_accuracy:>7.2f}%")
            else:
                self._log(f"{target:>7.1f}% | {'NOT FOUND':>14} | {'-':>7} | {'-':>8} | {'-':>8}")

        return results


def save_results(search, results, output_file):
    """Save all results to JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {},
        "pareto_frontier": [],
        "all_runs": [],
    }

    for target, r in results.items():
        if r:
            output["summary"][str(target)] = {
                "architecture": r.architecture,
                "image_dims": f"{r.image_width}x{r.image_height}",
                "hidden_size": r.hidden_size,
                "synapses": r.total_synapses,
                "accuracy": r.best_accuracy,
            }
        else:
            output["summary"][str(target)] = None

    # Build Pareto frontier from all results
    all_sorted = sorted(search.all_results, key=lambda r: r.total_synapses)
    best_acc = 0.0
    for r in all_sorted:
        if r.success and r.best_accuracy > best_acc:
            best_acc = r.best_accuracy
            output["pareto_frontier"].append({
                "synapses": r.total_synapses,
                "accuracy": r.best_accuracy,
                "architecture": r.architecture,
                "dims": r.dims,
            })

    # All runs
    for r in search.all_results:
        output["all_runs"].append({
            "image_width": r.image_width,
            "image_height": r.image_height,
            "hidden_size": r.hidden_size,
            "input_size": r.input_size,
            "architecture": r.architecture,
            "dims": r.dims,
            "synapses": r.total_synapses,
            "best_accuracy": r.best_accuracy,
            "final_accuracy": r.final_accuracy,
            "epochs": r.epochs,
            "seed": r.seed,
            "success": r.success,
        })

    output["all_runs"].sort(key=lambda x: x["synapses"])

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent search for minimum synapses to achieve target accuracies"
    )
    parser.add_argument("--targets", type=float, nargs="+", default=TARGET_ACCURACIES,
                        help="Target accuracies (default: 95 90 85 80 70)")
    parser.add_argument("--probe-epochs", type=int, default=5,
                        help="Epochs for coarse probing phase (default: 5)")
    parser.add_argument("--full-epochs", type=int, default=15,
                        help="Epochs for full training phase (default: 15)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--gilgamesh-bin", type=str, default="./target/release/gilgamesh")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3-epoch probes, 8-epoch full")
    parser.add_argument("--min-features", type=int, default=9,
                        help="Minimum input features (default: 9)")
    parser.add_argument("--max-features", type=int, default=120,
                        help="Maximum input features (default: 120)")
    parser.add_argument("--spiking", action="store_true",
                        help="Use spiking input encoding (slower tau_m, more timesteps)")
    parser.add_argument("--noise", action="store_true",
                        help="Enable noise during training and evaluation")

    args = parser.parse_args()

    if args.quick:
        args.probe_epochs = 3
        args.full_epochs = 8

    if args.output is None:
        suffix = ""
        if args.spiking:
            suffix += "_spiking"
        if args.noise:
            suffix += "_noise"
        args.output = f"synapse_search{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    search = SynapseSearch(
        data_dir=args.data_dir,
        gilgamesh_bin=args.gilgamesh_bin,
        probe_epochs=args.probe_epochs,
        full_epochs=args.full_epochs,
        verbose=not args.quiet,
        spiking=args.spiking,
        noise=args.noise,
    )

    results = search.run(args.targets, min_features=args.min_features,
                         max_features=args.max_features)
    save_results(search, results, args.output)
    print(f"\nResults saved to: {args.output}")
    print(f"Total training runs: {len(search.all_results)}")


if __name__ == "__main__":
    main()
