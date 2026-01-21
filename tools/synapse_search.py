#!/usr/bin/env python3
"""
Binary search for minimum synapses to achieve target accuracies.

This script runs the gilgamesh MNIST trainer in physics mode and uses binary search
to find the minimum number of synapses required to achieve target accuracy levels.

The search varies both:
- Image size (MNIST downsampling): affects input layer size (image_size² features)
- Hidden layer size: the main tunable parameter

Synapse formula (weights only, no biases):
  Total = input_size × hidden_size + hidden_size × 10
        = hidden_size × (input_size + 10)
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


# Target accuracies to search for
TARGET_ACCURACIES = [95.0, 90.0, 85.0, 80.0, 70.0]

# Search bounds
MIN_HIDDEN = 1
MAX_HIDDEN = 30  # 49-9-10 already gets ~90%, so keep this small
MIN_IMAGE_SIZE = 3
MAX_IMAGE_SIZE = 10  # Keep input size reasonable


@dataclass
class TrainingResult:
    """Result from a single training run."""
    image_size: int
    hidden_size: int
    output_size: int = 10
    best_accuracy: float = 0.0
    final_accuracy: float = 0.0
    total_synapses: int = 0
    success: bool = False
    error: Optional[str] = None
    input_size: int = field(init=False)

    def __post_init__(self):
        self.input_size = self.image_size ** 2
        self.total_synapses = self.calculate_synapses()

    def calculate_synapses(self) -> int:
        """Calculate total synapse count (weights only, no biases)."""
        # FC1: input -> hidden
        fc1 = self.input_size * self.hidden_size
        # FC2: hidden -> output
        fc2 = self.hidden_size * self.output_size
        return fc1 + fc2

    @property
    def architecture(self) -> str:
        return f"{self.input_size}-{self.hidden_size}-{self.output_size}"


@dataclass
class SearchResult:
    """Result of binary search for a target accuracy."""
    target_accuracy: float
    best_result: Optional[TrainingResult] = None
    all_attempts: list = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.best_result is not None

    @property
    def min_synapses(self) -> Optional[int]:
        return self.best_result.total_synapses if self.best_result else None


def create_physics_config(
    image_size: int,
    hidden_size: int,
    epochs: int = 15,
    seed: int = 42,
) -> dict:
    """Create a physics-mode training config."""
    input_size = image_size ** 2
    return {
        "mode": "physics",
        "network": {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": 10,
            "image_size": image_size,
        },
        "neuron": {
            "beta": 0.9,
            "threshold": 1.0,
            "slope": 25.0,
            "reset_mechanism": "subtract",
        },
        "physics": {
            "enabled": True,
            "tau_m": 0.00396,
            "dt": 0.001,
            "adaptation_enabled": False,
        },
        "training": {
            "lr": 0.001,
            "epochs": epochs,
            "batch_size": 128,
            "num_steps": 25,
            "seed": seed,
        },
        "input_encoding": {
            "encoding_type": "rate_coded",
        },
        "quantization": {
            "enabled": False,
        },
        "noise": {
            "enabled": False,
        },
        "output": {
            "mode": "spike_count",
            "analog_gain": 0.0,
        },
    }


def run_training(
    config: dict,
    data_dir: str = "./data",
    gilgamesh_bin: str = "./target/release/gilgamesh",
    verbose: bool = False,
) -> TrainingResult:
    """Run a single training experiment and return the result."""
    image_size = config["network"]["image_size"]
    hidden_size = config["network"]["hidden_size"]

    result = TrainingResult(
        image_size=image_size,
        hidden_size=hidden_size,
    )

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        cmd = [gilgamesh_bin, "train", "--config", config_path, "--data-dir", data_dir]

        if verbose:
            print(f"  Running: {result.architecture} ({result.total_synapses} synapses)")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        output = proc.stdout + proc.stderr

        # Check return code first
        if proc.returncode != 0:
            result.error = f"Exit code {proc.returncode}"
            result.success = False
        else:
            # Parse best test accuracy
            best_match = re.search(r"Best Test Accuracy:\s*([\d.]+)%", output)
            final_match = re.search(r"Final Test Accuracy:\s*([\d.]+)%", output)

            if best_match:
                result.best_accuracy = float(best_match.group(1))
                result.success = True
            if final_match:
                result.final_accuracy = float(final_match.group(1))

        if verbose:
            print(f"    -> Best: {result.best_accuracy:.2f}%, Final: {result.final_accuracy:.2f}%")

    except subprocess.TimeoutExpired:
        result.error = "Timeout"
        if verbose:
            print(f"    -> TIMEOUT")
    except Exception as e:
        result.error = str(e)
        if verbose:
            print(f"    -> ERROR: {e}")
    finally:
        os.unlink(config_path)

    return result


def binary_search_hidden_size(
    image_size: int,
    target_accuracy: float,
    min_hidden: int = MIN_HIDDEN,
    max_hidden: int = MAX_HIDDEN,
    epochs: int = 15,
    data_dir: str = "./data",
    gilgamesh_bin: str = "./target/release/gilgamesh",
    verbose: bool = True,
    cache: Optional[dict] = None,
) -> tuple[Optional[TrainingResult], list[TrainingResult]]:
    """
    Binary search to find minimum hidden size that achieves target accuracy.

    Returns:
        Tuple of (best_result achieving target, list of all attempts)
    """
    attempts = []
    best_achieving = None
    best_accuracy_seen = 0.0

    low, high = min_hidden, max_hidden

    if verbose:
        input_size = image_size ** 2
        print(f"\n  Binary search for {target_accuracy}% with image_size={image_size} (input={input_size})")
        print(f"  Search range: hidden=[{low}, {high}]")

    while low <= high:
        mid = (low + high) // 2

        # Check cache
        cache_key = (image_size, mid)
        if cache and cache_key in cache:
            result = cache[cache_key]
            if verbose:
                print(f"    [cached] hidden={mid}: {result.best_accuracy:.2f}%")
        else:
            config = create_physics_config(image_size, mid, epochs=epochs)
            result = run_training(config, data_dir, gilgamesh_bin, verbose=False)
            if cache is not None:
                cache[cache_key] = result

            if verbose:
                status = "OK" if result.success else "FAIL"
                print(f"    hidden={mid}: {result.best_accuracy:.2f}% [{status}]")

        attempts.append(result)
        best_accuracy_seen = max(best_accuracy_seen, result.best_accuracy)

        if result.success and result.best_accuracy >= target_accuracy:
            # Found a valid configuration, try smaller
            best_achieving = result
            high = mid - 1
        else:
            # Need more capacity
            low = mid + 1

        # Early termination: if we're at max hidden and still far from target, give up
        if mid == max_hidden and result.best_accuracy < target_accuracy - 5.0:
            if verbose:
                print(f"    -> Giving up: max hidden ({max_hidden}) only achieves {best_accuracy_seen:.2f}%")
            break

    return best_achieving, attempts


def search_for_target(
    target_accuracy: float,
    image_sizes: list[int],
    epochs: int = 15,
    data_dir: str = "./data",
    gilgamesh_bin: str = "./target/release/gilgamesh",
    verbose: bool = True,
    cache: Optional[dict] = None,
) -> SearchResult:
    """
    Search across multiple image sizes to find minimum synapses for target accuracy.
    """
    search_result = SearchResult(target_accuracy=target_accuracy)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Searching for {target_accuracy}% accuracy")
        print(f"{'='*60}")

    for image_size in image_sizes:
        # Quick check: test max hidden first to see if this image size can even reach target
        cache_key = (image_size, MAX_HIDDEN)
        if cache and cache_key in cache:
            max_result = cache[cache_key]
        else:
            if verbose:
                print(f"\n  Probing image_size={image_size} with max hidden={MAX_HIDDEN}...")
            config = create_physics_config(image_size, MAX_HIDDEN, epochs=epochs)
            max_result = run_training(config, data_dir, gilgamesh_bin, verbose=False)
            if cache is not None:
                cache[cache_key] = max_result
            if verbose:
                print(f"    -> {max_result.best_accuracy:.2f}%")

        # Skip this image size if max hidden can't reach target
        if max_result.best_accuracy < target_accuracy:
            if verbose:
                print(f"  Skipping image_size={image_size}: max achievable is {max_result.best_accuracy:.2f}%")
            search_result.all_attempts.append(max_result)
            continue

        best, attempts = binary_search_hidden_size(
            image_size=image_size,
            target_accuracy=target_accuracy,
            epochs=epochs,
            data_dir=data_dir,
            gilgamesh_bin=gilgamesh_bin,
            verbose=verbose,
            cache=cache,
        )

        search_result.all_attempts.extend(attempts)

        if best:
            if search_result.best_result is None or best.total_synapses < search_result.best_result.total_synapses:
                search_result.best_result = best
                if verbose:
                    print(f"  -> New best: {best.architecture} = {best.total_synapses} synapses ({best.best_accuracy:.2f}%)")

    return search_result


def run_full_search(
    target_accuracies: list[float] = TARGET_ACCURACIES,
    image_sizes: Optional[list[int]] = None,
    epochs: int = 15,
    data_dir: str = "./data",
    gilgamesh_bin: str = "./target/release/gilgamesh",
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> dict[float, SearchResult]:
    """
    Run full binary search for all target accuracies.
    """
    if image_sizes is None:
        image_sizes = list(range(MIN_IMAGE_SIZE, MAX_IMAGE_SIZE + 1))

    # Shared cache to avoid redundant training runs
    cache = {}
    results = {}

    print(f"\nSynapse Search Configuration:")
    print(f"  Target accuracies: {target_accuracies}")
    print(f"  Image sizes: {image_sizes}")
    print(f"  Epochs per run: {epochs}")
    print(f"  Data directory: {data_dir}")

    # Sort targets from hardest to easiest (highest accuracy first)
    # This allows us to use cache effectively
    sorted_targets = sorted(target_accuracies, reverse=True)

    for target in sorted_targets:
        results[target] = search_for_target(
            target_accuracy=target,
            image_sizes=image_sizes,
            epochs=epochs,
            data_dir=data_dir,
            gilgamesh_bin=gilgamesh_bin,
            verbose=verbose,
            cache=cache,
        )

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Minimum Synapses for Target Accuracies")
    print(f"{'='*60}")
    print(f"{'Target':>10} | {'Architecture':>15} | {'Synapses':>10} | {'Achieved':>10}")
    print(f"{'-'*10}-+-{'-'*15}-+-{'-'*10}-+-{'-'*10}")

    for target in sorted(results.keys(), reverse=True):
        r = results[target]
        if r.found:
            print(f"{target:>9.1f}% | {r.best_result.architecture:>15} | {r.min_synapses:>10} | {r.best_result.best_accuracy:>9.2f}%")
        else:
            print(f"{target:>9.1f}% | {'NOT FOUND':>15} | {'-':>10} | {'-':>10}")

    # Save results to file
    if output_file:
        save_results(results, cache, output_file)
        print(f"\nResults saved to: {output_file}")

    return results


def save_results(results: dict[float, SearchResult], cache: dict, output_file: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {},
        "all_runs": [],
    }

    for target, sr in results.items():
        if sr.found:
            output["summary"][str(target)] = {
                "architecture": sr.best_result.architecture,
                "image_size": sr.best_result.image_size,
                "hidden_size": sr.best_result.hidden_size,
                "synapses": sr.min_synapses,
                "accuracy": sr.best_result.best_accuracy,
            }
        else:
            output["summary"][str(target)] = None

    # Add all cached runs
    for (img_size, hidden), result in cache.items():
        output["all_runs"].append({
            "image_size": img_size,
            "hidden_size": hidden,
            "input_size": result.input_size,
            "architecture": result.architecture,
            "synapses": result.total_synapses,
            "best_accuracy": result.best_accuracy,
            "final_accuracy": result.final_accuracy,
            "success": result.success,
        })

    # Sort by synapses
    output["all_runs"].sort(key=lambda x: x["synapses"])

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Binary search for minimum synapses to achieve target accuracies"
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=TARGET_ACCURACIES,
        help=f"Target accuracies to search for (default: {TARGET_ACCURACIES})",
    )
    parser.add_argument(
        "--image-sizes",
        type=int,
        nargs="+",
        default=None,
        help=f"Image sizes to try (default: {MIN_IMAGE_SIZE}-{MAX_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs per run (default: 15)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="MNIST data directory (default: ./data)",
    )
    parser.add_argument(
        "--gilgamesh-bin",
        type=str,
        default="./target/release/gilgamesh",
        help="Path to gilgamesh binary (default: ./target/release/gilgamesh)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer epochs (5) and limited image sizes (5,6,7,8)",
    )

    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        args.image_sizes = [5, 6, 7, 8]
        if args.output is None:
            args.output = "synapse_search_quick.json"

    if args.output is None:
        args.output = f"synapse_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    run_full_search(
        target_accuracies=args.targets,
        image_sizes=args.image_sizes,
        epochs=args.epochs,
        data_dir=args.data_dir,
        gilgamesh_bin=args.gilgamesh_bin,
        output_file=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
