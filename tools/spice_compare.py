#!/usr/bin/env python3
"""Run SPICE + Rust equivalent model comparison and produce plots.

This tool orchestrates the existing neuronSim SPICE generator and the Rust
comparator implemented in this crate. It executes four phases:

1. (Optional) Run the SPICE network generator to produce ngspice outputs.
2. Run the Rust comparator (`cargo run -- compare …`) to compute metrics.
3. Load the JSON report emitted by the comparator.
4. Emit summary plots and a metrics CSV inside the chosen output directory.

Example:
    python tools/spice_compare.py \
        --mode detailed \
        --output-dir comparison_runs/detailed
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


def resolve_paths() -> tuple[Path, Path]:
    """Return (gilgamesh_root, spice_root) based on this script location."""
    this_file = Path(__file__).resolve()
    gilgamesh_root = this_file.parent.parent
    spice_root = gilgamesh_root.parent / "SPICE" / "neuronSim"
    return gilgamesh_root, spice_root


def run_spice(
    generator: Path,
    network: Path,
    neuron: Path,
    mode: str,
    runs: int = 1,
) -> List[float]:
    cmd = [
        sys.executable,
        str(generator),
        "--network",
        str(network),
        "--neuron",
        str(neuron),
        "--mode",
        mode,
        "--yes",
    ]
    print(f"[spice] Command: {' '.join(cmd)}")
    durations: List[float] = []
    for idx in range(runs):
        if runs > 1:
            print(f"[spice] Benchmark run {idx + 1}/{runs}")
        start = time.perf_counter()
        subprocess.run(cmd, cwd=str(generator.parent), check=True)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"[spice] Completed in {elapsed:.3f}s")
    return durations


def run_rust_comparator(
    gilgamesh_root: Path,
    network: Path,
    neuron: Path,
    spice_csv: Path,
    json_out: Path,
    equivalent_csv: Path,
    release: bool,
) -> Dict:
    cmd = ["cargo", "run"]
    if release:
        cmd.append("--release")
    cmd.extend(
        [
            "--",
            "compare",
            "--network",
            str(network),
            "--neuron",
            str(neuron),
            "--spice-csv",
            str(spice_csv),
            "--json-out",
            str(json_out),
            "--equivalent-csv",
            str(equivalent_csv),
        ]
    )
    print(f"[rust] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(gilgamesh_root), check=True)

    with open(json_out, "r", encoding="utf-8") as fh:
        return json.load(fh)


def benchmark_rust_comparator(
    gilgamesh_root: Path,
    network: Path,
    neuron: Path,
    spice_csv: Path,
    json_out: Path,
    equivalent_csv: Path,
    runs: int = 1,
    release: bool = False,
    prebuild: bool = True,
) -> tuple[Dict, List[float]]:
    durations: List[float] = []
    result: Optional[Dict] = None
    if prebuild:
        build_cmd = ["cargo", "build"]
        if release:
            build_cmd.append("--release")
        print(f"[rust] Pre-building comparator: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, cwd=str(gilgamesh_root), check=True)
    for idx in range(runs):
        if runs > 1:
            print(f"[rust] Benchmark run {idx + 1}/{runs}")
        start = time.perf_counter()
        result = run_rust_comparator(
            gilgamesh_root,
            network,
            neuron,
            spice_csv,
            json_out,
            equivalent_csv,
            release,
        )
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"[rust] Completed in {elapsed:.3f}s")
    if result is None:
        raise RuntimeError("rust comparator did not produce a result")
    return result, durations


def write_metrics_csv(metrics: List[Dict], path: Path) -> None:
    if not metrics:
        return
    header = ["signal", "rms_error", "mean_abs_error", "max_abs_error", "max_abs_error_time", "sample_count"]
    lines = [",".join(header)]
    for m in metrics:
        lines.append(
            ",".join(
                [
                    m["signal"],
                    f"{m['rms_error']:.6e}",
                    f"{m['mean_abs_error']:.6e}",
                    f"{m['max_abs_error']:.6e}",
                    f"{m['max_abs_error_time']:.6e}",
                    str(m["sample_count"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_signal(signal: str, samples: List[Dict], outdir: Path) -> None:
    if not samples:
        return
    if plt is None:
        print(f"[plot] matplotlib not installed; skipping plot for {signal}")
        return
    times = [s["time_s"] for s in samples]
    eq_vals = [s["equivalent"] for s in samples]
    spice_vals = [s["spice"] for s in samples]
    deltas = [s["delta"] for s in samples]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_top.plot(times, eq_vals, label="Equivalent", linewidth=1.6)
    ax_top.plot(times, spice_vals, label="SPICE", linewidth=1.0, linestyle="--")
    ax_top.set_ylabel("Voltage (V)")
    ax_top.set_title(f"{signal} — Equivalent vs SPICE")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    ax_bottom.plot(times, deltas, color="tab:red", linewidth=1.4)
    ax_bottom.axhline(0.0, color="k", linewidth=0.8, linestyle=":")
    ax_bottom.set_ylabel("Delta (V)")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.grid(True, alpha=0.3)

    fig.tight_layout()
    outfile = outdir / f"{signal}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def summarise_durations(durations: List[float], sample_count: Optional[int] = None) -> Optional[Dict[str, float]]:
    if not durations:
        return None
    mean_s = statistics.fmean(durations)
    summary: Dict[str, float] = {
        "runs": float(len(durations)),
        "total_s": float(sum(durations)),
        "mean_s": float(mean_s),
        "median_s": float(statistics.median(durations)),
        "min_s": float(min(durations)),
        "max_s": float(max(durations)),
    }
    if sample_count and mean_s > 0:
        summary["mean_ms_per_sample"] = float((mean_s / sample_count) * 1e3)
        summary["samples_per_second"] = float(sample_count / mean_s)
        summary["sample_count"] = float(sample_count)
    return summary


def write_benchmark_report(data: Dict, path: Path) -> None:
    if not data:
        return
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    gilgamesh_root, default_spice_root = resolve_paths()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fast", "detailed"], default="detailed")
    parser.add_argument("--network", type=Path, help="Path to network JSON", default=None)
    parser.add_argument("--neuron", type=Path, help="Path to neuron JSON", default=None)
    parser.add_argument("--spice-csv", type=Path, default=None, help="Use an existing SPICE CSV")
    parser.add_argument("--output-dir", type=Path, default=gilgamesh_root / "comparison_outputs")
    parser.add_argument(
        "--spice-root",
        type=Path,
        default=default_spice_root,
        help="Root directory of neuronSim (contains lif_network_generator.py)",
    )
    parser.add_argument("--skip-spice", action="store_true", help="Do not re-run ngspice")
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=1,
        help="How many times to execute each simulator when measuring runtime.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Run the Rust comparator with cargo --release for benchmarking.",
    )
    parser.add_argument(
        "--no-prebuild",
        action="store_true",
        help="Skip the initial cargo build step before benchmarking.",
    )

    args = parser.parse_args()

    spice_root = args.spice_root.resolve()
    generator = spice_root / "lif_network_generator.py"
    if not generator.exists():
        raise SystemExit(f"Could not find lif_network_generator.py at {generator}")

    network = (args.network or (spice_root / "defaults" / "network_default.json")).resolve()
    neuron = (args.neuron or (spice_root / "defaults" / "neuron_default.json")).resolve()

    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    bench_runs = max(1, args.benchmark_runs)
    spice_durations: List[float] = []

    if not args.skip_spice and args.spice_csv is None:
        spice_durations = run_spice(generator, network, neuron, args.mode, runs=bench_runs)
    elif args.skip_spice:
        print("[spice] Skipping SPICE execution by request; no timing recorded")
    else:
        print("[spice] Using pre-generated CSV; no timing recorded")

    spice_csv = (
        args.spice_csv.resolve()
        if args.spice_csv is not None
        else (spice_root / "outputs" / f"lif_{args.mode}.csv").resolve()
    )
    if not spice_csv.exists():
        raise SystemExit(f"Expected SPICE CSV at {spice_csv}; rerun with --skip-spice disabled?")

    comparison_json = output_dir / f"comparison_{args.mode}.json"
    equivalent_csv = output_dir / f"equivalent_{args.mode}.csv"

    result, rust_durations = benchmark_rust_comparator(
        gilgamesh_root,
        network,
        neuron,
        spice_csv,
        comparison_json,
        equivalent_csv,
        runs=bench_runs,
        release=args.release,
        prebuild=not args.no_prebuild,
    )

    metrics = result.get("metrics", [])
    print("\n=== Metrics ===")
    for metric in metrics:
        print(
            f"{metric['signal']}: rms={metric['rms_error']:.6e}, "
            f"mean_abs={metric['mean_abs_error']:.6e}, max_abs={metric['max_abs_error']:.6e} "
            f"@ {metric['max_abs_error_time']:.6e}s (n={metric['sample_count']})"
        )

    write_metrics_csv(metrics, output_dir / f"metrics_{args.mode}.csv")

    benchmark: Dict[str, Dict[str, float]] = {}
    sample_count = None
    if metrics:
        sample_count = metrics[0].get("sample_count")
    elif result.get("equivalent", {}).get("samples"):
        sample_count = len(result["equivalent"]["samples"])

    spice_summary = summarise_durations(spice_durations)
    if spice_summary:
        benchmark["spice"] = spice_summary
        print(
            f"\n[bench] SPICE mean {spice_summary['mean_s']:.3f}s over {int(spice_summary['runs'])} run(s)"
        )

    rust_summary = summarise_durations(rust_durations, sample_count=sample_count)
    if rust_summary:
        benchmark["rust"] = rust_summary
        print(
            f"[bench] Rust comparator mean {rust_summary['mean_s']:.3f}s over {int(rust_summary['runs'])} run(s)"
        )

    if "spice" in benchmark and "rust" in benchmark:
        mean_spice = benchmark["spice"]["mean_s"]
        mean_rust = benchmark["rust"]["mean_s"]
        if mean_rust > 0:
            benchmark["rust_speedup_vs_spice"] = mean_spice / mean_rust
            print(
                f"[bench] Rust speedup vs SPICE (mean): {benchmark['rust_speedup_vs_spice']:.2f}×"
            )

    write_benchmark_report(benchmark, output_dir / f"benchmark_{args.mode}.json")

    signal_series: Dict[str, List[Dict]] = result.get("signal_series", {})
    for signal, samples in signal_series.items():
        plot_signal(signal, samples, plots_dir)

    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
