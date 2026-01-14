#!/usr/bin/env python3
"""
Single-neuron SPICE vs Gilgamesh comparison tool.

Automates the full comparison pipeline:
1. Generates SPICE netlist (passive RC model)
2. Runs ngspice simulation
3. Runs Gilgamesh Rust simulation
4. Parses both outputs
5. Generates comprehensive comparison plots

Usage:
    python tools/neuron_compare.py
    python tools/neuron_compare.py --input-current 10e-6 --duration 0.05
    python tools/neuron_compare.py --output-dir ./my_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

# Add SPICE neuronSim to path for importing the generator
SCRIPT_DIR = Path(__file__).resolve().parent
GILGAMESH_ROOT = SCRIPT_DIR.parent
TARSKII_ROOT = GILGAMESH_ROOT.parent
SPICE_NEURON_SIM = TARSKII_ROOT / "SPICE" / "neuronSim"

sys.path.insert(0, str(SPICE_NEURON_SIM))

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


@dataclass
class NeuronParams:
    """Parameters for single neuron comparison."""
    # Membrane
    c_mem: float = 33e-9       # 33 nF
    r_leak: float = 120e3      # 120 kOhm -> tau = 1.2 ms

    # Threshold
    vdd: float = 5.0
    vref: float = 0.0
    threshold_over_vref: float = 0.8  # Threshold at Vref + 0.8V = 0.8V

    # Pulse stretching
    # Shorter tau gives faster reset, closer to Gilgamesh instantaneous reset
    tau_pulse: float = 0.5e-3  # 0.5 ms (reset held for ~0.35ms)

    # Reset switch threshold
    # Lower = longer reset hold time (switch stays on longer)
    # Higher = shorter reset hold time (switch turns off sooner)
    reset_switch_vt: float = 2.5  # Default: half of VDD

    # Hardware timing (for Gilgamesh to match SPICE)
    comparator_delay: float = 50e-9  # 50ns comparator propagation delay
    reset_hold: float = 0.0  # Reset hold period (0 = auto-calculate from tau_pulse)
    v_peak: float = 2.6  # Peak pulse voltage (accounts for diode drop in SPICE)

    # Simulation
    dt: float = 1e-6           # 1 us timestep
    duration: float = 0.05     # 50 ms total

    # Input
    input_current: float = 10e-6  # 10 uA constant input (enough to spike)

    @property
    def tau_m(self) -> float:
        return self.c_mem * self.r_leak

    @property
    def effective_reset_hold(self) -> float:
        """Calculate effective reset hold period.

        If reset_hold is 0, auto-calculate from tau_pulse.

        Theoretical: tau_pulse * ln(2) ≈ 0.693 * tau_pulse (time for pulse to reach VDD/2)
        Empirical: ~0.3 * tau_pulse works better because Gilgamesh uses instant reset
        while SPICE uses RC decay, so Gilgamesh needs a shorter hold period.
        """
        if self.reset_hold > 0:
            return self.reset_hold
        # Auto-calculate from tau_pulse (empirically calibrated)
        return self.tau_pulse * 0.3  # ~0.15ms for tau_pulse=0.5ms

    @property
    def num_steps(self) -> int:
        return int(self.duration / self.dt)

    @property
    def v_steady_state(self) -> float:
        """Steady-state voltage above Vref with constant input."""
        return self.input_current * self.r_leak

    def print_summary(self):
        """Print parameter summary."""
        print("=" * 60)
        print("Neuron Parameters")
        print("=" * 60)
        print(f"  C_mem:      {self.c_mem * 1e9:.1f} nF")
        print(f"  R_leak:     {self.r_leak / 1e3:.1f} kΩ")
        print(f"  tau_m:      {self.tau_m * 1000:.3f} ms")
        print(f"  Threshold:  {self.vref + self.threshold_over_vref:.2f} V ({self.threshold_over_vref:.2f} V above Vref)")
        print(f"  tau_pulse:  {self.tau_pulse * 1000:.3f} ms")
        print(f"  Input:      {self.input_current * 1e6:.2f} µA")
        print(f"  V_ss:       {self.v_steady_state:.3f} V above Vref")
        print(f"  Duration:   {self.duration * 1000:.1f} ms")
        print(f"  dt:         {self.dt * 1e6:.1f} µs")
        print(f"  Will spike: {'Yes' if self.v_steady_state > self.threshold_over_vref else 'No'}")
        print("-" * 60)
        print("Hardware Timing (Gilgamesh)")
        print("-" * 60)
        print(f"  Comp delay: {self.comparator_delay * 1e9:.1f} ns")
        print(f"  Reset hold: {self.effective_reset_hold * 1000:.3f} ms (from tau_pulse)")
        print(f"  V_peak:     {self.v_peak:.2f} V (with diode drop)")
        print("=" * 60)


@dataclass
class SimulationResults:
    """Container for simulation results."""
    time: np.ndarray
    membrane: np.ndarray  # Relative to Vref
    pulse: np.ndarray     # Comparator/pulse output (0-5V)
    spike_times: List[float]  # Times when spikes occurred
    source: str  # "spice" or "gilgamesh"


def generate_spice_netlist(params: NeuronParams, output_dir: Path) -> Path:
    """Generate passive SPICE netlist."""
    from lif_neuron_generator_passive import PassiveNeuronConfig, generate_passive_neuron

    cfg = PassiveNeuronConfig.default()
    cfg.supplies.vdd = params.vdd
    cfg.supplies.vref = params.vref
    cfg.membrane.C_mem_F = params.c_mem
    cfg.membrane.R_leak_ohm = params.r_leak
    cfg.threshold.over_vref_V = params.threshold_over_vref
    cfg.simulation.tstop_s = params.duration
    cfg.simulation.tstep_s = params.dt

    if params.tau_pulse > 0:
        cfg.pulse_stretch.enable = True
        cfg.pulse_stretch.C_pw_F = 100e-9
        cfg.pulse_stretch.R_pw_ohm = params.tau_pulse / cfg.pulse_stretch.C_pw_F

    # Configure reset switch threshold for faster turn-off
    cfg.reset.switch_vt = params.reset_switch_vt

    subckt = generate_passive_neuron(cfg)
    subckt_path = output_dir / "neuron_passive.subckt"
    with open(subckt_path, 'w') as f:
        f.write(subckt)

    csv_abs = (output_dir / 'spice_output.csv').resolve()
    subckt_abs = subckt_path.resolve()

    netlist = f"""* Passive LIF Neuron Test - Gilgamesh Comparison
* Generated by neuron_compare.py

.include {subckt_abs}

* Power supplies
Vdd vdd 0 DC {params.vdd}
Vref vref 0 DC {params.vref}

* Neuron instance (pins: mem vref vdd comp_pulse sum)
Xneuron mem vref vdd comp_pulse sum lif_passive_passive

* Constant current input
Iin 0 sum DC {params.input_current}

* Initial conditions
.ic V(mem)={params.vref}

* Transient analysis
.tran {params.dt} {params.duration} 0 {params.dt}

* Save membrane, pulse output, and comparator output
.control
run
wrdata {csv_abs} v(mem) v(comp_pulse) v(xneuron.comp_out)
.endc

.end
"""
    netlist_path = output_dir / "test_neuron.cir"
    with open(netlist_path, 'w') as f:
        f.write(netlist)

    return netlist_path


def run_ngspice(netlist_path: Path, output_dir: Path) -> Optional[Path]:
    """Run ngspice simulation."""
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        print("ERROR: ngspice not found in PATH")
        return None

    log_path = output_dir / "ngspice.log"
    print(f"Running ngspice: {netlist_path.name}")

    result = subprocess.run(
        [ngspice, "-b", str(netlist_path.resolve())],
        cwd=str(output_dir.resolve()),
        capture_output=True,
        text=True
    )

    with open(log_path, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if result.returncode != 0:
        print(f"ngspice failed with code {result.returncode}")
        print(result.stderr[:500] if result.stderr else "Check ngspice.log")
        return None

    csv_path = output_dir / "spice_output.csv"
    if not csv_path.exists():
        print(f"Expected output CSV not found: {csv_path}")
        return None

    return csv_path


def parse_spice_csv(csv_path: Path, vref: float = 2.5) -> SimulationResults:
    """Parse ngspice wrdata output."""
    times = []
    v_mem = []
    v_pulse = []
    v_comp = []

    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    vm_abs = float(parts[1])
                    # wrdata format: time val1 time val2 time val3 ...
                    vp = float(parts[3]) if len(parts) > 3 else 0.0
                    vc = float(parts[5]) if len(parts) > 5 else vp

                    times.append(t)
                    v_mem.append(vm_abs - vref)  # Convert to relative
                    v_pulse.append(vp)
                    v_comp.append(vc)
                except (ValueError, IndexError):
                    continue

    # Detect spike times (when pulse goes high)
    times_arr = np.array(times)
    pulse_arr = np.array(v_pulse)
    spike_times = []

    if len(pulse_arr) > 1:
        # Find rising edges above 2.5V threshold
        above_thresh = pulse_arr > 2.5
        rising_edges = np.diff(above_thresh.astype(int)) > 0
        spike_indices = np.where(rising_edges)[0]
        spike_times = times_arr[spike_indices].tolist()

    return SimulationResults(
        time=times_arr,
        membrane=np.array(v_mem),
        pulse=pulse_arr,
        spike_times=spike_times,
        source="spice"
    )


def run_gilgamesh(params: NeuronParams, output_dir: Path) -> Optional[Path]:
    """Run Gilgamesh neuron simulation."""
    print("Building Gilgamesh...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=str(GILGAMESH_ROOT),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"cargo build failed: {result.stderr[:500]}")
        return None

    csv_path = output_dir / "gilgamesh_output.csv"

    cmd = [
        str(GILGAMESH_ROOT / "target" / "release" / "gilgamesh"),
        "neuron-test",
        "--tau-m", str(params.tau_m),
        "--dt", str(params.dt),
        "--threshold", str(params.threshold_over_vref + params.vref),
        "--vref", str(params.vref),
        "--input-current", str(params.input_current),
        "--duration", str(params.duration),
        "--output", str(csv_path),
    ]

    if params.tau_pulse > 0:
        cmd.extend(["--tau-pulse", str(params.tau_pulse)])

    # Hardware timing parameters
    if params.comparator_delay > 0:
        cmd.extend(["--comparator-delay", str(params.comparator_delay)])
    if params.effective_reset_hold > 0:
        cmd.extend(["--reset-hold", str(params.effective_reset_hold)])
    # Peak voltage (with diode drop)
    cmd.extend(["--v-peak", str(params.v_peak)])

    print(f"Running Gilgamesh neuron-test...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Gilgamesh failed: {result.stderr}")
        return None

    if not csv_path.exists():
        print(f"Expected output not found: {csv_path}")
        return None

    return csv_path


def parse_gilgamesh_csv(csv_path: Path) -> SimulationResults:
    """Parse Gilgamesh CSV output."""
    times = []
    v_mem = []
    spikes = []
    pulse = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            v_mem.append(float(row['membrane']))
            spike_val = float(row.get('spike', 0))
            spikes.append(spike_val)
            # If there's a pulse column, use it; otherwise derive from spike
            pulse_val = float(row.get('pulse', spike_val * 5.0))
            pulse.append(pulse_val)

    times_arr = np.array(times)
    pulse_arr = np.array(pulse)

    # Detect spike times using rising edges (same as SPICE)
    spike_times = []
    if len(pulse_arr) > 1:
        # Find rising edges above 2.5V threshold
        above_thresh = pulse_arr > 2.5
        rising_edges = np.diff(above_thresh.astype(int)) > 0
        spike_indices = np.where(rising_edges)[0]
        spike_times = times_arr[spike_indices].tolist()

    return SimulationResults(
        time=times_arr,
        membrane=np.array(v_mem),
        pulse=pulse_arr,
        spike_times=spike_times,
        source="gilgamesh"
    )


def compute_metrics(spice: SimulationResults, rust: SimulationResults) -> Dict:
    """Compute comparison metrics."""
    # Interpolate to common time base
    t_start = max(spice.time[0], rust.time[0])
    t_end = min(spice.time[-1], rust.time[-1])
    n_samples = min(len(spice.time), len(rust.time))
    t_common = np.linspace(t_start, t_end, n_samples)

    v_spice = np.interp(t_common, spice.time, spice.membrane)
    v_rust = np.interp(t_common, rust.time, rust.membrane)

    diff = v_spice - v_rust

    return {
        "rms_error_mV": float(np.sqrt(np.mean(diff**2)) * 1000),
        "max_abs_error_mV": float(np.max(np.abs(diff)) * 1000),
        "mean_abs_error_mV": float(np.mean(np.abs(diff)) * 1000),
        "correlation": float(np.corrcoef(v_spice, v_rust)[0, 1]) if len(v_spice) > 1 else 0.0,
        "spice_spike_count": len(spice.spike_times),
        "rust_spike_count": len(rust.spike_times),
        "num_samples": n_samples
    }


def plot_comparison(
    spice: SimulationResults,
    rust: SimulationResults,
    params: NeuronParams,
    metrics: Dict,
    output_path: Path
):
    """Generate comprehensive comparison plot."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed - skipping plot")
        return

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1.5, 1], hspace=0.3)

    t_spice_ms = spice.time * 1000
    t_rust_ms = rust.time * 1000

    # 1. Membrane voltage comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_spice_ms, spice.membrane, 'b-', label='SPICE v(mem)', linewidth=1.2, alpha=0.8)
    ax1.plot(t_rust_ms, rust.membrane, 'r--', label='Gilgamesh', linewidth=1.2, alpha=0.8)
    ax1.axhline(params.threshold_over_vref, color='g', linestyle=':', linewidth=1, label=f'Threshold ({params.threshold_over_vref}V)')
    ax1.set_ylabel('Membrane (V above Vref)')
    ax1.set_title(f'LIF Neuron: SPICE vs Gilgamesh (τ_m={params.tau_m*1000:.2f}ms, I={params.input_current*1e6:.1f}µA)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, params.duration * 1000)

    # 2. Pulse output comparison (overlaid)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t_spice_ms, spice.pulse, 'b-', linewidth=1.2, alpha=0.8, label=f'SPICE ({len(spice.spike_times)} spikes)')
    ax2.plot(t_rust_ms, rust.pulse, 'r--', linewidth=1.2, alpha=0.8, label=f'Gilgamesh ({len(rust.spike_times)} spikes)')
    ax2.axhline(2.5, color='orange', linestyle=':', linewidth=0.8, label='Switch Vt (2.5V)')
    ax2.set_ylabel('Pulse Output (V)')
    ax2.set_title('Pulse Output Comparison')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    max_pulse = max(spice.pulse.max(), rust.pulse.max())
    ax2.set_ylim(-0.2, max_pulse + 0.5)

    # 3. Error/difference
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    t_common = np.linspace(
        max(spice.time[0], rust.time[0]),
        min(spice.time[-1], rust.time[-1]),
        min(len(spice.time), len(rust.time))
    )
    v_spice_interp = np.interp(t_common, spice.time, spice.membrane)
    v_rust_interp = np.interp(t_common, rust.time, rust.membrane)
    diff_mV = (v_spice_interp - v_rust_interp) * 1000

    ax3.plot(t_common * 1000, diff_mV, 'g-', linewidth=0.8)
    ax3.axhline(0, color='k', linestyle=':', linewidth=0.5)
    ax3.fill_between(t_common * 1000, diff_mV, alpha=0.3, color='green')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Error (mV)')
    ax3.set_title(f'Membrane Error: RMS={metrics["rms_error_mV"]:.2f}mV, Max={metrics["max_abs_error_mV"]:.2f}mV, Corr={metrics["correlation"]:.4f}')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")


def plot_spice_only(spice: SimulationResults, params: NeuronParams, output_path: Path):
    """Plot SPICE results only (when Gilgamesh not available)."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    t_ms = spice.time * 1000

    axes[0].plot(t_ms, spice.membrane, 'b-', linewidth=1)
    axes[0].axhline(params.threshold_over_vref, color='g', linestyle=':', label='Threshold')
    axes[0].set_ylabel('Membrane (V above Vref)')
    axes[0].set_title(f'SPICE LIF Neuron (τ_m={params.tau_m*1000:.2f}ms, I={params.input_current*1e6:.1f}µA)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ms, spice.pulse, 'b-', linewidth=1)
    axes[1].axhline(2.5, color='orange', linestyle=':', label='Switch threshold')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Pulse Output (V)')
    axes[1].set_title(f'Comparator/Pulse Output ({len(spice.spike_times)} spikes detected)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.5, 5.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-current", type=float, default=10e-6,
                        help="Input current in Amps (default: 10e-6 = 10µA)")
    parser.add_argument("--duration", type=float, default=0.05,
                        help="Simulation duration in seconds (default: 0.05 = 50ms)")
    parser.add_argument("--dt", type=float, default=1e-6,
                        help="Timestep in seconds (default: 1e-6 = 1µs)")
    parser.add_argument("--tau-m", type=float, default=None,
                        help="Membrane time constant (default: C*R = 1.2ms)")
    parser.add_argument("--tau-pulse", type=float, default=0.5e-3,
                        help="Pulse stretch time constant (default: 0.5ms for faster reset)")
    parser.add_argument("--comparator-delay", type=float, default=50e-9,
                        help="Comparator propagation delay in seconds (default: 50ns)")
    parser.add_argument("--reset-hold", type=float, default=0.0,
                        help="Reset hold period in seconds (default: auto from tau_pulse)")
    parser.add_argument("--v-peak", type=float, default=2.6,
                        help="Peak pulse voltage in V (default: 2.6, accounts for diode drop)")
    parser.add_argument("--no-hardware-timing", action="store_true",
                        help="Disable hardware timing in Gilgamesh (instant response)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: gilgamesh/neuron_comparison)")
    parser.add_argument("--skip-spice", action="store_true",
                        help="Skip SPICE simulation (use existing files)")
    parser.add_argument("--skip-rust", action="store_true",
                        help="Skip Gilgamesh simulation")
    parser.add_argument("--spice-only", action="store_true",
                        help="Only run SPICE (no Gilgamesh)")

    args = parser.parse_args()

    # Set up parameters
    params = NeuronParams(
        input_current=args.input_current,
        duration=args.duration,
        dt=args.dt,
        tau_pulse=args.tau_pulse,
        comparator_delay=0.0 if args.no_hardware_timing else args.comparator_delay,
        reset_hold=0.0 if args.no_hardware_timing else args.reset_hold,
        v_peak=args.v_peak,
    )

    if args.tau_m:
        params.r_leak = args.tau_m / params.c_mem

    params.print_summary()

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = GILGAMESH_ROOT / "neuron_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Run SPICE
    spice_results = None
    if not args.skip_spice:
        print("\n--- Running SPICE Simulation ---")
        netlist_path = generate_spice_netlist(params, output_dir)
        spice_csv = run_ngspice(netlist_path, output_dir)
        if spice_csv:
            spice_results = parse_spice_csv(spice_csv, vref=params.vref)
            print(f"SPICE: {len(spice_results.time)} samples, {len(spice_results.spike_times)} spikes detected")
    else:
        spice_csv = output_dir / "spice_output.csv"
        if spice_csv.exists():
            spice_results = parse_spice_csv(spice_csv, vref=params.vref)
            print(f"Loaded existing SPICE: {len(spice_results.time)} samples")

    # Run Gilgamesh
    rust_results = None
    if not args.skip_rust and not args.spice_only:
        print("\n--- Running Gilgamesh Simulation ---")
        rust_csv = run_gilgamesh(params, output_dir)
        if rust_csv:
            rust_results = parse_gilgamesh_csv(rust_csv)
            print(f"Gilgamesh: {len(rust_results.time)} samples, {len(rust_results.spike_times)} spikes")
    elif not args.spice_only:
        rust_csv = output_dir / "gilgamesh_output.csv"
        if rust_csv.exists():
            rust_results = parse_gilgamesh_csv(rust_csv)
            print(f"Loaded existing Gilgamesh: {len(rust_results.time)} samples")

    # Generate plots and metrics
    print("\n--- Results ---")

    if spice_results and rust_results:
        metrics = compute_metrics(spice_results, rust_results)
        print(f"RMS Error:     {metrics['rms_error_mV']:.3f} mV")
        print(f"Max Error:     {metrics['max_abs_error_mV']:.3f} mV")
        print(f"Correlation:   {metrics['correlation']:.6f}")
        print(f"SPICE spikes:  {metrics['spice_spike_count']}")
        print(f"Rust spikes:   {metrics['rust_spike_count']}")

        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        plot_comparison(spice_results, rust_results, params, metrics, output_dir / "comparison.png")

    elif spice_results:
        print(f"SPICE spikes: {len(spice_results.spike_times)}")
        print(f"Membrane range: {spice_results.membrane.min():.3f}V to {spice_results.membrane.max():.3f}V")
        plot_spice_only(spice_results, params, output_dir / "spice_only.png")

    print(f"\nFiles saved to: {output_dir}")


if __name__ == "__main__":
    main()
