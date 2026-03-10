use anyhow::{bail, Context, Result};
use clap::Parser;
use gilgamesh::hardware::HardwareConfig;
use gilgamesh::layers::linear::{DEFAULT_SYNAPSE_NEG_GAIN, DEFAULT_SYNAPSE_POS_GAIN};
use gilgamesh::neurons::{Leaky, LeakyState, NeuronMode};
use ndarray::array;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "replay_current_trace")]
#[command(about = "Replay time-varying synaptic current through gilgamesh hardware-timing neuron")]
struct Args {
    /// Input CSV path with header: time_s,cmd_current_a
    #[arg(long)]
    input_csv: PathBuf,

    /// Output CSV path with header: time_s,cmd_current_a,gil_current_a,membrane_v,pulse_v,spike
    #[arg(long)]
    output_csv: PathBuf,

    /// Membrane capacitance (F)
    #[arg(long, default_value = "1.0e-8")]
    c_mem: f32,

    /// Leak resistance (Ohm)
    #[arg(long, default_value = "1.2e5")]
    r_leak: f32,

    /// Threshold above vref (V)
    #[arg(long, default_value = "0.773196")]
    threshold_v: f32,

    /// Reference voltage (V)
    #[arg(long, default_value = "0.0")]
    vref: f32,

    /// Pulse stretch tau (s)
    #[arg(long, default_value = "1.5e-6")]
    tau_pulse: f32,

    /// Pulse peak voltage (V)
    #[arg(long, default_value = "4.42")]
    v_peak: f32,

    /// Comparator delay (s)
    #[arg(long, default_value = "4.0e-8")]
    comparator_delay: f32,

    /// Reset hold (s)
    #[arg(long, default_value = "3.57e-7")]
    reset_hold: f32,

    /// Enable adaptive threshold state (theta dynamics)
    #[arg(long, default_value_t = false)]
    enable_theta_adapt: bool,

    /// Adaptive-threshold time constant (s)
    #[arg(long, default_value = "5.96e-3")]
    tau_theta: f32,

    /// Low threshold equilibrium above vref (V)
    #[arg(long, default_value = "0.773196")]
    theta_low_v: f32,

    /// High threshold equilibrium above vref (V)
    #[arg(long, default_value = "0.773196")]
    theta_high_v: f32,

    /// Positive synapse gain
    #[arg(long, default_value_t = DEFAULT_SYNAPSE_POS_GAIN)]
    pos_gain: f32,

    /// Negative synapse gain
    #[arg(long, default_value_t = DEFAULT_SYNAPSE_NEG_GAIN)]
    neg_gain: f32,

    /// Optional absolute current cap after gain (A). <=0 disables cap.
    #[arg(long, default_value = "5.0e-5")]
    total_current_cap: f32,
}

fn parse_input_rows(path: &PathBuf) -> Result<Vec<(f32, f32)>> {
    let file = File::open(path).with_context(|| format!("Failed to open input CSV {:?}", path))?;
    let mut rows = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("Failed reading line {} from {:?}", idx + 1, path))?;
        if idx == 0 && line.to_ascii_lowercase().contains("time") {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            bail!("Invalid CSV line {} in {:?}: {}", idx + 1, path, line);
        }
        let t: f32 = parts[0].trim().parse().with_context(|| format!("Bad time at line {}", idx + 1))?;
        let i: f32 = parts[1]
            .trim()
            .parse()
            .with_context(|| format!("Bad current at line {}", idx + 1))?;
        rows.push((t, i));
    }
    if rows.len() < 2 {
        bail!("Need at least 2 rows in input CSV {:?}", path);
    }
    Ok(rows)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let rows = parse_input_rows(&args.input_csv)?;

    let dt0 = (rows[1].0 - rows[0].0).abs().max(1e-9);
    let tau_m = args.c_mem * args.r_leak;
    let beta = (-dt0 / tau_m).exp();

    let _hw = HardwareConfig {
        c_mem: args.c_mem,
        r_leak: args.r_leak,
        vdd: 5.0,
        vref: args.vref,
        v_threshold: args.threshold_v,
        dt: dt0,
        i_syn_max: 3e-6,
        i_total_max: 50e-6,
    };

    let mode = NeuronMode::Physics {
        tau_m,
        dt: dt0,
        tau_pulse: args.tau_pulse,
        v_peak: args.v_peak,
        tau_theta: args.tau_theta,
        theta_low: args.theta_low_v,
        theta_high: args.theta_high_v,
        v_min: 0.0,
        v_max: 5.0,
        comparator_delay_s: args.comparator_delay,
        reset_hold_s: args.reset_hold,
    };

    let neuron = Leaky::new(1, beta)
        .with_mode(mode)
        .with_threshold(args.threshold_v);
    let mut state = if args.enable_theta_adapt {
        LeakyState::new_full_physics(array![[0.0]], args.theta_low_v)
    } else {
        neuron.init_state_with_hardware_timing(1)
    };

    let mut out = File::create(&args.output_csv)
        .with_context(|| format!("Failed to create output CSV {:?}", args.output_csv))?;
    writeln!(
        out,
        "time_s,cmd_current_a,gil_current_a,membrane_v,pulse_v,spike"
    )?;

    for idx in 0..rows.len() {
        let (t, cmd_i) = rows[idx];
        let dt = if idx == 0 {
            dt0
        } else {
            (rows[idx].0 - rows[idx - 1].0).max(1e-9)
        };

        let mut gil_i = if cmd_i >= 0.0 {
            cmd_i * args.pos_gain
        } else {
            cmd_i * args.neg_gain
        };
        if args.total_current_cap > 0.0 {
            gil_i = gil_i.clamp(-args.total_current_cap, args.total_current_cap);
        }

        // The physics-mode membrane update uses:
        //   mem_new = mem_prev*exp(-dt/tau) + input*(1-exp(-dt/tau))
        // where `input` is the steady-state drive voltage (V_ss = I*R).
        let v_ss = gil_i * args.r_leak;
        let input = array![[v_ss]];
        let (pulse, new_state, cache) = neuron.forward_with_hardware_timing(&input, &state, dt);
        let spike = if cache.spikes[[0, 0]] > 0.5 { 1.0 } else { 0.0 };

        writeln!(
            out,
            "{:.9},{:.9e},{:.9e},{:.6},{:.6},{}",
            t, cmd_i, gil_i, state.mem[[0, 0]], pulse[[0, 0]], spike
        )?;
        state = new_state;
    }

    Ok(())
}
