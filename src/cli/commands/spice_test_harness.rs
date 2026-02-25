use anyhow::{Context, Result};

use gilgamesh::hardware::HardwareConfig;
use gilgamesh::neurons::{Leaky, LeakyState, NeuronMode};
use gilgamesh::spice::SpiceNetlist;
use gilgamesh::spice::SpiceParams;
use ndarray::array;

use std::path::Path;

pub(crate) fn run_spice_test_harness(
    output_dir: &str,
    run_ngspice: bool,
    two_neurons: bool,
    input_current: f32,
    synapse_gain: Option<f32>,
    duration: f32,
    dt: f32,
    v_peak: f32,
    enable_inject: bool,
) -> Result<()> {
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    let mut params = SpiceParams::default();
    params.dt = dt;
    // Disable adaptive injection by default for clean threshold alignment.
    if !enable_inject {
        params.threshold.r_inject = 1e12;
    }

    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

    let spice_output = if two_neurons {
        "mini_two_output.txt"
    } else {
        "mini_output.txt"
    };

    let default_syn_gain = (input_current / v_peak.max(1e-6)).max(1e-12);
    let syn_gain = synapse_gain.unwrap_or(default_syn_gain);

    let netlist = if two_neurons {
        SpiceNetlist::two_neuron(&params, input_current, syn_gain, duration, spice_output)
    } else {
        SpiceNetlist::single_neuron(&params, input_current, duration, spice_output)
    };

    let netlist_path = output_path.join("mini.cir");
    netlist.write(&netlist_path)?;

    println!("=== SPICE Mini Harness ===");
    println!();
    println!(
        "Mode:            {}",
        if two_neurons {
            "two-neuron"
        } else {
            "single-neuron"
        }
    );
    println!(
        "Membrane:        C={:.1} nF, R={:.1} kΩ, tau={:.3} ms",
        params.membrane.c_mem * 1e9,
        params.membrane.r_leak / 1e3,
        params.tau_m() * 1e3
    );
    println!("Input current:   {:.3} µA", input_current * 1e6);
    println!(
        "Adapt injection: {}",
        if enable_inject { "enabled" } else { "disabled" }
    );
    if two_neurons {
        println!("Synapse gain:    {:.3e} A/V", syn_gain);
    }
    println!("Duration:        {:.3} ms", duration * 1e3);
    println!("dt:              {:.3} µs", dt * 1e6);
    println!("SPICE netlist:   {}", netlist_path.display());
    println!(
        "SPICE output:    {}",
        output_path.join(spice_output).display()
    );
    println!();

    if run_ngspice {
        println!("Running ngspice...");
        let netlist_name = netlist_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("mini.cir");
        let status = Command::new("ngspice")
            .args(["-b"])
            .arg(netlist_name)
            .current_dir(output_path)
            .status()
            .with_context(|| "Failed to run ngspice. Is it installed?")?;
        if !status.success() {
            anyhow::bail!("ngspice failed (exit code {})", status);
        }
        println!("ngspice complete.");
        println!();
    }

    // Rust comparison
    let vref = params.supply.vref;
    let threshold = vref + params.threshold.over_vref;
    let _tau_m = params.tau_m();
    let c_mem = params.membrane.c_mem;
    let r_leak = params.membrane.r_leak;
    let tau_pulse = params.tau_pulse();
    let (tau_theta, theta_low, theta_high) = compute_injection_thresholds(&params);
    let comparator_delay = params.comparator.prop_delay;
    let reset_hold = if params.pulse_stretch.enable {
        let v_start = v_peak.max(1e-6);
        let v_stop = params.reset.switch_vt.max(1e-6);
        if v_start > v_stop {
            (tau_pulse * (v_start / v_stop).ln()).max(0.0)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let reset_floor = compute_reset_floor(&params, reset_hold, theta_low);

    if two_neurons {
        let rust_out = output_path.join("rust_two_neuron.csv");
        simulate_two_neurons(
            c_mem,
            r_leak,
            threshold,
            vref,
            input_current,
            duration,
            tau_pulse,
            v_peak,
            dt,
            syn_gain,
            comparator_delay,
            reset_hold,
            reset_floor,
            enable_inject,
            tau_theta,
            theta_low,
            theta_high,
            &rust_out,
        )?;
        println!("Rust output:     {}", rust_out.display());
    } else {
        let rust_out = output_path.join("rust_single_neuron.csv");
        simulate_single_neuron(
            c_mem,
            r_leak,
            threshold,
            vref,
            input_current,
            duration,
            tau_pulse,
            v_peak,
            dt,
            comparator_delay,
            reset_hold,
            reset_floor,
            enable_inject,
            tau_theta,
            theta_low,
            theta_high,
            &rust_out,
        )?;
        println!("Rust output:     {}", rust_out.display());
    }

    Ok(())
}

fn compute_injection_thresholds(params: &SpiceParams) -> (f32, f32, f32) {
    let r_top = params.threshold.r_top;
    let r_bottom = params.threshold.r_bottom;
    let r_inject = params.threshold.r_inject.max(1.0);
    let g_top = 1.0 / r_top;
    let g_bottom = 1.0 / r_bottom;
    let g_sum = g_top + g_bottom;
    let vdd = params.supply.vdd;
    let vref = params.supply.vref;
    let v_comp = params.comparator.vhigh;

    let v_low = vref + (vdd - vref) * (r_bottom / (r_top + r_bottom));
    let i_inject = (v_comp - vref) / r_inject;
    let v_high = (vdd * g_top + vref * g_bottom + i_inject) / g_sum;
    let tau_theta = params.threshold.c_adapt / g_sum;

    (tau_theta, v_low - vref, v_high - vref)
}

fn compute_reset_floor(params: &SpiceParams, reset_hold: f32, theta_low: f32) -> f32 {
    if reset_hold <= 0.0 {
        return 0.0;
    }

    let tau_reset = params.reset.mux_ron * params.membrane.c_mem;
    if tau_reset <= 0.0 {
        return 0.0;
    }

    theta_low * (-reset_hold / tau_reset).exp()
}

fn create_hardware_neuron_mode(
    tau_m: f32,
    dt: f32,
    tau_pulse: f32,
    v_peak: f32,
    comparator_delay: f32,
    reset_hold: f32,
    reset_floor: f32,
    enable_inject: bool,
    tau_theta: f32,
    theta_low: f32,
    theta_high: f32,
) -> NeuronMode {
    if enable_inject {
        NeuronMode::Physics {
            tau_m,
            dt,
            tau_pulse,
            v_peak,
            tau_theta,
            theta_low,
            theta_high,
            v_min: reset_floor.max(0.0),
            v_max: 5.0,
            comparator_delay_s: comparator_delay,
            reset_hold_s: reset_hold,
        }
    } else {
        NeuronMode::physics_with_hardware_timing(
            tau_m,
            dt,
            tau_pulse,
            v_peak,
            comparator_delay,
            reset_hold,
        )
    }
}

fn simulate_single_neuron(
    c_mem: f32,
    r_leak: f32,
    threshold: f32,
    vref: f32,
    input_current: f32,
    duration: f32,
    tau_pulse: f32,
    v_peak: f32,
    dt: f32,
    comparator_delay: f32,
    reset_hold: f32,
    reset_floor: f32,
    enable_inject: bool,
    tau_theta: f32,
    theta_low: f32,
    theta_high: f32,
    output_path: &Path,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let v_threshold = threshold - vref;
    let tau_m = c_mem * r_leak;

    let hw = HardwareConfig {
        c_mem,
        r_leak,
        vdd: 5.0,
        vref,
        v_threshold,
        dt,
        i_syn_max: 3e-6,
        i_total_max: 50e-6,
    };

    let mode = create_hardware_neuron_mode(
        tau_m,
        dt,
        tau_pulse,
        v_peak,
        comparator_delay,
        reset_hold,
        reset_floor,
        enable_inject,
        tau_theta,
        theta_low,
        theta_high,
    );

    let neuron = Leaky::new(1, (-dt / tau_m).exp())
        .with_mode(mode)
        .with_threshold(if enable_inject {
            theta_low
        } else {
            v_threshold
        });

    let mut state = if enable_inject {
        LeakyState::new_full_physics(ndarray::Array2::zeros((1, 1)), theta_low)
    } else {
        neuron.init_state_with_hardware_timing(1)
    };

    let num_steps = (duration / dt).max(1.0) as usize;
    let input_step = hw.dv_per_step(input_current);

    let mut file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {:?}", output_path))?;
    writeln!(file, "time,membrane,spike,pulse")?;

    for step in 0..num_steps {
        let time = step as f32 * dt;
        let input = array![[input_step]];
        let (pulse, new_state, cache) = neuron.forward_with_hardware_timing(&input, &state, dt);
        let spiked = cache.spikes[[0, 0]] > 0.5;

        writeln!(
            file,
            "{:.9},{:.6},{:.1},{:.6}",
            time,
            state.mem[[0, 0]],
            if spiked { 1.0 } else { 0.0 },
            pulse[[0, 0]]
        )?;

        state = new_state;
    }

    Ok(())
}

fn simulate_two_neurons(
    c_mem: f32,
    r_leak: f32,
    threshold: f32,
    vref: f32,
    input_current: f32,
    duration: f32,
    tau_pulse: f32,
    v_peak: f32,
    dt: f32,
    synapse_gain: f32,
    comparator_delay: f32,
    reset_hold: f32,
    reset_floor: f32,
    enable_inject: bool,
    tau_theta: f32,
    theta_low: f32,
    theta_high: f32,
    output_path: &Path,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let v_threshold = threshold - vref;
    let tau_m = c_mem * r_leak;

    let hw = HardwareConfig {
        c_mem,
        r_leak,
        vdd: 5.0,
        vref,
        v_threshold,
        dt,
        i_syn_max: 3e-6,
        i_total_max: 50e-6,
    };

    let mode_a = create_hardware_neuron_mode(
        tau_m,
        dt,
        tau_pulse,
        v_peak,
        comparator_delay,
        reset_hold,
        reset_floor,
        enable_inject,
        tau_theta,
        theta_low,
        theta_high,
    );
    let mode_b = create_hardware_neuron_mode(
        tau_m,
        dt,
        tau_pulse,
        v_peak,
        comparator_delay,
        reset_hold,
        reset_floor,
        enable_inject,
        tau_theta,
        theta_low,
        theta_high,
    );

    let neuron_a = Leaky::new(1, (-dt / tau_m).exp())
        .with_mode(mode_a)
        .with_threshold(if enable_inject {
            theta_low
        } else {
            v_threshold
        });
    let neuron_b = Leaky::new(1, (-dt / tau_m).exp())
        .with_mode(mode_b)
        .with_threshold(if enable_inject {
            theta_low
        } else {
            v_threshold
        });

    let mut state_a = if enable_inject {
        LeakyState::new_full_physics(ndarray::Array2::zeros((1, 1)), theta_low)
    } else {
        neuron_a.init_state_with_hardware_timing(1)
    };
    let mut state_b = if enable_inject {
        LeakyState::new_full_physics(ndarray::Array2::zeros((1, 1)), theta_low)
    } else {
        neuron_b.init_state_with_hardware_timing(1)
    };

    let num_steps = (duration / dt).max(1.0) as usize;
    let input_step_a = hw.dv_per_step(input_current);

    let mut file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {:?}", output_path))?;
    writeln!(file, "time,mem_a,pulse_a,mem_b,pulse_b,i_b")?;

    for step in 0..num_steps {
        let time = step as f32 * dt;
        let input_a = array![[input_step_a]];
        let (pulse_a, new_state_a, _) =
            neuron_a.forward_with_hardware_timing(&input_a, &state_a, dt);

        let i_b = synapse_gain * pulse_a[[0, 0]];
        let dv_b = hw.dv_per_step(i_b);
        let input_b = array![[dv_b]];
        let (pulse_b, new_state_b, _) =
            neuron_b.forward_with_hardware_timing(&input_b, &state_b, dt);

        writeln!(
            file,
            "{:.9},{:.6},{:.6},{:.6},{:.6},{:.6e}",
            time,
            state_a.mem[[0, 0]],
            pulse_a[[0, 0]],
            state_b.mem[[0, 0]],
            pulse_b[[0, 0]],
            i_b
        )?;

        state_a = new_state_a;
        state_b = new_state_b;
    }

    Ok(())
}
