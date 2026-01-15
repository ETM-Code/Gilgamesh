use anyhow::{Context, Result};

use gilgamesh::hardware::HardwareConfig;
use gilgamesh::neurons::Leaky;
use gilgamesh::spice::SpiceNetlist;
use gilgamesh::spice::SpiceParams;
use ndarray::array;

use std::path::Path;

use super::neuron_test::run_neuron_test;

pub(crate) fn run_spice_mini(
    output_dir: &str,
    run_ngspice: bool,
    two_neurons: bool,
    input_current: f32,
    synapse_gain: Option<f32>,
    duration: f32,
    dt: f32,
    v_peak: f32,
) -> Result<()> {
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    let mut params = SpiceParams::default();
    params.dt = dt;

    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

    let spice_output = if two_neurons {
        "mini_two_output.txt"
    } else {
        "mini_output.txt"
    };

    let netlist = if two_neurons {
        let syn_gain = synapse_gain.unwrap_or_else(|| {
            // Default: scale so a full pulse roughly matches the input current.
            let pulse_v = params.comparator.vhigh.max(1e-6);
            input_current / pulse_v
        });
        SpiceNetlist::two_neuron(&params, input_current, syn_gain, duration, spice_output)
    } else {
        SpiceNetlist::single_neuron(&params, input_current, duration, spice_output)
    };

    let netlist_path = output_path.join("mini.cir");
    netlist.write(&netlist_path)?;

    println!("=== SPICE Mini Harness ===");
    println!();
    println!("Mode:            {}", if two_neurons { "two-neuron" } else { "single-neuron" });
    println!(
        "Membrane:        C={:.1} nF, R={:.1} kΩ, tau={:.3} ms",
        params.membrane.c_mem * 1e9,
        params.membrane.r_leak / 1e3,
        params.tau_m() * 1e3
    );
    println!("Input current:   {:.3} µA", input_current * 1e6);
    if two_neurons {
        let syn_gain = synapse_gain.unwrap_or_else(|| {
            let pulse_v = params.comparator.vhigh.max(1e-6);
            input_current / pulse_v
        });
        println!("Synapse gain:    {:.3e} A/V", syn_gain);
    }
    println!("Duration:        {:.3} ms", duration * 1e3);
    println!("dt:              {:.3} µs", dt * 1e6);
    println!("SPICE netlist:   {}", netlist_path.display());
    println!("SPICE output:    {}", output_path.join(spice_output).display());
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
    let tau_m = params.tau_m();
    let c_mem = params.membrane.c_mem;
    let r_leak = params.membrane.r_leak;
    let tau_pulse = params.tau_pulse();

    if two_neurons {
        let rust_out = output_path.join("rust_two_neuron.csv");
        run_two_neuron_rust(
            c_mem,
            r_leak,
            threshold,
            vref,
            input_current,
            duration,
            tau_pulse,
            v_peak,
            dt,
            synapse_gain.unwrap_or_else(|| {
                let pulse_v = params.comparator.vhigh.max(1e-6);
                input_current / pulse_v
            }),
            &rust_out,
        )?;
        println!("Rust output:     {}", rust_out.display());
    } else {
        let rust_out = output_path.join("rust_single_neuron.csv");
        run_neuron_test(
            tau_m,
            dt,
            threshold,
            vref,
            input_current,
            duration,
            tau_pulse,
            v_peak,
            0.0,
            0.0,
            c_mem,
            rust_out.to_str().unwrap_or("rust_single_neuron.csv"),
        )?;
    }

    Ok(())
}

fn run_two_neuron_rust(
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

    let neuron_a = Leaky::new_physics_with_pulse(1, tau_m, dt, tau_pulse, v_peak)
        .with_threshold(v_threshold);
    let neuron_b = Leaky::new_physics_with_pulse(1, tau_m, dt, tau_pulse, v_peak)
        .with_threshold(v_threshold);

    let mut state_a = neuron_a.init_state_with_pulse(1);
    let mut state_b = neuron_b.init_state_with_pulse(1);

    let num_steps = (duration / dt).max(1.0) as usize;
    let input_step_a = hw.dv_per_step(input_current);

    let mut file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {:?}", output_path))?;
    writeln!(file, "time,mem_a,pulse_a,mem_b,pulse_b,i_b")?;

    for step in 0..num_steps {
        let time = step as f32 * dt;
        let input_a = array![[input_step_a]];
        let (pulse_a, new_state_a, _) = neuron_a.forward_with_pulse(&input_a, &state_a, dt);

        let i_b = synapse_gain * pulse_a[[0, 0]];
        let dv_b = hw.dv_per_step(i_b);
        let input_b = array![[dv_b]];
        let (pulse_b, new_state_b, _) = neuron_b.forward_with_pulse(&input_b, &state_b, dt);

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
