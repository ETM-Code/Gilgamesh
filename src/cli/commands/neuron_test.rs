use anyhow::{Context, Result};

pub(crate) fn run_neuron_test(
    tau_m: f32,
    dt: f32,
    threshold: f32,
    vref: f32,
    input_current: f32,
    duration: f32,
    tau_pulse: f32,
    v_peak: f32,
    comparator_delay: f32,
    reset_hold: f32,
    c_mem: f32,
    output_path: &str,
) -> Result<()> {
    use gilgamesh::hardware::HardwareConfig;
    use gilgamesh::neurons::{Leaky, NeuronMode};
    use std::fs::File;
    use std::io::Write;

    let r_leak = tau_m / c_mem;
    let v_threshold = threshold - vref;

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

    let use_hardware_timing = comparator_delay > 0.0 || reset_hold > 0.0;

    println!("=== Single Neuron Test (Hardware Mode) ===");
    println!();
    hw.print_summary();
    println!();
    println!("Test parameters:");
    println!("  Input current    = {:.4} µA", input_current * 1e6);
    println!("  Duration         = {:.2} ms", duration * 1000.0);
    println!("  tau_pulse        = {:.4} ms", tau_pulse * 1000.0);
    println!("  v_peak           = {:.2} V", v_peak);
    if use_hardware_timing {
        println!("  comparator_delay = {:.1} ns", comparator_delay * 1e9);
        println!("  reset_hold       = {:.3} ms", reset_hold * 1000.0);
    }
    println!();

    let neuron = if use_hardware_timing {
        let mode = NeuronMode::physics_with_hardware_timing(
            tau_m,
            dt,
            tau_pulse,
            v_peak,
            comparator_delay,
            reset_hold,
        );
        Leaky::new(1, (-dt / tau_m).exp())
            .with_mode(mode)
            .with_threshold(v_threshold)
    } else if tau_pulse > 0.0 {
        Leaky::new_physics_with_pulse(1, tau_m, dt, tau_pulse, v_peak).with_threshold(v_threshold)
    } else {
        Leaky::new_physics(1, tau_m, dt).with_threshold(v_threshold)
    };

    let num_steps = (duration / dt) as usize;
    println!(
        "Running {} timesteps{}...",
        num_steps,
        if use_hardware_timing {
            " (with hardware timing)"
        } else {
            ""
        }
    );

    let mut file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {}", output_path))?;

    writeln!(file, "time,membrane,spike,pulse")?;

    let mut state = if use_hardware_timing {
        neuron.init_state_with_hardware_timing(1)
    } else {
        neuron.init_state(1)
    };
    let mut spike_count = 0;
    let mut pulse_value = 0.0f32;
    let mut prev_pulse_high = false;

    let input_per_step = hw.dv_per_step(input_current);
    println!(
        "  Input/step       = {:.6} V ({:.4} mV)",
        input_per_step,
        input_per_step * 1e3
    );
    println!(
        "  Steady-state     = {:.4} V (I×R)",
        input_current * r_leak
    );
    println!();

    let input = ndarray::array![[input_per_step]];

    for step in 0..num_steps {
        let time = step as f32 * dt;

        let (output, new_state, _) = if use_hardware_timing {
            neuron.forward_with_hardware_timing(&input, &state, dt)
        } else {
            neuron.forward(&input, &state)
        };

        let spiked = output[[0, 0]] > 0.5;

        if use_hardware_timing {
            pulse_value = output[[0, 0]];
            let pulse_high = pulse_value > 2.5;
            if pulse_high && !prev_pulse_high {
                spike_count += 1;
            }
            prev_pulse_high = pulse_high;
        } else {
            if spiked {
                pulse_value = v_peak;
                spike_count += 1;
            } else if tau_pulse > 0.0 {
                let decay = (-dt / tau_pulse).exp();
                pulse_value *= decay;
            }
        }

        let membrane = state.mem[[0, 0]];

        let spike_val = if spiked { 1.0 } else { 0.0 };
        writeln!(
            file,
            "{:.9},{:.6},{:.1},{:.6}",
            time, membrane, spike_val, pulse_value
        )?;

        state = new_state;
    }

    println!();
    println!("Simulation complete:");
    println!("  Total spikes:    {}", spike_count);
    println!("  Final membrane:  {:.4} V", state.mem[[0, 0]]);
    println!("  Output:          {}", output_path);

    Ok(())
}
