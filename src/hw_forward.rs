//! Hardware-accurate forward pass for fine-tuning.
//!
//! Reimplements the essential HardwareNetwork dynamics from the emulator
//! so gilgamesh can use it as a forward pass during training.
//! This avoids the cyclic dependency between gilgamesh and tarski-core.
//!
//! Uses the two-phase pulse model: full current for t_on, zero for t_off.

use ndarray::Array2;

// Hardware constants (must match emulator/tarski-core/src/neuron/hardware_sim.rs)
const C_MEM: f64 = 10e-9;
const R_LEAK: f64 = 120e3;
const TAU_M: f64 = C_MEM * R_LEAK; // 1.2ms
const V_DD: f64 = 5.0;
const V_REF: f64 = 2.5;
const DAC_VREF: f64 = 5.0; // MCP4728 VDD reference (factory EEPROM default)
const V_BE: f64 = 0.65;
const R_SET_INPUT: f64 = 174e3;
const R_SET_SYNAPSE: f64 = 10e6;
const R_TOP: f64 = 820e3;
const R_BOTTOM: f64 = 150e3;
const R_STRETCH: f64 = 150e3;
const C_STRETCH: f64 = 5.8e-9;
const DT: f64 = 0.001; // 1ms

/// Resting threshold from divider
fn theta_0() -> f64 {
    let g_top = 1.0 / R_TOP;
    let g_bottom = 1.0 / R_BOTTOM;
    ((V_DD * g_top + V_REF * g_bottom) / (g_top + g_bottom)) - V_REF
}

/// Pulse duty cycle
fn duty_cycle() -> f64 {
    let v_peak = V_DD - 0.56;
    let tau_pulse = R_STRETCH * C_STRETCH;
    let t_on = tau_pulse * (v_peak / V_REF).ln();
    (t_on / DT).min(1.0)
}

/// Advance a single neuron's membrane by dt seconds with given current.
/// Returns (new_u, spiked).
fn advance_neuron(u: f64, theta: f64, i_syn: f64, dt: f64) -> (f64, bool) {
    let u_inf = R_LEAK * i_syn;
    let decay = (-dt / TAU_M).exp();
    let mut u_new = u_inf + (u - u_inf) * decay;
    u_new = u_new.clamp(0.0, V_DD - V_REF);

    if u_new > theta {
        // Spike: subtract threshold
        (u_new - theta, true)
    } else {
        (u_new, false)
    }
}

/// Run a complete hardware-accurate inference.
///
/// Takes gilgamesh's fc1 outputs and quantized fc2 weights,
/// simulates the 9-hidden + 10-output neuron network for num_steps,
/// returns output spike counts.
///
/// `fc1_outputs`: [batch, 9] — gilgamesh fc1 weighted sums per sample
/// `fc2_quantized`: [9][10] — integer weights (-7 to +7)
/// `dac_scale`: scale factor for fc1→DAC mapping (1.16× optimal)
/// `num_steps`: number of simulation timesteps (25)
pub fn hw_forward_batch(
    fc1_outputs: &Array2<f32>,
    fc2_quantized: &[Vec<i8>],
    dac_scale: f64,
    num_steps: usize,
) -> Array2<f32> {
    let batch_size = fc1_outputs.nrows();
    let mut spike_counts = Array2::zeros((batch_size, 10));
    let theta = theta_0();
    let duty = duty_cycle();
    let t_on = duty * DT;
    let t_off = DT - t_on;
    let nominal_scale = theta * R_SET_INPUT / R_LEAK;
    let scale = nominal_scale * dac_scale;

    // Per-synapse current unit
    let i_unit = (V_DD - V_BE) / R_SET_SYNAPSE;

    for b in 0..batch_size {
        // Per-neuron state
        let mut h_u = [0.0f64; 9];
        let mut h_theta = [theta; 9];
        let mut o_u = [0.0f64; 10];
        let mut o_theta = [theta; 10];
        let mut o_spike_counts = [0.0f32; 10];

        // Convert fc1 to DAC currents
        let mut h_current = [0.0f64; 9];
        for i in 0..9 {
            let g = fc1_outputs[[b, i]] as f64;
            let v_dac = (g * scale + V_BE).clamp(0.0, DAC_VREF);
            h_current[i] = if v_dac > V_BE {
                (v_dac - V_BE) / R_SET_INPUT
            } else {
                0.0
            };
        }

        for _step in 0..num_steps {
            // 1. Advance hidden neurons (full timestep, constant DAC current)
            let mut h_spikes = [false; 9];
            for i in 0..9 {
                let (u_new, spiked) = advance_neuron(h_u[i], h_theta[i], h_current[i], DT);
                h_u[i] = u_new;
                h_spikes[i] = spiked;
                if spiked {
                    h_theta[i] = theta; // reset threshold (simplified)
                }
            }

            // 2. Compute synapse current per output neuron
            let any_spike = h_spikes.iter().any(|&s| s);

            if any_spike {
                // Compute full synapse current (un-duty-scaled)
                let mut o_current = [0.0f64; 10];
                for o in 0..10 {
                    for h in 0..9 {
                        if h_spikes[h] {
                            let w = if h < fc2_quantized.len() && o < fc2_quantized[h].len() {
                                fc2_quantized[h][o] as f64
                            } else {
                                0.0
                            };
                            o_current[o] += w * i_unit;
                        }
                    }
                }

                // Phase 1: full current for t_on
                for i in 0..10 {
                    let (u_new, spiked) = advance_neuron(o_u[i], o_theta[i], o_current[i], t_on);
                    o_u[i] = u_new;
                    if spiked {
                        o_spike_counts[i] += 1.0;
                        o_theta[i] = theta;
                    }
                }

                // Phase 2: zero current for t_off
                for i in 0..10 {
                    let (u_new, spiked) = advance_neuron(o_u[i], o_theta[i], 0.0, t_off);
                    o_u[i] = u_new;
                    if spiked {
                        o_spike_counts[i] += 1.0;
                        o_theta[i] = theta;
                    }
                }
            } else {
                // No spikes — just leak
                for i in 0..10 {
                    let (u_new, spiked) = advance_neuron(o_u[i], o_theta[i], 0.0, DT);
                    o_u[i] = u_new;
                    if spiked {
                        o_spike_counts[i] += 1.0;
                        o_theta[i] = theta;
                    }
                }
            }
        }

        for i in 0..10 {
            spike_counts[[b, i]] = o_spike_counts[i];
        }
    }

    spike_counts
}

/// Run hardware forward for a single sample (convenience wrapper).
pub fn hw_forward_single(
    fc1_output: &[f64],
    fc2_quantized: &[Vec<i8>],
    dac_scale: f64,
    num_steps: usize,
) -> Vec<f64> {
    let fc1 = Array2::from_shape_fn((1, fc1_output.len()), |(_, j)| fc1_output[j] as f32);
    let counts = hw_forward_batch(&fc1, fc2_quantized, dac_scale, num_steps);
    counts
        .row(0)
        .to_vec()
        .into_iter()
        .map(|v| v as f64)
        .collect()
}
