//! Hardware-accurate forward pass for fine-tuning.
//!
//! Reimplements the essential HardwareNetwork dynamics from the emulator
//! so gilgamesh can use it as a forward pass during training.
//! This avoids the cyclic dependency between gilgamesh and tarski-core.
//!
//! Uses the two-phase pulse model: full current for t_on, zero for t_off.
//!
//! REFERENCE FRAME: this module treats `u` as the **absolute membrane
//! voltage above GND**, matching the real Tarski PCB topology
//! (R_bottom → GND, membrane rests at ~0 V, current drives it upward).
//! Earlier revisions assumed R_bottom → V_REF = 2.5 V, which was wrong
//! per the netlist. The threshold is now simply
//! `theta_0 = V_DD × R_bot / (R_top + R_bot)`.

use ndarray::Array2;

// Hardware constants (must match emulator/tarski-core/src/neuron/hardware_sim.rs)
const C_MEM: f64 = 10e-9;
const R_LEAK: f64 = 120e3;
const TAU_M: f64 = C_MEM * R_LEAK; // 1.2ms
const V_DD: f64 = 5.0;
const DAC_VREF: f64 = 5.0; // MCP4728 VDD reference (factory EEPROM default)
const V_BE: f64 = 0.65;
const R_SET_INPUT: f64 = 174e3;
const R_SET_SYNAPSE: f64 = 10e6;
const R_TOP: f64 = 820e3;
/// Default R_bottom for the threshold divider. **Per-layer values** can be
/// supplied to [`hw_forward_batch_with_thresholds`]; this nominal is used
/// by the legacy shared-threshold entry point.
pub const R_BOTTOM_NOMINAL: f64 = 220e3;
const R_STRETCH: f64 = 150e3;
const C_STRETCH: f64 = 5.8e-9;
const DT: f64 = 0.001; // 1ms

/// Resting threshold from a GND-referenced divider:
/// `V_th = V_DD × R_bot / (R_top + R_bot)`.
pub fn theta_from_r_bottom(r_bottom: f64) -> f64 {
    V_DD * r_bottom / (R_TOP + r_bottom)
}

/// Inverse of [`theta_from_r_bottom`] — given a target threshold voltage,
/// compute the R_bottom value that would produce it (with R_top fixed).
pub fn r_bottom_for_theta(target_v: f64) -> f64 {
    R_TOP * target_v / (V_DD - target_v)
}

/// Back-compat shared threshold. Uses the nominal 220 kΩ R_bottom.
fn theta_0() -> f64 {
    theta_from_r_bottom(R_BOTTOM_NOMINAL)
}

/// Pulse duty cycle: fraction of DT that the stretched V_out pulse
/// decays from v_peak down to V_BE (below which the NPN synapse mirror
/// can't conduct meaningful current). Previously this was
/// `ln(v_peak / V_REF=2.5)` in the V_REF-referenced model; now it is
/// `ln(v_peak / V_BE)` in the GND-referenced model.
fn duty_cycle() -> f64 {
    let v_peak = V_DD - 0.56;
    let tau_pulse = R_STRETCH * C_STRETCH;
    let t_on = tau_pulse * (v_peak / V_BE).ln();
    (t_on / DT).min(1.0)
}

/// Advance a single neuron's membrane by dt seconds with given current.
/// Returns (new_u, spiked).
fn advance_neuron(u: f64, theta: f64, i_syn: f64, dt: f64) -> (f64, bool) {
    let u_inf = R_LEAK * i_syn;
    let decay = (-dt / TAU_M).exp();
    let mut u_new = u_inf + (u - u_inf) * decay;
    // Membrane range is [0, V_DD]; real-hardware caps at the analog rail.
    u_new = u_new.clamp(0.0, V_DD);

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
    let theta = theta_0();
    hw_forward_batch_with_thresholds(
        fc1_outputs,
        fc2_quantized,
        dac_scale,
        num_steps,
        theta,
        theta,
    )
}

/// Hardware forward pass with explicit per-layer threshold voltages.
/// `hidden_theta` and `output_theta` are absolute membrane thresholds in
/// volts (GND-referenced), typically `theta_from_r_bottom(r_bot)`.
pub fn hw_forward_batch_with_thresholds(
    fc1_outputs: &Array2<f32>,
    fc2_quantized: &[Vec<i8>],
    dac_scale: f64,
    num_steps: usize,
    hidden_theta: f64,
    output_theta: f64,
) -> Array2<f32> {
    let batch_size = fc1_outputs.nrows();
    let mut spike_counts = Array2::zeros((batch_size, 10));
    let duty = duty_cycle();
    let t_on = duty * DT;
    let t_off = DT - t_on;
    // fc1 → DAC scale is **threshold-independent**: we pick V_DAC so that
    // the membrane asymptote `u_inf = R_LEAK × I_in` equals `g` (in volts).
    //   I_in  = (V_DAC − V_BE) / R_SET_INPUT                  (emulator sign; PNP
    //                                                          inversion is handled
    //                                                          by the host driver)
    //   u_inf = R_LEAK × I_in = (V_DAC − V_BE) × R_LEAK/R_SET
    //   set u_inf = g  ⇒  V_DAC = g × R_SET/R_LEAK + V_BE
    //
    // `dac_scale` (default 1.0) is a small headroom factor that can be
    // tuned per-board to compensate for parts tolerance.
    //
    // Older revisions had `nominal_scale = theta × R_SET/R_LEAK`, which made
    // the scaling depend on threshold — a quirk introduced when theta was a
    // single global constant. It doesn't correspond to anything on the real
    // board (the netlist's current-mirror equation has no theta term).
    let nominal_scale = R_SET_INPUT / R_LEAK;
    let scale = nominal_scale * dac_scale;

    // Per-synapse current unit
    let i_unit = (V_DD - V_BE) / R_SET_SYNAPSE;

    for b in 0..batch_size {
        // Per-neuron state
        let mut h_u = [0.0f64; 9];
        let mut h_theta_state = [hidden_theta; 9];
        let mut o_u = [0.0f64; 10];
        let mut o_theta_state = [output_theta; 10];
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
                let (u_new, spiked) = advance_neuron(h_u[i], h_theta_state[i], h_current[i], DT);
                h_u[i] = u_new;
                h_spikes[i] = spiked;
                if spiked {
                    h_theta_state[i] = hidden_theta; // reset threshold (simplified)
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
                    let (u_new, spiked) =
                        advance_neuron(o_u[i], o_theta_state[i], o_current[i], t_on);
                    o_u[i] = u_new;
                    if spiked {
                        o_spike_counts[i] += 1.0;
                        o_theta_state[i] = output_theta;
                    }
                }

                // Phase 2: zero current for t_off
                for i in 0..10 {
                    let (u_new, spiked) =
                        advance_neuron(o_u[i], o_theta_state[i], 0.0, t_off);
                    o_u[i] = u_new;
                    if spiked {
                        o_spike_counts[i] += 1.0;
                        o_theta_state[i] = output_theta;
                    }
                }
            } else {
                // No spikes — just leak
                for i in 0..10 {
                    let (u_new, spiked) =
                        advance_neuron(o_u[i], o_theta_state[i], 0.0, DT);
                    o_u[i] = u_new;
                    if spiked {
                        o_spike_counts[i] += 1.0;
                        o_theta_state[i] = output_theta;
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
