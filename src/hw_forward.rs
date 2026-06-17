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
/// As-built **output**-layer R_bottom: the stock 150 kΩ divider bottom in
/// parallel with the surgery's 300 kΩ rework (150k‖300k = 100k), which lands
/// the output comparator threshold at ≈0.543 V. This is the *faithful* output
/// threshold the real board operates at (the weak ~0.43 µA synapse mirror only
/// charges the output membrane to ~0.5 V, so the nominal 1.058 V is unreachable
/// — see `tarski-works/OUTPUT_THRESHOLD_AND_CURRENT_MATH.md`). Used as the
/// [`HwForwardConfig`] default output threshold.
pub const R_BOTTOM_OUTPUT_FAITHFUL: f64 = 100e3;
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

/// Configuration for the hardware forward pass.
///
/// `Default` is the **faithful as-built board** config — the one we want
/// training/evaluation to use so it matches the physical Tarski:
/// - `hidden_theta` at the nominal 1.058 V (220 kΩ R_bottom, reference-faithful,
///   DAC-injected hidden current);
/// - `output_theta` at the **real** ≈0.543 V (150k‖300k surged output divider),
///   *not* the unreachable 1.058 V nominal;
/// - outputs **O2 and O10 masked** (indices 1 and 9) — physically inert on the
///   as-built board, so they emit no spikes;
/// - matched mirrors (`mirror_mismatch_cv = 0`) for determinism.
///
/// Every field is overridable, so nothing the legacy entry points could do is
/// lost: the back-compat [`hw_forward_batch`] / [`hw_forward_batch_with_thresholds`]
/// build a config with no masking and matched mirrors, reproducing the old
/// numbers exactly.
#[derive(Debug, Clone)]
pub struct HwForwardConfig {
    /// Hidden-neuron membrane threshold (V, GND-referenced).
    pub hidden_theta: f64,
    /// Output-neuron membrane threshold (V, GND-referenced).
    pub output_theta: f64,
    /// fc1→DAC headroom scale (1.16 is the per-board optimum).
    pub dac_scale: f64,
    /// Number of 1 ms integration steps.
    pub num_steps: usize,
    /// 0-indexed output neurons forced to zero spikes (physically masked on the
    /// as-built board). Default: O2 and O10 → `[1, 9]`. Set empty once those
    /// outputs are repaired on the bench.
    pub masked_outputs: Vec<usize>,
    /// Coefficient of variation of the per-output synapse-current gain, modelling
    /// BJT current-mirror array mismatch (Gaussian, drawn once per call and held
    /// across the batch — mismatch is fixed per board). `0.0` ⇒ ideal matched
    /// mirrors (deterministic, the legacy behaviour). ~0.02–0.05 is realistic for
    /// the dense-array mirrors and is what compresses the real board's class
    /// margins below gilgamesh's idealised ~82%.
    pub mirror_mismatch_cv: f64,
    /// Seed for the deterministic mismatch draw (only consulted when cv > 0).
    pub mismatch_seed: u64,
}

impl Default for HwForwardConfig {
    fn default() -> Self {
        Self {
            hidden_theta: theta_from_r_bottom(R_BOTTOM_NOMINAL),
            output_theta: theta_from_r_bottom(R_BOTTOM_OUTPUT_FAITHFUL),
            dac_scale: 1.16,
            num_steps: 25,
            masked_outputs: vec![1, 9],
            mirror_mismatch_cv: 0.0,
            mismatch_seed: 0,
        }
    }
}

/// Per-output synapse-current gain vector from the mismatch config. With
/// `cv == 0` every gain is exactly 1.0 (no allocation of randomness, bit-identical
/// to the matched-mirror model). With `cv > 0`, draws 10 Gaussian gains from a
/// deterministic splitmix64 + Box–Muller stream seeded by `mismatch_seed`.
fn mirror_gains(cv: f64, seed: u64) -> [f64; 10] {
    let mut gains = [1.0f64; 10];
    if cv <= 0.0 {
        return gains;
    }
    // splitmix64: deterministic, dependency-free.
    let mut state = seed;
    let mut unit = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / ((1u64 << 53) as f64) // [0,1)
    };
    for g in gains.iter_mut() {
        // Box–Muller; guard u1 away from 0.
        let u1 = unit().max(1e-12);
        let u2 = unit();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        *g = (1.0 + cv * z).max(0.0);
    }
    gains
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
///
/// Back-compat shim: builds a [`HwForwardConfig`] with **no output masking** and
/// **matched mirrors**, reproducing the historical numbers exactly.
pub fn hw_forward_batch_with_thresholds(
    fc1_outputs: &Array2<f32>,
    fc2_quantized: &[Vec<i8>],
    dac_scale: f64,
    num_steps: usize,
    hidden_theta: f64,
    output_theta: f64,
) -> Array2<f32> {
    let cfg = HwForwardConfig {
        hidden_theta,
        output_theta,
        dac_scale,
        num_steps,
        masked_outputs: Vec::new(),
        mirror_mismatch_cv: 0.0,
        mismatch_seed: 0,
    };
    hw_forward_batch_cfg(fc1_outputs, fc2_quantized, &cfg)
}

/// Hardware forward pass driven by a [`HwForwardConfig`]. This is the primary
/// entry point; [`HwForwardConfig::default()`] is the faithful as-built board.
pub fn hw_forward_batch_cfg(
    fc1_outputs: &Array2<f32>,
    fc2_quantized: &[Vec<i8>],
    cfg: &HwForwardConfig,
) -> Array2<f32> {
    let HwForwardConfig {
        hidden_theta,
        output_theta,
        dac_scale,
        num_steps,
        ref masked_outputs,
        mirror_mismatch_cv,
        mismatch_seed,
    } = *cfg;
    let gains = mirror_gains(mirror_mismatch_cv, mismatch_seed);
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
                    // Current-mirror device mismatch (gains[o] == 1.0 when cv == 0).
                    o_current[o] *= gains[o];
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
        // Physically masked outputs emit no readable spikes.
        for &m in masked_outputs {
            if m < 10 {
                spike_counts[[b, m]] = 0.0;
            }
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
