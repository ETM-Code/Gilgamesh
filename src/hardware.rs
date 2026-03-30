//! Hardware configuration and mapping for physical deployment.
//!
//! This module provides the bridge between trained (normalized) SNN weights
//! and physical hardware parameters (currents, voltages, capacitances).
//!
//! # Key Concepts
//!
//! - **Normalized training**: Train with dimensionless units (threshold=1.0)
//! - **Hardware mapping**: Convert trained weights to physical currents
//! - **Fixed threshold**: Use 0.8V threshold, scale currents to match
//!
//! # Physical Model (Passive RC Membrane)
//!
//! ```text
//! dV/dt = I_syn/C - V/(R*C) = I_syn/C - V/τ
//! ```
//!
//! Where:
//! - V: membrane voltage above Vref
//! - I_syn: total synaptic current
//! - C: membrane capacitance
//! - R: leak resistance
//! - τ = R*C: membrane time constant

use serde::{Deserialize, Serialize};

/// Physical hardware configuration for neuromorphic deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    // Membrane parameters
    /// Membrane capacitance in Farads (e.g., 10e-9 for 10nF)
    pub c_mem: f32,
    /// Leak resistance in Ohms (e.g., 120e3 for 120kΩ)
    pub r_leak: f32,

    // Voltage parameters
    /// Supply voltage in Volts (e.g., 5.0)
    pub vdd: f32,
    /// Reference voltage in Volts (e.g., 2.5)
    pub vref: f32,
    /// Threshold voltage above Vref in Volts (e.g., 0.8)
    pub v_threshold: f32,

    // Timing
    /// Simulation timestep in seconds (e.g., 1e-6 for 1µs)
    pub dt: f32,

    // Current limits
    /// Maximum synaptic current per synapse in Amps (e.g., 3e-6 for 3µA)
    pub i_syn_max: f32,
    /// Maximum total current into neuron in Amps (e.g., 50e-6 for 50µA)
    pub i_total_max: f32,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            // Membrane: τ = R*C = 120kΩ × 10nF = 1.2ms
            // Updated 2026-03-24 to match final InputSystem PCB schematic
            // (was 33nF / 3.96ms in earlier SPICE model)
            c_mem: 10e-9,  // 10 nF (schematic: C_mem*)
            r_leak: 120e3, // 120 kΩ (schematic: R_leak*)

            // Voltages
            vdd: 5.0,
            vref: 0.0,
            // Threshold from resistor divider: R_top=820kΩ to VDD, R_bottom=150kΩ to Vref
            // θ₀ = (5.0/820k + 2.5/150k) / (1/820k + 1/150k) - 2.5 ≈ 0.387V
            // (was 0.8V estimate)
            v_threshold: 0.387,

            // Timing
            dt: 1e-6, // 1 µs

            // Current limits from R_set=174kΩ synapse mirrors
            // At midscale DAC (2.5V): I = (2.5-0.65)/174k = 10.6µA
            // At full scale (5.0V): I = (5.0-0.65)/174k = 25.0µA
            i_syn_max: 25e-6,    // 25 µA per synapse at full-scale DAC
            i_total_max: 225e-6, // 9 synapses × 25µA max
        }
    }
}

impl HardwareConfig {
    /// Create a new hardware configuration with custom parameters.
    pub fn new(c_mem: f32, r_leak: f32, v_threshold: f32) -> Self {
        Self {
            c_mem,
            r_leak,
            v_threshold,
            ..Default::default()
        }
    }

    /// Membrane time constant τ = R * C
    pub fn tau_m(&self) -> f32 {
        self.r_leak * self.c_mem
    }

    /// Decay factor α = exp(-dt/τ)
    pub fn alpha(&self) -> f32 {
        (-self.dt / self.tau_m()).exp()
    }

    /// Current needed to reach threshold at steady-state: I = V_th / R
    pub fn i_threshold(&self) -> f32 {
        self.v_threshold / self.r_leak
    }

    /// Voltage change per timestep for a given current: dV = (I * dt) / C
    pub fn dv_per_step(&self, current: f32) -> f32 {
        (current * self.dt) / self.c_mem
    }

    /// Current needed for a given voltage change per timestep: I = (dV * C) / dt
    pub fn current_for_dv(&self, dv: f32) -> f32 {
        (dv * self.c_mem) / self.dt
    }

    /// Calculate the current gain needed to map normalized input to physical current.
    ///
    /// For normalized training: threshold = 1.0, input adds directly to membrane
    /// For hardware: threshold = v_threshold, need I_syn such that dV reaches threshold
    ///
    /// To reach threshold in approximately τ/4 time with maximum input:
    /// I_max = C * V_th / (τ/4) = 4 * C * V_th / τ = 4 * V_th / R
    pub fn compute_current_gain(&self) -> f32 {
        // Target: reach threshold in τ/4 with normalized input of 1.0
        let t_target = self.tau_m() / 4.0;
        let i_max = self.c_mem * self.v_threshold / t_target;

        // The gain converts normalized weight output to physical current
        // normalized_input * current_gain = I_syn (in Amps)
        i_max
    }

    /// Convert normalized trained weights to physical synaptic currents.
    ///
    /// # Arguments
    /// * `normalized_weight` - Weight from trained network (typically -1 to 1 range)
    /// * `current_gain` - Gain factor (use compute_current_gain() or custom value)
    ///
    /// # Returns
    /// Physical current in Amps
    pub fn weight_to_current(&self, normalized_weight: f32, current_gain: f32) -> f32 {
        let current = normalized_weight * current_gain;
        // Clamp to hardware limits
        current.clamp(-self.i_syn_max, self.i_syn_max)
    }

    /// Convert normalized input to membrane voltage increment.
    ///
    /// This is what the physics-mode neuron should use for proper SPICE matching.
    ///
    /// # Arguments
    /// * `normalized_input` - Input from previous layer (weighted sum)
    /// * `current_gain` - Current gain factor
    ///
    /// # Returns
    /// Voltage increment to add to membrane
    pub fn input_to_dv(&self, normalized_input: f32, current_gain: f32) -> f32 {
        let current = normalized_input * current_gain;
        self.dv_per_step(current)
    }

    /// Print hardware configuration summary.
    pub fn print_summary(&self) {
        println!("=== Hardware Configuration ===");
        println!("Membrane:");
        println!("  C_mem      = {:.1} nF", self.c_mem * 1e9);
        println!("  R_leak     = {:.1} kΩ", self.r_leak / 1e3);
        println!("  τ_m        = {:.3} ms", self.tau_m() * 1e3);
        println!("  α (decay)  = {:.6}", self.alpha());
        println!();
        println!("Voltages:");
        println!("  Vdd        = {:.1} V", self.vdd);
        println!("  Vref       = {:.1} V", self.vref);
        println!("  V_th       = {:.2} V (above Vref)", self.v_threshold);
        println!();
        println!("Currents:");
        println!(
            "  I_threshold = {:.2} µA (steady-state)",
            self.i_threshold() * 1e6
        );
        println!(
            "  I_syn_max   = {:.1} µA (per synapse)",
            self.i_syn_max * 1e6
        );
        println!(
            "  I_total_max = {:.1} µA (per neuron)",
            self.i_total_max * 1e6
        );
        println!();
        println!("Timing:");
        println!("  dt         = {:.1} µs", self.dt * 1e6);
        println!();
        let gain = self.compute_current_gain();
        println!("Mapping:");
        println!(
            "  Current gain = {:.2} µA (for threshold in τ/4)",
            gain * 1e6
        );
        println!("  dV/step at gain = {:.4} V", self.dv_per_step(gain));
    }
}

/// Hardware-aware layer wrapper that applies current gain scaling.
#[derive(Debug, Clone)]
pub struct HardwareMapping {
    /// Physical hardware configuration
    pub config: HardwareConfig,
    /// Current gain factor (normalized input → Amps)
    pub current_gain: f32,
}

impl HardwareMapping {
    /// Create a new hardware mapping with default current gain.
    pub fn new(config: HardwareConfig) -> Self {
        let current_gain = config.compute_current_gain();
        Self {
            config,
            current_gain,
        }
    }

    /// Create with custom current gain.
    pub fn with_gain(config: HardwareConfig, current_gain: f32) -> Self {
        Self {
            config,
            current_gain,
        }
    }

    /// Convert normalized layer output to physical voltage increment.
    pub fn apply(&self, normalized_input: f32) -> f32 {
        self.config.input_to_dv(normalized_input, self.current_gain)
    }

    /// Convert a batch of normalized inputs to voltage increments.
    pub fn apply_batch(&self, inputs: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        inputs.mapv(|x| self.apply(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = HardwareConfig::default();

        // τ = R * C = 120kΩ * 10nF = 1.2ms (updated from 33nF/3.96ms)
        assert!(
            (cfg.tau_m() - 1.2e-3).abs() < 1e-6,
            "τ_m should be 1.2ms, got {:.4}ms",
            cfg.tau_m() * 1e3
        );

        // I_threshold = V_th / R = 0.387V / 120kΩ ≈ 3.23µA
        assert!(
            (cfg.i_threshold() - 3.23e-6).abs() < 0.1e-6,
            "I_threshold should be ~3.23µA, got {:.2}µA",
            cfg.i_threshold() * 1e6
        );
    }

    #[test]
    fn test_current_gain() {
        let cfg = HardwareConfig::default();
        let gain = cfg.compute_current_gain();

        // With gain applied, reaching threshold in τ/4 = 0.3ms
        // I = C * V_th / t = 10nF * 0.387V / 0.3ms ≈ 12.9µA
        let expected = cfg.c_mem * cfg.v_threshold / (cfg.tau_m() / 4.0);
        assert!(
            (gain - expected).abs() < 1e-6,
            "Current gain should be {:.1}µA, got {:.1}µA",
            expected * 1e6,
            gain * 1e6
        );
    }

    #[test]
    fn test_dv_per_step() {
        let cfg = HardwareConfig::default();

        // With 1µA for 1µs into 10nF: dV = (1µA * 1µs) / 10nF = 0.1mV
        let dv = cfg.dv_per_step(1e-6);
        assert!(
            (dv - 0.1e-3).abs() < 1e-9,
            "dV should be 0.1mV, got {:.4}mV",
            dv * 1e3
        );
    }
}
