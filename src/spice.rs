//! SPICE netlist generation and comparison for hardware validation
//!
//! Generates ngspice-compatible netlists from gilgamesh networks and
//! compares simulation results for physics accuracy validation.
//!
//! This implementation matches the detailed pulse-stretch LIF neuron circuit
//! from the Python reference implementation, including:
//! - Vref buffer (OPA604-like follower)
//! - TIA with finite DC gain and compensation
//! - Physical hysteresis divider for threshold
//! - Adaptive threshold injection with diode
//! - Soft comparator with RC shaping
//! - Pulse stretching circuit (diode + RC for extended spike duration)
//! - Reset path with MUX switch model
//! - Optional analog output stage (inverting amplifier)

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::Command;

use crate::network::{Network, SimulationTrace};

/// Supply voltage configuration
#[derive(Clone, Debug)]
pub struct SupplyConfig {
    pub vdd: f32,
    pub vref: f32,
}

impl Default for SupplyConfig {
    fn default() -> Self {
        Self {
            vdd: 5.0,
            vref: 2.5,
        }
    }
}

/// Membrane configuration
#[derive(Clone, Debug)]
pub struct MembraneConfig {
    pub c_mem: f32,    // Farads
    pub r_leak: f32,   // Ohms
}

impl Default for MembraneConfig {
    fn default() -> Self {
        Self {
            c_mem: 10e-9,     // 10nF
            r_leak: 120e3,    // 120kΩ -> tau = 1.2ms
        }
    }
}

/// Threshold configuration with hysteresis
#[derive(Clone, Debug)]
pub struct ThresholdConfig {
    pub over_vref: f32,      // Threshold above Vref (V)
    pub hysteresis: f32,     // Hysteresis window (V)
    pub c_adapt: f32,        // Adaptive threshold capacitor (F)
    pub divider_scale: f32,  // Scale for Rvh/Rvl/Rf
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            over_vref: 0.8,
            hysteresis: 0.05,
            c_adapt: 22e-9,      // 22nF
            divider_scale: 5.0,
        }
    }
}

/// Reset path configuration
#[derive(Clone, Debug)]
pub struct ResetConfig {
    pub enable: bool,
    pub series_r: f32,      // Series resistance (Ohms)
    pub mux_ron: f32,       // Switch on resistance
    pub mux_roff: f32,      // Switch off resistance
    pub mux_coff: f32,      // Switch off capacitance
    pub switch_vt: f32,     // Switch threshold voltage
    pub switch_vh: f32,     // Switch hysteresis
}

impl Default for ResetConfig {
    fn default() -> Self {
        Self {
            enable: true,
            series_r: 200.0,
            mux_ron: 10.0,
            mux_roff: 1e9,
            mux_coff: 7e-12,
            switch_vt: 2.5,
            switch_vh: 0.1,
        }
    }
}

/// Comparator configuration
#[derive(Clone, Debug)]
pub struct ComparatorConfig {
    pub offset: f32,         // Input offset (V)
    pub prop_delay: f32,     // Propagation delay (s)
    pub vlow: f32,           // Output low voltage
    pub vhigh: f32,          // Output high voltage
}

impl Default for ComparatorConfig {
    fn default() -> Self {
        Self {
            offset: 0.003,
            prop_delay: 40e-9,   // 40ns
            vlow: 0.0,
            vhigh: 5.0,
        }
    }
}

/// Pulse stretching configuration
#[derive(Clone, Debug)]
pub struct PulseStretchConfig {
    pub enable: bool,
    pub r_pw: f32,    // Pulse width resistor (Ohms)
    pub c_pw: f32,    // Pulse width capacitor (F)
}

impl Default for PulseStretchConfig {
    fn default() -> Self {
        Self {
            enable: true,
            r_pw: 100e3,     // 100kΩ
            c_pw: 100e-9,    // 100nF -> tau = 10ms
        }
    }
}

/// Analog output stage configuration
#[derive(Clone, Debug)]
pub struct AnalogOutConfig {
    pub enable: bool,
    pub gain: f32,
    pub sign: i32,           // +1 or -1
    pub clamp_to_rails: bool,
    pub load_r: f32,         // Output load resistance
    pub inverting: bool,
    pub r1: f32,             // Input/ground resistor
    pub r2: f32,             // Feedback resistor
}

impl Default for AnalogOutConfig {
    fn default() -> Self {
        Self {
            enable: false,   // Disabled by default
            gain: 2.0,
            sign: 1,
            clamp_to_rails: false,
            load_r: 10e3,
            inverting: true,
            r1: 10e3,
            r2: 40e3,
        }
    }
}

/// Bias currents configuration
#[derive(Clone, Debug)]
pub struct BiasCurrents {
    pub tia: f32,
    pub comparator: f32,
    pub analog_out: f32,
}

impl Default for BiasCurrents {
    fn default() -> Self {
        Self {
            tia: 20e-6,
            comparator: 2e-6,
            analog_out: 10e-6,
        }
    }
}

/// Complete physics parameters for SPICE circuit generation
#[derive(Clone, Debug)]
pub struct SpiceParams {
    pub supply: SupplyConfig,
    pub membrane: MembraneConfig,
    pub threshold: ThresholdConfig,
    pub reset: ResetConfig,
    pub comparator: ComparatorConfig,
    pub pulse_stretch: PulseStretchConfig,
    pub analog_out: AnalogOutConfig,
    pub bias: BiasCurrents,
    /// Integration timestep (seconds)
    pub dt: f32,
}

impl Default for SpiceParams {
    fn default() -> Self {
        Self {
            supply: SupplyConfig::default(),
            membrane: MembraneConfig::default(),
            threshold: ThresholdConfig::default(),
            reset: ResetConfig::default(),
            comparator: ComparatorConfig::default(),
            pulse_stretch: PulseStretchConfig::default(),
            analog_out: AnalogOutConfig::default(),
            bias: BiasCurrents::default(),
            dt: 1e-6,  // 1μs timestep
        }
    }
}

impl SpiceParams {
    /// Create from network (extracts physics parameters where available)
    pub fn from_network(net: &Network) -> Self {
        let mut params = Self::default();

        // Extract tau_m from network if available
        if let Some(tau_m) = net.lif1.mode.tau_m() {
            // tau = R * C, so R = tau / C
            params.membrane.r_leak = tau_m / params.membrane.c_mem;
        }

        // Use network threshold (normalized to voltage)
        // The network threshold is typically 1.0, which maps to over_vref
        params.threshold.over_vref = net.lif1.threshold * 0.8; // Scale to reasonable voltage

        params
    }

    /// Create with analog output enabled
    pub fn with_analog_output(mut self, enable: bool) -> Self {
        self.analog_out.enable = enable;
        self
    }

    /// Create with pulse stretching enabled/disabled
    pub fn with_pulse_stretch(mut self, enable: bool) -> Self {
        self.pulse_stretch.enable = enable;
        self
    }

    /// Membrane time constant
    pub fn tau_m(&self) -> f32 {
        self.membrane.c_mem * self.membrane.r_leak
    }

    /// Pulse stretch time constant
    pub fn tau_pulse(&self) -> f32 {
        self.pulse_stretch.c_pw * self.pulse_stretch.r_pw
    }
}

/// SPICE netlist generator
pub struct SpiceNetlist {
    /// Netlist content
    content: String,
}

impl SpiceNetlist {
    /// Generate netlist from network and input
    ///
    /// The subcircuit uses nodes: mem vref vdd comp_pulse analog_out sum
    /// - Input currents flow into the `sum` node
    /// - `comp_pulse` provides the stretched spike output for downstream neurons
    /// - `analog_out` provides membrane voltage (if enabled)
    pub fn from_network(
        net: &Network,
        input: &[f32],
        num_steps: usize,
        params: &SpiceParams,
    ) -> Self {
        let input_size = net.fc1.in_features;
        let hidden_size = net.fc1.out_features;
        let output_size = net.fc2.out_features;

        // Simulation time in seconds
        let sim_time = num_steps as f32 * params.dt * 1000.0; // Convert to ms-scale
        let sim_step = params.dt * 100.0; // Finer resolution

        let mut content = String::new();

        // Header with circuit info
        content.push_str("* ================================================================\n");
        content.push_str("* gilgamesh SNN - SPICE Simulation (Detailed Pulse-Stretch Model)\n");
        content.push_str("* ================================================================\n");
        content.push_str(&format!("* Architecture: {} -> {} -> {}\n", input_size, hidden_size, output_size));
        content.push_str(&format!("* Simulation time: {:.2}ms\n", sim_time * 1000.0));
        content.push_str(&format!("* Membrane: C={:.3e}F, R={:.3e}Ω, tau={:.2}ms\n",
            params.membrane.c_mem, params.membrane.r_leak, params.tau_m() * 1000.0));
        content.push_str(&format!("* Pulse stretch: tau={:.2}ms (enabled={})\n",
            params.tau_pulse() * 1000.0, params.pulse_stretch.enable));
        content.push_str(&format!("* Threshold: Vref+{:.2}V, hysteresis={:.3}V\n",
            params.threshold.over_vref, params.threshold.hysteresis));
        content.push_str(&format!("* Analog output: enabled={}\n", params.analog_out.enable));
        content.push_str("\n");

        // LIF neuron subcircuit definition
        content.push_str(&Self::lif_subcircuit(params));

        // ========== Power supplies ==========
        content.push_str("* ========== Power Supplies ==========\n");
        content.push_str(&format!("Vdd vdd 0 DC {}\n", params.supply.vdd));
        content.push_str(&format!("Vref vref 0 DC {}\n", params.supply.vref));
        content.push_str("\n");

        // ========== Input voltage sources ==========
        content.push_str("* ========== Input Voltage Sources ==========\n");
        content.push_str("* Inputs scaled to voltage around Vref\n");
        for (i, &val) in input.iter().enumerate() {
            // Scale input to reasonable voltage swing around Vref
            // Input values are typically normalized, scale to ~0.5V swing
            let voltage = params.supply.vref + val.clamp(-1.0, 1.0) * 0.5;
            content.push_str(&format!("Vin_{} in_{} 0 DC {:.6}\n", i, i, voltage));
        }
        content.push_str("\n");

        // ========== Hidden layer neurons ==========
        // Subcircuit: mem vref vdd comp_pulse analog_out sum
        content.push_str("* ========== Hidden Layer Neurons ==========\n");
        for h in 0..hidden_size {
            content.push_str(&format!(
                "Xh_{} mem_h_{} vref vdd pulse_h_{} ana_h_{} sum_h_{} lif_neuron\n",
                h, h, h, h, h
            ));
        }
        content.push_str("\n");

        // ========== Input to hidden synapses (VCCS) ==========
        content.push_str("* ========== Input -> Hidden Synapses (VCCS) ==========\n");
        content.push_str("* G<name> n+ n- nc+ nc- transconductance\n");
        content.push_str("* Current from n+ to n- = transconductance * (V(nc+) - V(nc-))\n");
        for h in 0..hidden_size {
            for i in 0..input_size {
                let weight = net.fc1.weight[[i, h]];
                if weight.abs() > 1e-6 {
                    // Scale weight to appropriate current (nA to uA range)
                    let scaled_weight = weight * 1e-6; // Convert to microamps/volt
                    content.push_str(&format!(
                        "Gw1_{}_{} sum_h_{} 0 in_{} vref {:.6e}\n",
                        i, h, h, i, scaled_weight
                    ));
                }
            }
        }

        // Bias currents for hidden layer
        if let Some(ref bias) = net.fc1.bias {
            content.push_str("\n* Hidden layer bias currents\n");
            for h in 0..hidden_size {
                let b = bias[h];
                if b.abs() > 1e-6 {
                    let scaled_bias = b * 1e-6;
                    content.push_str(&format!("Ib1_{} sum_h_{} 0 DC {:.6e}\n", h, h, scaled_bias));
                }
            }
        }
        content.push_str("\n");

        // ========== Output layer neurons ==========
        content.push_str("* ========== Output Layer Neurons ==========\n");
        for o in 0..output_size {
            content.push_str(&format!(
                "Xo_{} mem_o_{} vref vdd pulse_o_{} ana_o_{} sum_o_{} lif_neuron\n",
                o, o, o, o, o
            ));
        }
        content.push_str("\n");

        // ========== Hidden to output synapses ==========
        content.push_str("* ========== Hidden -> Output Synapses (VCCS) ==========\n");
        content.push_str("* Uses stretched pulse output (pulse_h_*) for better charge transfer\n");
        for o in 0..output_size {
            for h in 0..hidden_size {
                let weight = net.fc2.weight[[h, o]];
                if weight.abs() > 1e-6 {
                    let scaled_weight = weight * 1e-6;
                    content.push_str(&format!(
                        "Gw2_{}_{} sum_o_{} 0 pulse_h_{} 0 {:.6e}\n",
                        h, o, o, h, scaled_weight
                    ));
                }
            }
        }

        // Bias currents for output layer
        if let Some(ref bias) = net.fc2.bias {
            content.push_str("\n* Output layer bias currents\n");
            for o in 0..output_size {
                let b = bias[o];
                if b.abs() > 1e-6 {
                    let scaled_bias = b * 1e-6;
                    content.push_str(&format!("Ib2_{} sum_o_{} 0 DC {:.6e}\n", o, o, scaled_bias));
                }
            }
        }
        content.push_str("\n");

        // ========== Simulation commands ==========
        content.push_str("* ========== Simulation ==========\n");
        content.push_str(&format!(".tran {:.6e} {:.6e}\n", sim_step, sim_time));
        content.push_str("\n");

        // ========== Control section ==========
        content.push_str(".control\n");
        content.push_str("run\n");
        content.push_str("set filetype=ascii\n");
        content.push_str("wrdata gilgamesh_output.txt");

        // Save output pulse voltages (for spike counting)
        for o in 0..output_size {
            content.push_str(&format!(" v(pulse_o_{})", o));
        }
        // Save output membrane voltages
        for o in 0..output_size {
            content.push_str(&format!(" v(mem_o_{})", o));
        }
        // Save some hidden layer data for debugging
        let sample_hidden = hidden_size.min(5);
        for h in 0..sample_hidden {
            content.push_str(&format!(" v(pulse_h_{}) v(mem_h_{})", h, h));
        }
        content.push_str("\n");
        content.push_str("quit\n");
        content.push_str(".endc\n");
        content.push_str("\n");

        content.push_str(".end\n");

        Self { content }
    }

    /// Generate detailed LIF neuron subcircuit matching pulse-stretch reference
    ///
    /// Nodes:
    /// - mem: membrane voltage output
    /// - vref: reference voltage input
    /// - vdd: supply voltage
    /// - comp_pulse: stretched spike output (for downstream neurons)
    /// - analog_out: analog membrane output (optional)
    /// - sum: summing node for input currents
    fn lif_subcircuit(params: &SpiceParams) -> String {
        let mut s = String::new();

        s.push_str("* ========== LIF Neuron Detailed Subcircuit (Pulse Stretching) ==========\n");
        s.push_str("* Matches Python reference: lif_neuron_generator_pulse_stretch.py\n");
        s.push_str("* Nodes: mem vref vdd comp_pulse analog_out sum\n");
        s.push_str(".subckt lif_neuron mem vref vdd comp_pulse analog_out sum\n");
        s.push_str("\n");

        // ========== Vref Buffer (OPA604-like follower) ==========
        s.push_str("* ---------- Vref buffer (OPA604-like) ----------\n");
        s.push_str("Ebuf vref_buf 0 vref 0 1e5\n");
        s.push_str("Rbuf vref_buf vref 1k\n");
        s.push_str("Cbuf vref 0 80p\n");
        s.push_str("Rvr vref vref_buf 1m\n");
        s.push_str("\n");

        // ========== TIA with finite DC gain and compensation ==========
        s.push_str("* ---------- TIA op-amp with finite gain and compensation ----------\n");
        s.push_str("Eint mem 0 sum vref 2e5\n");
        s.push_str("Rout_int mem 0 20\n");
        s.push_str("Cint mem 0 5p\n");
        s.push_str(&format!("Cmem mem sum {:.3e}\n", params.membrane.c_mem));
        s.push_str(&format!("Rleak mem sum {:.3e}\n", params.membrane.r_leak));
        s.push_str(&format!("Iint_bias vdd 0 {:.3e}\n", params.bias.tia));
        s.push_str("\n");

        // ========== Physical hysteresis divider ==========
        let vhi = params.comparator.vhigh;
        let vlo = params.comparator.vlow;
        let dv_out = (vhi - vlo).max(1e-6);
        let beta = params.threshold.hysteresis / dv_out;

        s.push_str("* ---------- Threshold divider (places threshold near Vref + over_vref) ----------\n");
        let scale = params.threshold.divider_scale.max(1e-3);
        let rvh = 681e3 * scale;
        let rvl = 316e3 * scale;
        s.push_str(&format!("Rvh vdd vth_node {:.3e}\n", rvh));
        s.push_str(&format!("Rvl vth_node vref {:.3e}\n", rvl));

        // Hysteresis feedback resistor
        let g_div = (1.0 / rvh) + (1.0 / rvl);
        let rf = 1.0 / (beta * g_div / (1.0 - beta).max(1e-6));
        let rf_ohm = rf.max(100e3);
        s.push_str(&format!("Rf comp_out vth_node {:.3e}\n", rf_ohm));
        s.push_str("\n");

        // ========== Adaptive threshold injection ==========
        s.push_str("* ---------- Adaptive threshold injection ----------\n");
        s.push_str(&format!("Cadapt vth_node vref {:.3e}\n", params.threshold.c_adapt));
        s.push_str(".model DADAPT D(Is=1e-6 N=1.05 Rs=2 Cjo=1p Eg=0.69)\n");
        s.push_str("Rinj comp_out ninj 2.2Meg\n");
        s.push_str("Dinj ninj vth_node DADAPT\n");
        s.push_str("\n");

        // ========== Soft comparator with RC shaping ==========
        s.push_str("* ---------- Soft comparator (deflection space) ----------\n");
        s.push_str(&format!(".param VLO={}\n", vlo));
        s.push_str(&format!(".param VHI={}\n", vhi));
        s.push_str(".param VSW=0.01\n");

        // RC shaping from propagation delay
        let rc = (params.comparator.prop_delay / 10.0).max(1e-9);
        let rout = (params.comparator.prop_delay / rc).max(10.0);

        s.push_str("Bdef vdef 0 V = V(vref) - V(mem)\n");
        s.push_str("Btheta_rel vtheta_rel 0 V = V(vth_node) - V(vref)\n");
        s.push_str(&format!(
            "Bcomp comp_raw 0 V = VLO + (VHI - VLO)*(0.5*(1 + tanh( ( V(vdef) - ( {} + {} ) ) / VSW )))\n",
            params.threshold.over_vref, params.comparator.offset
        ));
        s.push_str(&format!("Rcout comp_raw comp_out {:.3e}\n", rout));
        s.push_str(&format!("Ccout comp_out 0 {:.3e}\n", rc));
        s.push_str(&format!("Icomp_bias vdd 0 {:.3e}\n", params.bias.comparator));
        s.push_str("\n");

        // ========== Pulse stretching circuit ==========
        if params.pulse_stretch.enable {
            let tau_pulse = params.pulse_stretch.r_pw * params.pulse_stretch.c_pw;
            s.push_str("* ---------- Pulse stretching circuit ----------\n");
            s.push_str(&format!(
                "* tau_pulse = {:.3e}Ω × {:.3e}F = {:.2}ms\n",
                params.pulse_stretch.r_pw, params.pulse_stretch.c_pw, tau_pulse * 1e3
            ));
            s.push_str("* Diode allows fast charging, slow discharge through Rpw\n");
            s.push_str(".model DPW D(Is=1e-12 N=1.05 Rs=10 Cjo=1p)\n");
            s.push_str("Dpw comp_out comp_pulse DPW\n");
            s.push_str(&format!("Rpw comp_pulse 0 {:.3e}\n", params.pulse_stretch.r_pw));
            s.push_str(&format!("Cpw comp_pulse 0 {:.3e}\n", params.pulse_stretch.c_pw));
        } else {
            s.push_str("* ---------- Pulse stretching disabled - direct connection ----------\n");
            s.push_str("Rpw_bypass comp_out comp_pulse 1\n");
        }
        s.push_str("\n");

        // ========== Reset path ==========
        if params.reset.enable {
            s.push_str("* ---------- Reset path ----------\n");
            s.push_str(&format!("Rreset mem reset_node {:.3e}\n", params.reset.series_r));
            s.push_str(&format!("Coff_reset reset_node vref {:.3e}\n", params.reset.mux_coff));
            s.push_str("Sreset reset_node vref comp_out 0 SWMUX\n");
            s.push_str(&format!(
                ".model SWMUX SW(Ron={} Roff={:.3e} Vt={} Vh={})\n",
                params.reset.mux_ron, params.reset.mux_roff,
                params.reset.switch_vt, params.reset.switch_vh
            ));
        }
        s.push_str("\n");

        // ========== Analog output stage (optional) ==========
        if params.analog_out.enable {
            s.push_str("* ---------- Analog output stage (inverting amplifier) ----------\n");

            if params.analog_out.inverting {
                // Inverting op-amp: gain = -R2/R1
                let desired_gain = params.analog_out.gain.abs();
                let r1 = params.analog_out.r1;
                let r2 = r1 * desired_gain;

                // Differential input
                if params.analog_out.sign >= 0 {
                    s.push_str("Bdiff ana_in 0 V = V(mem) - V(vref)\n");
                } else {
                    s.push_str("Bdiff ana_in 0 V = V(vref) - V(mem)\n");
                }

                // Inverting amplifier op-amp model
                s.push_str("Eana_opamp ana_opamp_out 0 0 ana_inv_in 1e5\n");
                s.push_str("Rana_int ana_opamp_out analog_out 50\n");
                s.push_str("Cana_comp analog_out 0 2p\n");

                // Inverting input network
                s.push_str(&format!("R1_ana ana_in ana_inv_in {:.3e}\n", r1));
                s.push_str(&format!("R2_ana analog_out ana_inv_in {:.3e}\n", r2));

                // Output load
                s.push_str(&format!("Rana_load analog_out 0 {:.3e}\n", params.analog_out.load_r));
            } else {
                // Non-inverting: gain = 1 + R2/R1
                let desired_gain = params.analog_out.gain.abs();
                let r1 = params.analog_out.r1;
                let r2 = r1 * (desired_gain - 1.0).max(0.0);

                if params.analog_out.sign >= 0 {
                    s.push_str("Bdiff ana_in 0 V = V(mem) - V(vref)\n");
                } else {
                    s.push_str("Bdiff ana_in 0 V = V(vref) - V(mem)\n");
                }

                s.push_str("Eana_opamp ana_opamp_out 0 ana_in ana_fb 1e5\n");
                s.push_str("Rana_int ana_opamp_out analog_out 50\n");
                s.push_str("Cana_comp analog_out 0 2p\n");
                s.push_str(&format!("R2_ana analog_out ana_fb {:.3e}\n", r2));
                s.push_str(&format!("R1_ana ana_fb 0 {:.3e}\n", r1));
                s.push_str(&format!("Rana_load analog_out 0 {:.3e}\n", params.analog_out.load_r));
            }

            if params.analog_out.clamp_to_rails {
                s.push_str("* Rail clamps\n");
                s.push_str(".model DCLAMP D(Is=1e-15 N=1.8 Rs=1)\n");
                s.push_str("Dcl_lo 0 analog_out DCLAMP\n");
                s.push_str("Dcl_hi analog_out vdd DCLAMP\n");
            }

            s.push_str(&format!("Iana_bias vdd 0 {:.3e}\n", params.bias.analog_out));
        } else {
            // Analog output disabled - tie to vref
            s.push_str("* ---------- Analog output disabled ----------\n");
            s.push_str("Rana_tie analog_out vref 1Meg\n");
        }
        s.push_str("\n");

        s.push_str(".ends lif_neuron\n");
        s.push_str("\n");

        s
    }

    /// Write netlist to file
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = fs::File::create(path.as_ref())
            .with_context(|| format!("Failed to create netlist file: {:?}", path.as_ref()))?;

        file.write_all(self.content.as_bytes())
            .with_context(|| "Failed to write netlist")?;

        Ok(())
    }

    /// Get netlist content
    pub fn content(&self) -> &str {
        &self.content
    }
}

/// Parsed SPICE simulation output
#[derive(Clone, Debug)]
pub struct SpiceOutput {
    /// Time points (seconds)
    pub time: Vec<f32>,
    /// Node voltages: HashMap<node_name, Vec<voltage_at_each_time>>
    pub voltages: HashMap<String, Vec<f32>>,
}

impl SpiceOutput {
    /// Parse ngspice ASCII raw file
    pub fn parse_raw<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = fs::File::open(path.as_ref())
            .with_context(|| format!("Failed to open raw file: {:?}", path.as_ref()))?;

        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut num_variables = 0;
        let mut num_points = 0;
        let mut variable_names: Vec<String> = Vec::new();
        let mut in_data = false;

        // Parse header
        while let Some(line) = lines.next() {
            let line = line?;
            let line = line.trim();

            if line.starts_with("No. Variables:") {
                num_variables = line.split(':').nth(1)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("No. Points:") {
                num_points = line.split(':').nth(1)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("Variables:") {
                // Read variable definitions
                for _ in 0..num_variables {
                    if let Some(var_line) = lines.next() {
                        let var_line = var_line?;
                        let parts: Vec<&str> = var_line.trim().split_whitespace().collect();
                        if parts.len() >= 2 {
                            variable_names.push(parts[1].to_string());
                        }
                    }
                }
            } else if line == "Values:" {
                in_data = true;
                break;
            }
        }

        if !in_data || num_variables == 0 || num_points == 0 {
            anyhow::bail!("Invalid raw file format");
        }

        // Initialize data storage
        let mut data: Vec<Vec<f32>> = vec![Vec::with_capacity(num_points); num_variables];

        // Parse data values
        let mut current_var = 0;

        for line in lines {
            let line = line?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            // Each line can have format: "index value" or just "value"
            let value_str = if line.contains('\t') {
                line.split('\t').last().unwrap_or(line)
            } else {
                line.split_whitespace().last().unwrap_or(line)
            };

            if let Ok(value) = value_str.parse::<f32>() {
                if current_var < num_variables {
                    data[current_var].push(value);
                    current_var += 1;

                    if current_var >= num_variables {
                        current_var = 0;
                    }
                }
            }
        }

        // Build output
        let time = if !variable_names.is_empty() && variable_names[0].to_lowercase() == "time" {
            data[0].clone()
        } else {
            Vec::new()
        };

        let mut voltages = HashMap::new();
        for (i, name) in variable_names.iter().enumerate() {
            if i > 0 && i < data.len() {
                voltages.insert(name.clone(), data[i].clone());
            }
        }

        Ok(Self { time, voltages })
    }

    /// Get voltage trace for a node
    pub fn get_voltage(&self, node: &str) -> Option<&Vec<f32>> {
        self.voltages.get(node)
    }
}

/// Run ngspice on a netlist file
pub fn run_ngspice<P1: AsRef<Path>, P2: AsRef<Path>>(netlist_path: P1, output_dir: P2) -> Result<SpiceOutput> {
    let netlist_path = netlist_path.as_ref();
    let output_dir = output_dir.as_ref();

    // Ensure output directory exists
    fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;

    // Run ngspice in batch mode
    let output = Command::new("ngspice")
        .args(["-b", "-r"])
        .arg(output_dir.join("gilgamesh_output.raw"))
        .arg(netlist_path)
        .current_dir(output_dir)
        .output()
        .with_context(|| "Failed to run ngspice. Is it installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ngspice failed: {}", stderr);
    }

    // Parse output
    let raw_path = output_dir.join("gilgamesh_output.raw");
    SpiceOutput::parse_raw(&raw_path)
}

/// Comparison result between gilgamesh and SPICE simulation
#[derive(Clone, Debug)]
pub struct ComparisonResult {
    /// Mean absolute error for output membrane voltages
    pub output_mem_mae: f32,
    /// Output spike counts from gilgamesh
    pub gilgamesh_spikes: Vec<f32>,
    /// Output spike counts from SPICE (estimated from voltage crossings)
    pub spice_spikes: Vec<f32>,
    /// Predicted class from gilgamesh
    pub gilgamesh_prediction: usize,
    /// Predicted class from SPICE
    pub spice_prediction: usize,
    /// Whether predictions match
    pub predictions_match: bool,
}

impl ComparisonResult {
    /// Compare gilgamesh trace with SPICE output
    pub fn compare(
        trace: &SimulationTrace,
        spice: &SpiceOutput,
        params: &SpiceParams,
    ) -> Self {
        let output_size = trace.output_spike_count.shape()[1];

        // Get gilgamesh spike counts (first batch element)
        let gilgamesh_spikes: Vec<f32> = trace.output_spike_count.row(0).to_vec();

        // Estimate SPICE spike counts from voltage threshold crossings
        let mut spice_spikes = vec![0.0f32; output_size];

        // Threshold for pulse detection (halfway between vlow and vhigh)
        let pulse_threshold = (params.comparator.vlow + params.comparator.vhigh) / 2.0;

        for o in 0..output_size {
            // Try new node naming first, then fall back to old
            let node_name = format!("v(pulse_o_{})", o);
            let alt_node_name = format!("v(out_o_{})", o);

            let voltages = spice.get_voltage(&node_name)
                .or_else(|| spice.get_voltage(&alt_node_name));

            if let Some(voltages) = voltages {
                // Count threshold crossings (rising edges)
                let mut prev_above = false;
                for &v in voltages {
                    let above = v > pulse_threshold;
                    if above && !prev_above {
                        spice_spikes[o] += 1.0;
                    }
                    prev_above = above;
                }
            }
        }

        // Compute predictions
        let gilgamesh_prediction = gilgamesh_spikes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let spice_prediction = spice_spikes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Compute MAE (simplified - comparing spike counts)
        let output_mem_mae: f32 = gilgamesh_spikes
            .iter()
            .zip(spice_spikes.iter())
            .map(|(g, s)| (g - s).abs())
            .sum::<f32>() / output_size as f32;

        Self {
            output_mem_mae,
            gilgamesh_spikes,
            spice_spikes,
            gilgamesh_prediction,
            spice_prediction,
            predictions_match: gilgamesh_prediction == spice_prediction,
        }
    }

    /// Print comparison summary
    pub fn print_summary(&self) {
        println!("=== SPICE Comparison Results ===");
        println!();
        println!("Output Spike Counts:");
        println!("{:>6} {:>12} {:>12}", "Class", "Gilgamesh", "SPICE");
        println!("{:-<32}", "");
        for (i, (g, s)) in self.gilgamesh_spikes.iter().zip(self.spice_spikes.iter()).enumerate() {
            let marker = if i == self.gilgamesh_prediction || i == self.spice_prediction {
                if i == self.gilgamesh_prediction && i == self.spice_prediction {
                    " <-- both"
                } else if i == self.gilgamesh_prediction {
                    " <-- gil"
                } else {
                    " <-- spice"
                }
            } else {
                ""
            };
            println!("{:>6} {:>12.1} {:>12.1}{}", i, g, s, marker);
        }
        println!();
        println!("Gilgamesh prediction: {}", self.gilgamesh_prediction);
        println!("SPICE prediction:     {}", self.spice_prediction);
        println!("Predictions match:    {}", if self.predictions_match { "YES" } else { "NO" });
        println!("Mean absolute error:  {:.2} spikes", self.output_mem_mae);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spice_params_default() {
        let params = SpiceParams::default();
        // Default membrane: 10nF, 120kΩ -> tau = 1.2ms
        let tau_m = params.tau_m();
        assert!((tau_m - 1.2e-3).abs() < 1e-4, "tau_m should be ~1.2ms, got {}", tau_m);

        // Default pulse stretch: 100kΩ, 100nF -> tau = 10ms
        let tau_pulse = params.tau_pulse();
        assert!((tau_pulse - 10e-3).abs() < 1e-4, "tau_pulse should be ~10ms, got {}", tau_pulse);

        // Threshold
        assert!((params.threshold.over_vref - 0.8).abs() < 0.01);
        assert!((params.threshold.hysteresis - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_supply_config() {
        let params = SpiceParams::default();
        assert!((params.supply.vdd - 5.0).abs() < 0.01);
        assert!((params.supply.vref - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_analog_output_toggle() {
        let params = SpiceParams::default();
        assert!(!params.analog_out.enable, "Analog output should be disabled by default");

        let params_with_analog = params.with_analog_output(true);
        assert!(params_with_analog.analog_out.enable);
    }

    #[test]
    fn test_pulse_stretch_toggle() {
        let params = SpiceParams::default();
        assert!(params.pulse_stretch.enable, "Pulse stretch should be enabled by default");

        let params_no_stretch = params.with_pulse_stretch(false);
        assert!(!params_no_stretch.pulse_stretch.enable);
    }

    #[test]
    fn test_lif_subcircuit_generation() {
        let params = SpiceParams::default();
        let subckt = SpiceNetlist::lif_subcircuit(&params);

        // Check subcircuit header
        assert!(subckt.contains(".subckt lif_neuron"));
        assert!(subckt.contains(".ends lif_neuron"));

        // Check key components are present
        assert!(subckt.contains("Vref buffer"));
        assert!(subckt.contains("TIA op-amp"));
        assert!(subckt.contains("Threshold divider"));
        assert!(subckt.contains("Soft comparator"));
        assert!(subckt.contains("Pulse stretching"));
        assert!(subckt.contains("Reset path"));

        // Check analog output is disabled by default
        assert!(subckt.contains("Analog output disabled"));
    }

    #[test]
    fn test_lif_subcircuit_with_analog() {
        let params = SpiceParams::default().with_analog_output(true);
        let subckt = SpiceNetlist::lif_subcircuit(&params);

        // Check analog output stage is present
        assert!(subckt.contains("Analog output stage"));
        assert!(subckt.contains("Eana_opamp"));
    }
}
