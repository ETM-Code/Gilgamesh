use crate::math_functions::lif::{adapt, relax_towards};
use crate::simulators::config::{
    AnalogOutCfg, ComparatorCfg, Membrane, NetworkConfig, NeuronConfig, SimCfg, Synapse,
    SynapseType,
};
use crate::simulators::constants::{
    COMPARATOR_RAIL_DROP_V, DEFAULT_ANALOG_LOAD_R_OHM, DEFAULT_ANALOG_SERIES_R_OHM,
    DEFAULT_DIODE_DROP_V, DEFAULT_INJECTION_R_OHM, DIVIDER_R_VH_OHM, DIVIDER_R_VL_OHM,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSample {
    pub time_s: f64,
    pub v_mem: f64,
    pub u_deflection: f64,
    pub v_comp: f64,
    pub v_ana: f64,
    pub v_outmix: f64,
    pub theta_over_vref: f64,
    pub v_syn: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct EquivalentNeuron {
    pub membrane: Membrane,
    pub comparator: ComparatorCfg,
    pub threshold: ThresholdModel,
    pub analog: AnalogModel,
    pub supplies: crate::simulators::config::Supplies,
    pub sim: SimCfg,
    pub synapses: Vec<Synapse>,
    pub tau_membrane: f64,
    pub comparator_high: f64,
}

#[derive(Debug, Clone)]
pub struct ThresholdModel {
    pub theta_low: f64,
    pub theta_high: f64,
    pub tau_theta: f64,
    pub c_adapt: f64,
}

#[derive(Debug, Clone)]
pub struct AnalogModel {
    pub gain: f64,
    pub sign: f64,
    pub clamp_to_rails: bool,
    pub analog_out_r: f64,
    pub inverting: bool,
    pub r1: f64,
    pub r2: f64,
}

impl From<&AnalogOutCfg> for AnalogModel {
    fn from(cfg: &AnalogOutCfg) -> Self {
        Self {
            gain: cfg.gain.abs(),
            sign: if cfg.sign >= 0 { 1.0 } else { -1.0 },
            clamp_to_rails: cfg.clamp_to_rails,
            analog_out_r: cfg.analog_out_r_ohm,
            inverting: cfg.inverting,
            r1: cfg.r1_ohm,
            r2: cfg.r2_ohm,
        }
    }
}

impl EquivalentNeuron {
    pub fn from_configs(neuron: &NeuronConfig, network: &NetworkConfig) -> Self {
        let theta_model = ThresholdModel::from_components(
            neuron.supplies.vref,
            neuron.supplies.vdd,
            &neuron.threshold,
            &neuron.comparator,
        );

        Self {
            membrane: neuron.membrane.clone(),
            comparator: neuron.comparator.clone(),
            threshold: theta_model,
            analog: AnalogModel::from(&neuron.analog_out),
            supplies: neuron.supplies.clone(),
            sim: neuron.simulation.clone(),
            synapses: network.synapses.clone(),
            tau_membrane: neuron.membrane.c_mem_f * neuron.membrane.r_leak_ohm,
            comparator_high: (neuron.comparator.vhigh_v - COMPARATOR_RAIL_DROP_V)
                .max(neuron.comparator.vlow_v),
        }
    }

    pub fn run(&self) -> SimulationResult {
        let steps = (self.sim.tstop_s / self.sim.tstep_s).ceil() as usize + 1;
        let mut samples = Vec::with_capacity(steps);

        let mut time = 0.0;
        let mut u = 0.0; // deflection above vref
        let mut theta = self.threshold.theta_low;
        let mut v_comp = self.comparator.vlow_v;

        while time <= self.sim.tstop_s + 1e-12 {
            let syn_map = synapse_voltages(&self.synapses, time, self.supplies.vref);

            // Total synaptic current relative to Vref
            let mut i_syn = 0.0;
            for syn in &self.synapses {
                if let Some(v_syn) = syn_map.get(&syn.name) {
                    let delta_v = v_syn - self.supplies.vref;
                    i_syn += delta_v / syn.weight_ohm;
                }
            }

            let u_inf = self.membrane.r_leak_ohm * i_syn;
            u = relax_towards(u, u_inf, self.tau_membrane, self.sim.tstep_s);

            // Comparator logic
            let theta_eff = theta + self.comparator.offset_v;
            if v_comp <= self.comparator.vlow_v + 1e-9 {
                if u > theta_eff {
                    v_comp = self.comparator_high;
                }
            } else if u <= theta_eff {
                v_comp = self.comparator.vlow_v;
            }

            let theta_target = if v_comp > (self.comparator.vlow_v + self.comparator.vhigh_v) * 0.5
            {
                self.threshold.theta_high
            } else {
                self.threshold.theta_low
            };

            theta = adapt(
                theta,
                theta_target,
                self.threshold.tau_theta,
                self.sim.tstep_s,
            );

            if v_comp > self.comparator.vlow_v + 1e-9 {
                let overdrive =
                    (v_comp - (self.supplies.vref + theta) - DEFAULT_DIODE_DROP_V).max(0.0);
                let dtheta = (overdrive / DEFAULT_INJECTION_R_OHM)
                    * (self.sim.tstep_s / self.threshold.c_adapt);
                theta += dtheta;
            }

            let v_mem = self.supplies.vref - u;
            let v_ana = self.analog_output(v_mem);
            let v_outmix = mixed_output(v_ana, v_comp);

            samples.push(SimulationSample {
                time_s: time,
                v_mem,
                u_deflection: u,
                v_comp,
                v_ana,
                v_outmix,
                theta_over_vref: theta,
                v_syn: syn_map,
            });

            time += self.sim.tstep_s;
        }

        SimulationResult { samples }
    }

    fn analog_output(&self, v_mem: f64) -> f64 {
        let vin = match self.analog.sign >= 0.0 {
            true => v_mem - self.supplies.vref,
            false => self.supplies.vref - v_mem,
        };

        let magnitude = if self.analog.gain > 0.0 {
            self.analog.gain
        } else if self.analog.inverting {
            (self.analog.r2 / self.analog.r1).abs()
        } else {
            1.0 + self.analog.r2 / self.analog.r1
        };

        let mut vout = if self.analog.inverting {
            -magnitude * vin
        } else {
            magnitude * vin
        };
        if self.analog.clamp_to_rails {
            vout = vout.clamp(0.0, self.supplies.vdd);
        }

        vout
    }
}

impl ThresholdModel {
    pub fn from_components(
        vref: f64,
        vdd: f64,
        th: &crate::simulators::config::Threshold,
        comp: &ComparatorCfg,
    ) -> Self {
        let scale = th.divider_scale.max(1e-6);
        let r_vh = DIVIDER_R_VH_OHM * scale;
        let r_vl = DIVIDER_R_VL_OHM * scale;
        let dv_out = (comp.vhigh_v - comp.vlow_v).max(1e-6);
        let beta = (th.hysteresis_v / dv_out).clamp(1e-6, 0.999999);
        let g_div = (1.0 / r_vh) + (1.0 / r_vl);
        let r_f = 1.0 / (beta * g_div / (1.0 - beta));

        let g_total = g_div + 1.0 / r_f;
        let tau_theta = th.c_adapt_f / g_total;

        let numerator_low = (vdd - vref) / r_vh + (comp.vlow_v - vref) / r_f;
        let theta_low = numerator_low / g_total;
        let numerator_high = (vdd - vref) / r_vh + (comp.vhigh_v - vref) / r_f;
        let theta_high = numerator_high / g_total;

        Self {
            theta_low,
            theta_high,
            tau_theta,
            c_adapt: th.c_adapt_f,
        }
    }
}

fn synapse_voltages(synapses: &[Synapse], time: f64, vref: f64) -> HashMap<String, f64> {
    let mut map = HashMap::with_capacity(synapses.len());
    for syn in synapses {
        let mut voltage = vref;
        for spike in &syn.spikes {
            let t0 = spike.t_ms / 1000.0;
            let t1 = t0 + spike.width_ms / 1000.0;
            if time >= t0 && time <= t1 {
                let sign = if syn.r#type == SynapseType::Excitatory {
                    1.0
                } else {
                    -1.0
                };
                voltage = vref + sign * spike.amp_v;
                break;
            }
        }
        map.insert(syn.name.clone(), voltage);
    }
    map
}

fn mixed_output(v_ana: f64, v_comp: f64) -> f64 {
    let alpha =
        DEFAULT_ANALOG_LOAD_R_OHM / (DEFAULT_ANALOG_LOAD_R_OHM + DEFAULT_ANALOG_SERIES_R_OHM);
    let analog_path = alpha * v_ana;
    let spike_path = (v_comp - DEFAULT_DIODE_DROP_V).max(analog_path);
    spike_path
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub samples: Vec<SimulationSample>,
}

impl SimulationResult {
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn to_csv(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record([
            "time_s",
            "v_mem",
            "u_deflection",
            "v_comp",
            "v_ana",
            "v_outmix",
            "theta_over_vref",
        ])?;

        for sample in &self.samples {
            wtr.write_record([
                sample.time_s.to_string(),
                sample.v_mem.to_string(),
                sample.u_deflection.to_string(),
                sample.v_comp.to_string(),
                sample.v_ana.to_string(),
                sample.v_outmix.to_string(),
                sample.theta_over_vref.to_string(),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }
}
