use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::simulators::config::{NetworkConfig, NeuronConfig};
use crate::simulators::lif_equivalent::{EquivalentNeuron, SimulationResult, SimulationSample};

#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    pub network_config: PathBuf,
    pub neuron_config: Option<PathBuf>,
    pub spice_csv: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalComparison {
    pub signal: String,
    pub rms_error: f64,
    pub mean_abs_error: f64,
    pub max_abs_error: f64,
    pub max_abs_error_time: f64,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalSamplePair {
    pub time_s: f64,
    pub equivalent: f64,
    pub spice: f64,
    pub delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub equivalent: SimulationResult,
    pub metrics: Vec<SignalComparison>,
    pub signal_series: HashMap<String, Vec<SignalSamplePair>>,
    #[serde(default)]
    pub timings: TimingBreakdown,
}

impl ComparisonResult {
    pub fn write_json(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_vec_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingBreakdown {
    pub rust_preprocess_ns: u128,
    pub rust_simulate_ns: u128,
    pub rust_analysis_ns: u128,
    pub rust_serialization_ns: u128,
    pub spice_parse_ns: u128,
    #[serde(default)]
    pub spice_preprocess_ns: Option<u128>,
    #[serde(default)]
    pub spice_simulate_ns: Option<u128>,
    #[serde(default)]
    pub spice_post_ns: Option<u128>,
}

pub fn run_comparison(cfg: &ComparisonConfig) -> Result<ComparisonResult> {
    let network_cfg = NetworkConfig::load(&cfg.network_config).with_context(|| {
        format!(
            "failed to load network config at {}",
            cfg.network_config.display()
        )
    })?;

    let neuron_path = if let Some(path) = &cfg.neuron_config {
        path.clone()
    } else {
        let base = cfg
            .network_config
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        base.join(&network_cfg.neuron_json)
    };

    let neuron_cfg = NeuronConfig::load(&neuron_path)
        .with_context(|| format!("failed to load neuron config at {}", neuron_path.display()))?;

    let build_start = Instant::now();
    let equivalent_model = EquivalentNeuron::from_configs(&neuron_cfg, &network_cfg);
    let rust_preprocess_ns = build_start.elapsed().as_nanos();

    let sim_start = Instant::now();
    let equivalent = equivalent_model.run();
    let rust_simulate_ns = sim_start.elapsed().as_nanos();

    let spice_start = Instant::now();
    let spice_trace = SpiceTrace::from_ascii(&cfg.spice_csv)
        .with_context(|| format!("failed to load SPICE CSV at {}", cfg.spice_csv.display()))?;
    let spice_parse_ns = spice_start.elapsed().as_nanos();

    let analysis_start = Instant::now();
    let (metrics, signal_series) = compare_against_spice(&equivalent, &spice_trace)?;
    let rust_analysis_ns = analysis_start.elapsed().as_nanos();

    Ok(ComparisonResult {
        equivalent,
        metrics,
        signal_series,
        timings: TimingBreakdown {
            rust_preprocess_ns,
            rust_simulate_ns,
            rust_analysis_ns,
            rust_serialization_ns: 0,
            spice_parse_ns,
            ..TimingBreakdown::default()
        },
    })
}

struct SpiceTrace {
    times: Vec<f64>,
    signals: HashMap<String, Vec<f64>>,
}

impl SpiceTrace {
    fn from_ascii(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("unable to open SPICE output file {}", path.display()))?;
        let reader = BufReader::new(file);

        let mut header: Option<Vec<String>> = None;
        let mut data: Vec<Vec<f64>> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if header.is_none() {
                let cols: Vec<String> = trimmed.split_whitespace().map(|s| s.to_string()).collect();
                if cols.is_empty() {
                    bail!("SPICE header row is empty in {}", path.display());
                }
                header = Some(cols);
                continue;
            }

            let header_ref = header.as_ref().unwrap();
            let values: Vec<f64> = trimmed
                .split_whitespace()
                .map(|s| s.parse::<f64>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to parse numeric data in {}", path.display()))?;

            if values.len() != header_ref.len() {
                bail!(
                    "row with {} columns does not match header length {} in {}",
                    values.len(),
                    header_ref.len(),
                    path.display()
                );
            }

            if data.is_empty() {
                data = vec![Vec::new(); values.len()];
            }

            for (idx, val) in values.into_iter().enumerate() {
                data[idx].push(val);
            }
        }

        let header =
            header.ok_or_else(|| anyhow::anyhow!("missing header in {}", path.display()))?;

        if data.is_empty() {
            bail!("no data rows found in {}", path.display());
        }

        let mut signals = HashMap::new();
        let mut seen: HashMap<String, usize> = HashMap::new();

        if data.is_empty() || data[0].is_empty() {
            bail!("SPICE trace is missing time samples in {}", path.display());
        }

        let times = data[0].clone();
        if let Some(first_name) = header.get(0) {
            seen.insert(first_name.clone(), 1);
        }

        for (idx, name) in header.into_iter().enumerate().skip(1) {
            let canonical = sanitise_column_name(&name, &mut seen);
            signals.insert(canonical, data[idx].clone());
        }

        Ok(SpiceTrace { times, signals })
    }

    fn interpolate(&self, signal: &str, time: f64) -> Option<f64> {
        let values = self.signals.get(signal)?;
        if self.times.is_empty() {
            return None;
        }
        let first = *self.times.first()?;
        let last = *self.times.last()?;
        if time < first || time > last {
            return None;
        }

        match self
            .times
            .binary_search_by(|probe| probe.partial_cmp(&time).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(idx) => values.get(idx).copied(),
            Err(idx) => {
                if idx == 0 || idx >= self.times.len() {
                    return None;
                }
                let t0 = self.times[idx - 1];
                let t1 = self.times[idx];
                let v0 = values[idx - 1];
                let v1 = values[idx];
                if (t1 - t0).abs() < f64::EPSILON {
                    Some(v0)
                } else {
                    let alpha = (time - t0) / (t1 - t0);
                    Some(v0 + (v1 - v0) * alpha)
                }
            }
        }
    }
}

fn sanitise_column_name(name: &str, seen: &mut HashMap<String, usize>) -> String {
    let entry = seen.entry(name.to_string()).or_insert(0);
    if *entry == 0 {
        *entry += 1;
        name.to_string()
    } else {
        *entry += 1;
        format!("{}_{}", name, *entry)
    }
}

struct SignalSpec {
    id: &'static str,
    spice_column: &'static str,
    extractor: fn(&SimulationSample) -> f64,
}

const SIGNAL_SPECS: &[SignalSpec] = &[
    SignalSpec {
        id: "v_mem",
        spice_column: "v(mem)",
        extractor: |s: &SimulationSample| s.v_mem,
    },
    SignalSpec {
        id: "v_comp",
        spice_column: "v(comp)",
        extractor: |s: &SimulationSample| s.v_comp,
    },
    SignalSpec {
        id: "v_outmix",
        spice_column: "v(n_outmix)",
        extractor: |s: &SimulationSample| s.v_outmix,
    },
    SignalSpec {
        id: "v_ana",
        spice_column: "v(ana)",
        extractor: |s: &SimulationSample| s.v_ana,
    },
];

fn compare_against_spice(
    equivalent: &SimulationResult,
    spice: &SpiceTrace,
) -> Result<(
    Vec<SignalComparison>,
    HashMap<String, Vec<SignalSamplePair>>,
)> {
    let mut metrics = Vec::new();
    let mut series_map: HashMap<String, Vec<SignalSamplePair>> = HashMap::new();

    for spec in SIGNAL_SPECS {
        let mut sum_sq = 0.0;
        let mut sum_abs = 0.0;
        let mut max_abs = 0.0;
        let mut max_time = 0.0;
        let mut count = 0usize;
        let mut series = Vec::new();

        for sample in &equivalent.samples {
            let eq_val = (spec.extractor)(sample);
            let time = sample.time_s;
            if let Some(spice_val) = spice.interpolate(spec.spice_column, time) {
                let delta = eq_val - spice_val;
                let abs_delta = delta.abs();
                sum_sq += delta * delta;
                sum_abs += abs_delta;
                if abs_delta > max_abs {
                    max_abs = abs_delta;
                    max_time = time;
                }
                count += 1;
                series.push(SignalSamplePair {
                    time_s: time,
                    equivalent: eq_val,
                    spice: spice_val,
                    delta,
                });
            }
        }

        if count == 0 {
            continue;
        }

        let rms = (sum_sq / count as f64).sqrt();
        let mean_abs = sum_abs / count as f64;
        metrics.push(SignalComparison {
            signal: spec.id.to_string(),
            rms_error: rms,
            mean_abs_error: mean_abs,
            max_abs_error: max_abs,
            max_abs_error_time: max_time,
            sample_count: count,
        });
        series_map.insert(spec.id.to_string(), series);
    }

    Ok((metrics, series_map))
}
