//! Stunning network animations using nannou
//!
//! Real-time visualization of spiking neural network activity with:
//! - Neuron circles that glow when spiking
//! - Synaptic connections with weight-based thickness
//! - Spike propagation animations
//! - Membrane potential color coding
//!
//! Enable with the `animation` feature:
//! ```toml
//! gilgamesh = { version = "0.1", features = ["animation"] }
//! ```

#[cfg(feature = "animation")]
use nannou::prelude::*;

#[cfg(feature = "animation")]
use std::sync::{Arc, Mutex, OnceLock};

/// Neuron state for visualization
#[cfg(feature = "animation")]
#[derive(Clone)]
pub struct NeuronVisual {
    /// Position on screen
    pub pos: (f32, f32),
    /// Current membrane potential (0.0 - 1.0 normalized)
    pub membrane: f32,
    /// Time since last spike (for glow decay)
    pub spike_time: f32,
    /// Is currently spiking
    pub spiking: bool,
    /// Layer index
    pub layer: usize,
    /// Neuron index within layer
    pub index: usize,
}

#[cfg(feature = "animation")]
impl NeuronVisual {
    pub fn new(pos: (f32, f32), layer: usize, index: usize) -> Self {
        Self {
            pos,
            membrane: 0.0,
            spike_time: f32::MAX,
            spiking: false,
            layer,
            index,
        }
    }
}

/// Synapse for visualization
#[cfg(feature = "animation")]
#[derive(Clone)]
pub struct SynapseVisual {
    /// Source neuron index
    pub from: usize,
    /// Target neuron index
    pub to: usize,
    /// Weight magnitude (determines line thickness)
    pub weight: f32,
    /// Is excitatory (positive weight)
    pub excitatory: bool,
    /// Animation progress for spike propagation (0.0 - 1.0)
    pub propagation: f32,
}

/// Network state for animation
#[cfg(feature = "animation")]
#[derive(Clone, Default)]
pub struct NetworkAnimation {
    /// All neurons across layers
    pub neurons: Vec<NeuronVisual>,
    /// All synapses
    pub synapses: Vec<SynapseVisual>,
    /// Layer sizes for layout
    pub layer_sizes: Vec<usize>,
    /// Current simulation time
    pub time: f32,
    /// Animation speed multiplier
    pub speed: f32,
    /// Whether animation is paused
    pub paused: bool,
}

#[cfg(feature = "animation")]
impl NetworkAnimation {
    /// Create a new network animation with given layer sizes
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut neurons = Vec::new();
        let mut layer_offset = 0;

        // Create neurons for each layer
        for (layer_idx, &size) in layer_sizes.iter().enumerate() {
            for neuron_idx in 0..size {
                neurons.push(NeuronVisual::new(
                    (0.0, 0.0), // Position will be calculated in layout
                    layer_idx,
                    neuron_idx,
                ));
            }
            layer_offset += size;
        }

        Self {
            neurons,
            synapses: Vec::new(),
            layer_sizes: layer_sizes.to_vec(),
            time: 0.0,
            speed: 1.0,
            paused: false,
        }
    }

    /// Calculate neuron positions based on window size
    pub fn layout(&mut self, width: f32, height: f32) {
        let num_layers = self.layer_sizes.len();
        let margin = 80.0;
        let usable_width = width - 2.0 * margin;
        let usable_height = height - 2.0 * margin;

        let layer_spacing = if num_layers > 1 {
            usable_width / (num_layers - 1) as f32
        } else {
            0.0
        };

        let mut idx = 0;
        for (layer_idx, &layer_size) in self.layer_sizes.iter().enumerate() {
            let x = -width / 2.0 + margin + layer_idx as f32 * layer_spacing;

            // Limit display neurons for large layers
            let display_size = layer_size.min(50);
            let neuron_spacing = if display_size > 1 {
                usable_height / (display_size - 1).max(1) as f32
            } else {
                0.0
            };

            let start_y = if display_size > 1 {
                height / 2.0 - margin
            } else {
                0.0
            };

            for i in 0..layer_size {
                if i < display_size {
                    let y = start_y - i as f32 * neuron_spacing;
                    self.neurons[idx].pos = (x, y);
                } else {
                    // Hide excess neurons off-screen
                    self.neurons[idx].pos = (x, -height);
                }
                idx += 1;
            }
        }
    }

    /// Set weights from weight matrices (creates synapses)
    pub fn set_weights(&mut self, weights: &[Vec<Vec<f32>>]) {
        self.synapses.clear();
        let num_neurons = self.neurons.len();

        let mut from_offset = 0;
        for (layer_idx, weight_matrix) in weights.iter().enumerate() {
            if layer_idx >= self.layer_sizes.len() {
                break;
            }
            let to_offset = from_offset + self.layer_sizes[layer_idx];

            for (to_idx, row) in weight_matrix.iter().enumerate() {
                for (from_idx, &weight) in row.iter().enumerate() {
                    let from = from_offset + from_idx;
                    let to = to_offset + to_idx;

                    // Bounds check
                    if from >= num_neurons || to >= num_neurons {
                        continue;
                    }

                    if weight.abs() > 0.01 {
                        self.synapses.push(SynapseVisual {
                            from,
                            to,
                            weight: weight.abs(),
                            excitatory: weight > 0.0,
                            propagation: 0.0,
                        });
                    }
                }
            }

            from_offset = to_offset;
        }
    }

    /// Update neuron states from simulation data
    pub fn update_neurons(&mut self, membranes: &[Vec<f32>], spikes: &[Vec<bool>], dt: f32) {
        if self.paused {
            return;
        }

        self.time += dt * self.speed;

        let mut idx = 0;
        for (layer_idx, layer_mem) in membranes.iter().enumerate() {
            let layer_spikes = &spikes[layer_idx];

            for (neuron_idx, &mem) in layer_mem.iter().enumerate() {
                if idx < self.neurons.len() {
                    self.neurons[idx].membrane = mem.clamp(0.0, 1.0);

                    if neuron_idx < layer_spikes.len() && layer_spikes[neuron_idx] {
                        self.neurons[idx].spiking = true;
                        self.neurons[idx].spike_time = self.time;
                    } else {
                        self.neurons[idx].spiking = false;
                    }
                }
                idx += 1;
            }
        }

        // Update spike propagation on synapses
        for synapse in &mut self.synapses {
            if synapse.from >= self.neurons.len() {
                continue;
            }
            let from_neuron = &self.neurons[synapse.from];
            if from_neuron.spiking {
                synapse.propagation = 0.0;
            } else if synapse.propagation < 1.0 {
                synapse.propagation += dt * self.speed * 3.0;
            }
        }
    }

    /// Get neuron index from layer and position
    fn neuron_index(&self, layer: usize, pos: usize) -> usize {
        let mut idx = 0;
        for i in 0..layer {
            idx += self.layer_sizes[i];
        }
        idx + pos
    }
}

/// Shared state for nannou app
#[cfg(feature = "animation")]
pub type SharedAnimation = Arc<Mutex<NetworkAnimation>>;

/// Global animation state (set before running nannou)
#[cfg(feature = "animation")]
static ANIMATION_STATE: OnceLock<SharedAnimation> = OnceLock::new();

/// Create shared animation state
#[cfg(feature = "animation")]
pub fn create_shared_animation(layer_sizes: &[usize]) -> SharedAnimation {
    Arc::new(Mutex::new(NetworkAnimation::new(layer_sizes)))
}

/// Nannou model
#[cfg(feature = "animation")]
struct Model {
    animation: SharedAnimation,
}

/// Run the animation window
#[cfg(feature = "animation")]
pub fn run_animation(animation: SharedAnimation) {
    // Store animation in global state for nannou model function
    let _ = ANIMATION_STATE.set(animation);

    nannou::app(model)
        .update(update)
        .simple_window(view)
        .size(1400, 900)
        .run();
}

#[cfg(feature = "animation")]
fn model(_app: &App) -> Model {
    let animation = ANIMATION_STATE.get().expect("Animation state not set").clone();
    Model { animation }
}

#[cfg(feature = "animation")]
fn update(app: &App, model: &mut Model, _update: Update) {
    let mut anim = model.animation.lock().unwrap();

    // Update layout if window size changed
    let win = app.window_rect();
    anim.layout(win.w(), win.h());
}

#[cfg(feature = "animation")]
fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let anim = model.animation.lock().unwrap();

    // Dark background
    draw.background().color(rgb(0.05, 0.05, 0.1));

    // Draw synapses
    for synapse in &anim.synapses {
        // Bounds check
        if synapse.from >= anim.neurons.len() || synapse.to >= anim.neurons.len() {
            continue;
        }
        let from = &anim.neurons[synapse.from];
        let to = &anim.neurons[synapse.to];

        // Skip if either neuron is off-screen or has invalid position
        if from.pos.1 < -400.0 || to.pos.1 < -400.0 {
            continue;
        }
        if !from.pos.0.is_finite() || !from.pos.1.is_finite() || !to.pos.0.is_finite() || !to.pos.1.is_finite() {
            continue;
        }

        let base_color = if synapse.excitatory {
            rgb(0.2, 0.4, 0.8) // Blue for excitatory
        } else {
            rgb(0.8, 0.2, 0.3) // Red for inhibitory
        };

        // Line thickness based on weight
        let thickness = (synapse.weight * 3.0).clamp(0.5, 4.0);

        // Fade alpha based on weight
        let alpha = (synapse.weight * 0.5).clamp(0.1, 0.6);

        draw.line()
            .start(pt2(from.pos.0, from.pos.1))
            .end(pt2(to.pos.0, to.pos.1))
            .weight(thickness)
            .color(rgba(base_color.red, base_color.green, base_color.blue, alpha));

        // Draw propagation pulse
        if synapse.propagation.is_finite() && synapse.propagation < 1.0 && synapse.propagation > 0.0 {
            let t = synapse.propagation;
            let pulse_x = from.pos.0 + (to.pos.0 - from.pos.0) * t;
            let pulse_y = from.pos.1 + (to.pos.1 - from.pos.1) * t;

            if pulse_x.is_finite() && pulse_y.is_finite() {
                let pulse_alpha = (1.0 - t) * 0.8;
                draw.ellipse()
                    .x_y(pulse_x, pulse_y)
                    .radius(4.0)
                    .color(rgba(1.0, 1.0, 0.5, pulse_alpha));
            }
        }
    }

    // Draw neurons
    for neuron in &anim.neurons {
        // Skip if off-screen or invalid position
        if neuron.pos.1 < -400.0 || !neuron.pos.0.is_finite() || !neuron.pos.1.is_finite() {
            continue;
        }

        let base_radius = 8.0;
        let time_since_spike = anim.time - neuron.spike_time;

        // Glow effect when recently spiked (only if valid time)
        if time_since_spike.is_finite() && time_since_spike >= 0.0 && time_since_spike < 0.5 {
            let glow_intensity = (1.0 - time_since_spike / 0.5).powi(2);
            let glow_radius = base_radius + 20.0 * glow_intensity;

            if glow_radius > 0.0 && glow_intensity > 0.0 {
                // Outer glow
                draw.ellipse()
                    .x_y(neuron.pos.0, neuron.pos.1)
                    .radius(glow_radius)
                    .color(rgba(1.0, 0.9, 0.3, glow_intensity * 0.3));

                // Middle glow
                draw.ellipse()
                    .x_y(neuron.pos.0, neuron.pos.1)
                    .radius(glow_radius * 0.6)
                    .color(rgba(1.0, 0.95, 0.5, glow_intensity * 0.5));
            }
        }

        // Neuron color based on membrane potential
        let mem = neuron.membrane.clamp(0.0, 1.0);
        let neuron_color = if neuron.spiking {
            rgb(1.0, 1.0, 0.8) // Bright white-yellow when spiking
        } else {
            // Gradient from dark blue (rest) to orange (near threshold)
            rgb(
                0.2 + mem * 0.7,
                0.2 + mem * 0.4,
                0.5 - mem * 0.3,
            )
        };

        // Draw neuron body
        draw.ellipse()
            .x_y(neuron.pos.0, neuron.pos.1)
            .radius(base_radius)
            .color(neuron_color);
    }

    // Draw legend in corner (simplified to avoid text rendering issues)
    let win = app.window_rect();
    let legend_x = win.w() / 2.0 - 80.0;
    let legend_y = -win.h() / 2.0 + 60.0;

    // Spike indicator
    draw.ellipse()
        .x_y(legend_x - 30.0, legend_y + 20.0)
        .radius(6.0)
        .color(rgb(1.0, 1.0, 0.8));

    // Excitatory line
    draw.line()
        .start(pt2(legend_x - 40.0, legend_y))
        .end(pt2(legend_x - 20.0, legend_y))
        .weight(2.0)
        .color(rgb(0.2, 0.4, 0.8));

    // Inhibitory line
    draw.line()
        .start(pt2(legend_x - 40.0, legend_y - 20.0))
        .end(pt2(legend_x - 20.0, legend_y - 20.0))
        .weight(2.0)
        .color(rgb(0.8, 0.2, 0.3));

    draw.to_frame(app, &frame).unwrap();
}

// Stub implementation when feature is disabled
#[cfg(not(feature = "animation"))]
#[derive(Clone, Default)]
pub struct NetworkAnimation;

#[cfg(not(feature = "animation"))]
impl NetworkAnimation {
    pub fn new(_layer_sizes: &[usize]) -> Self { Self }
    pub fn layout(&mut self, _width: f32, _height: f32) {}
    pub fn set_weights(&mut self, _weights: &[Vec<Vec<f32>>]) {}
    pub fn update_neurons(&mut self, _membranes: &[Vec<f32>], _spikes: &[Vec<bool>], _dt: f32) {}
}
