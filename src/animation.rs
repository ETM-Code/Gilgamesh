//! Interactive network visualization using nannou
//!
//! Real-time visualization of spiking neural network activity with:
//! - MNIST image display on left
//! - Network activity visualization on right
//! - Output probabilities shown next to neurons
//! - Arrow key navigation between samples
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
    /// Accumulated spike count (for output neurons)
    pub spike_count: u32,
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
            spike_count: 0,
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

/// Interactive viewer state
#[cfg(feature = "animation")]
#[derive(Clone)]
pub struct ViewerState {
    /// Current sample index
    pub current_sample: usize,
    /// Total number of samples
    pub total_samples: usize,
    /// Current label
    pub label: u8,
    /// Predicted class (from spike counts)
    pub prediction: usize,
    /// Whether prediction is correct
    pub correct: bool,
    /// Current simulation step
    pub current_step: usize,
    /// Total simulation steps
    pub total_steps: usize,
    /// MNIST image pixels (7x7 = 49 or 28x28 = 784)
    pub image_pixels: Vec<f32>,
    /// Image dimensions
    pub image_size: usize,
    /// Output spike counts for probabilities
    pub output_spikes: Vec<u32>,
    /// Request to change sample (set by key events)
    pub sample_request: Option<i32>, // -1 for prev, +1 for next
    /// Is simulation complete for current sample?
    pub simulation_complete: bool,
}

#[cfg(feature = "animation")]
impl Default for ViewerState {
    fn default() -> Self {
        Self {
            current_sample: 0,
            total_samples: 10000,
            label: 0,
            prediction: 0,
            correct: false,
            current_step: 0,
            total_steps: 25,
            image_pixels: vec![0.0; 36],
            image_size: 6,
            output_spikes: vec![0; 10],
            sample_request: None,
            simulation_complete: false,
        }
    }
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
    /// Interactive viewer state
    pub viewer: ViewerState,
}

#[cfg(feature = "animation")]
impl NetworkAnimation {
    /// Create a new network animation with given layer sizes
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut neurons = Vec::new();

        // Create neurons for each layer
        for (layer_idx, &size) in layer_sizes.iter().enumerate() {
            for neuron_idx in 0..size {
                neurons.push(NeuronVisual::new(
                    (0.0, 0.0), // Position will be calculated in layout
                    layer_idx,
                    neuron_idx,
                ));
            }
        }

        Self {
            neurons,
            synapses: Vec::new(),
            layer_sizes: layer_sizes.to_vec(),
            time: 0.0,
            speed: 1.0,
            paused: false,
            viewer: ViewerState::default(),
        }
    }

    /// Calculate neuron positions based on window size (right side of screen)
    pub fn layout(&mut self, width: f32, height: f32) {
        let num_layers = self.layer_sizes.len();
        let margin = 60.0;

        // Network takes right 60% of screen
        let network_left = -width / 2.0 + width * 0.4;
        let network_width = width * 0.55;
        let usable_height = height - 2.0 * margin;

        let layer_spacing = if num_layers > 1 {
            network_width / (num_layers - 1) as f32
        } else {
            0.0
        };

        let mut idx = 0;
        for (layer_idx, &layer_size) in self.layer_sizes.iter().enumerate() {
            let x = network_left + layer_idx as f32 * layer_spacing;

            // Limit display neurons for large layers
            let display_size = layer_size.min(40);
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

                    // Show connections with significant weights
                    if weight.abs() > 0.005 {
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
        // Note: current_step is set explicitly by the caller, not incremented here

        let mut idx = 0;
        for (layer_idx, layer_mem) in membranes.iter().enumerate() {
            let layer_spikes = &spikes[layer_idx];

            for (neuron_idx, &mem) in layer_mem.iter().enumerate() {
                if idx < self.neurons.len() {
                    self.neurons[idx].membrane = mem.clamp(0.0, 1.0);

                    if neuron_idx < layer_spikes.len() && layer_spikes[neuron_idx] {
                        self.neurons[idx].spiking = true;
                        self.neurons[idx].spike_time = self.time;
                        self.neurons[idx].spike_count += 1;

                        // Track output layer spikes
                        if layer_idx == membranes.len() - 1
                            && neuron_idx < self.viewer.output_spikes.len()
                        {
                            self.viewer.output_spikes[neuron_idx] += 1;
                        }
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

        // Update prediction based on spike counts
        if let Some((pred, _)) = self
            .viewer
            .output_spikes
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
        {
            self.viewer.prediction = pred;
            self.viewer.correct = pred == self.viewer.label as usize;
        }
    }

    /// Reset for new sample
    pub fn reset_for_sample(&mut self, sample_idx: usize, label: u8, image: &[f32]) {
        self.time = 0.0;
        self.viewer.current_sample = sample_idx;
        self.viewer.label = label;
        self.viewer.current_step = 0;
        self.viewer.simulation_complete = false;
        self.viewer.output_spikes = vec![0; 10];
        self.viewer.prediction = 0;
        self.viewer.correct = false;
        self.viewer.image_pixels = image.to_vec();
        self.viewer.image_size = (image.len() as f32).sqrt() as usize;

        // Reset all neurons
        for neuron in &mut self.neurons {
            neuron.membrane = 0.0;
            neuron.spike_time = f32::MAX;
            neuron.spiking = false;
            neuron.spike_count = 0;
        }

        // Reset synapses
        for synapse in &mut self.synapses {
            synapse.propagation = 1.0;
        }
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
        .event(event)
        .simple_window(view)
        .size(1600, 900)
        .run();
}

#[cfg(feature = "animation")]
fn model(_app: &App) -> Model {
    let animation = ANIMATION_STATE
        .get()
        .expect("Animation state not set")
        .clone();
    Model { animation }
}

#[cfg(feature = "animation")]
fn event(_app: &App, model: &mut Model, event: Event) {
    if let Event::WindowEvent {
        simple: Some(event),
        ..
    } = event
    {
        match event {
            KeyPressed(Key::Left) | KeyPressed(Key::A) => {
                let mut anim = model.animation.lock().unwrap();
                anim.viewer.sample_request = Some(-1);
            }
            KeyPressed(Key::Right) | KeyPressed(Key::D) => {
                let mut anim = model.animation.lock().unwrap();
                anim.viewer.sample_request = Some(1);
            }
            KeyPressed(Key::Space) => {
                let mut anim = model.animation.lock().unwrap();
                anim.paused = !anim.paused;
            }
            KeyPressed(Key::R) => {
                let mut anim = model.animation.lock().unwrap();
                anim.viewer.sample_request = Some(0); // Restart current
            }
            _ => {}
        }
    }
}

#[cfg(feature = "animation")]
fn update(app: &App, model: &mut Model, _update: Update) {
    let mut anim = model.animation.lock().unwrap();
    let win = app.window_rect();
    anim.layout(win.w(), win.h());
}

#[cfg(feature = "animation")]
fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let anim = model.animation.lock().unwrap();
    let win = app.window_rect();

    // Dark background
    draw.background().color(rgb(0.05, 0.05, 0.1));

    // === LEFT SIDE: MNIST Image ===
    let img_center_x = -win.w() / 2.0 + win.w() * 0.15;
    let img_center_y = win.h() * 0.15;
    let img_size = 200.0;
    let pixel_size = img_size / anim.viewer.image_size as f32;

    // Draw image background
    draw.rect()
        .x_y(img_center_x, img_center_y)
        .w_h(img_size + 10.0, img_size + 10.0)
        .color(rgb(0.15, 0.15, 0.2));

    // Draw MNIST pixels
    let half_img = img_size / 2.0;
    for (i, &pixel) in anim.viewer.image_pixels.iter().enumerate() {
        let row = i / anim.viewer.image_size;
        let col = i % anim.viewer.image_size;
        let x = img_center_x - half_img + col as f32 * pixel_size + pixel_size / 2.0;
        let y = img_center_y + half_img - row as f32 * pixel_size - pixel_size / 2.0;

        let brightness = pixel.clamp(0.0, 1.0);
        draw.rect()
            .x_y(x, y)
            .w_h(pixel_size - 1.0, pixel_size - 1.0)
            .color(rgb(brightness, brightness, brightness));
    }

    // Draw label and prediction below image
    let info_y = img_center_y - img_size / 2.0 - 40.0;

    // Sample info
    draw.text(&format!(
        "Sample: {}/{}",
        anim.viewer.current_sample + 1,
        anim.viewer.total_samples
    ))
    .x_y(img_center_x, info_y + 60.0)
    .color(rgb(0.7, 0.7, 0.8))
    .font_size(14);

    draw.text(&format!("Label: {}", anim.viewer.label))
        .x_y(img_center_x, info_y + 30.0)
        .color(rgb(0.9, 0.9, 0.5))
        .font_size(18);

    let pred_color = if anim.viewer.correct {
        rgb(0.3, 0.9, 0.3)
    } else {
        rgb(0.9, 0.3, 0.3)
    };
    draw.text(&format!("Pred: {}", anim.viewer.prediction))
        .x_y(img_center_x, info_y)
        .color(pred_color)
        .font_size(18);

    // Progress bar - positioned at bottom left corner
    let progress = anim.viewer.current_step as f32 / anim.viewer.total_steps as f32;
    let bar_width = 180.0;
    let bar_y = -win.h() / 2.0 + 60.0;

    draw.rect()
        .x_y(img_center_x, bar_y)
        .w_h(bar_width, 8.0)
        .color(rgb(0.2, 0.2, 0.25));
    draw.rect()
        .x_y(
            img_center_x - bar_width / 2.0 + (bar_width * progress) / 2.0,
            bar_y,
        )
        .w_h(bar_width * progress, 8.0)
        .color(rgb(0.3, 0.6, 0.9));

    // Step counter above progress bar
    draw.text(&format!(
        "Step {}/{}",
        anim.viewer.current_step, anim.viewer.total_steps
    ))
    .x_y(img_center_x, bar_y + 20.0)
    .color(rgb(0.6, 0.6, 0.7))
    .font_size(12);

    // Controls help - at very bottom
    draw.text("← → prev/next  Space: pause  R: restart")
        .x_y(img_center_x, bar_y - 25.0)
        .color(rgb(0.4, 0.4, 0.5))
        .font_size(10);

    // === RIGHT SIDE: Network Visualization ===

    // Draw synapses
    for synapse in &anim.synapses {
        if synapse.from >= anim.neurons.len() || synapse.to >= anim.neurons.len() {
            continue;
        }
        let from = &anim.neurons[synapse.from];
        let to = &anim.neurons[synapse.to];

        if from.pos.1 < -400.0 || to.pos.1 < -400.0 {
            continue;
        }
        if !from.pos.0.is_finite()
            || !from.pos.1.is_finite()
            || !to.pos.0.is_finite()
            || !to.pos.1.is_finite()
        {
            continue;
        }

        let base_color = if synapse.excitatory {
            rgb(0.2, 0.4, 0.8)
        } else {
            rgb(0.8, 0.2, 0.3)
        };

        let thickness = (synapse.weight * 2.0).clamp(0.3, 3.0);
        let alpha = (synapse.weight * 0.4).clamp(0.05, 0.4);

        draw.line()
            .start(pt2(from.pos.0, from.pos.1))
            .end(pt2(to.pos.0, to.pos.1))
            .weight(thickness)
            .color(rgba(
                base_color.red,
                base_color.green,
                base_color.blue,
                alpha,
            ));

        // Propagation pulse
        if synapse.propagation.is_finite() && synapse.propagation < 1.0 && synapse.propagation > 0.0
        {
            let t = synapse.propagation;
            let pulse_x = from.pos.0 + (to.pos.0 - from.pos.0) * t;
            let pulse_y = from.pos.1 + (to.pos.1 - from.pos.1) * t;

            if pulse_x.is_finite() && pulse_y.is_finite() {
                let pulse_alpha = (1.0 - t) * 0.8;
                draw.ellipse().x_y(pulse_x, pulse_y).radius(3.0).color(rgba(
                    1.0,
                    1.0,
                    0.5,
                    pulse_alpha,
                ));
            }
        }
    }

    // Draw neurons
    for (i, neuron) in anim.neurons.iter().enumerate() {
        if neuron.pos.1 < -400.0 || !neuron.pos.0.is_finite() || !neuron.pos.1.is_finite() {
            continue;
        }

        let base_radius = 6.0;
        let time_since_spike = anim.time - neuron.spike_time;

        // Glow effect when recently spiked
        if time_since_spike.is_finite() && time_since_spike >= 0.0 && time_since_spike < 0.5 {
            let glow_intensity = (1.0 - time_since_spike / 0.5).powi(2);
            let glow_radius = base_radius + 15.0 * glow_intensity;

            if glow_radius > 0.0 && glow_intensity > 0.0 {
                draw.ellipse()
                    .x_y(neuron.pos.0, neuron.pos.1)
                    .radius(glow_radius)
                    .color(rgba(1.0, 0.9, 0.3, glow_intensity * 0.3));
            }
        }

        // Neuron color based on membrane potential
        let mem = neuron.membrane.clamp(0.0, 1.0);
        let neuron_color = if neuron.spiking {
            rgb(1.0, 1.0, 0.8)
        } else {
            rgb(0.2 + mem * 0.7, 0.2 + mem * 0.4, 0.5 - mem * 0.3)
        };

        draw.ellipse()
            .x_y(neuron.pos.0, neuron.pos.1)
            .radius(base_radius)
            .color(neuron_color);

        // For output layer, show class label and spike count / probability
        if neuron.layer == anim.layer_sizes.len() - 1 {
            let total_spikes: u32 = anim.viewer.output_spikes.iter().sum();
            let prob = if total_spikes > 0 {
                anim.viewer.output_spikes[neuron.index] as f32 / total_spikes as f32
            } else {
                0.0
            };

            // Class label - to the LEFT of the neuron
            let label_color = if neuron.index == anim.viewer.prediction {
                rgb(0.4, 1.0, 0.5) // Bright green for prediction
            } else {
                rgb(0.7, 0.7, 0.8)
            };
            draw.text(&format!("{}", neuron.index))
                .x_y(neuron.pos.0 - base_radius - 15.0, neuron.pos.1)
                .color(label_color)
                .font_size(14);

            // Probability bar - to the RIGHT of the neuron
            let bar_max_width = 50.0;
            let bar_height = 8.0;
            let bar_x = neuron.pos.0 + base_radius + 8.0 + bar_max_width / 2.0;

            // Background
            draw.rect()
                .x_y(bar_x, neuron.pos.1)
                .w_h(bar_max_width, bar_height)
                .color(rgba(0.2, 0.2, 0.25, 0.8));

            // Fill
            let fill_width = bar_max_width * prob;
            if fill_width > 0.1 {
                let bar_color = if neuron.index == anim.viewer.prediction {
                    rgb(0.3, 0.8, 0.4)
                } else {
                    rgb(0.4, 0.5, 0.7)
                };
                draw.rect()
                    .x_y(bar_x - bar_max_width / 2.0 + fill_width / 2.0, neuron.pos.1)
                    .w_h(fill_width, bar_height)
                    .color(bar_color);
            }

            // Spike count - after the bar
            if neuron.index < anim.viewer.output_spikes.len() {
                let count = anim.viewer.output_spikes[neuron.index];
                if count > 0 {
                    draw.text(&format!("{}", count))
                        .x_y(bar_x + bar_max_width / 2.0 + 15.0, neuron.pos.1)
                        .color(rgb(0.6, 0.6, 0.7))
                        .font_size(11);
                }
            }
        }
    }

    // Title
    draw.text("gilgamesh - SNN Visualizer")
        .x_y(0.0, win.h() / 2.0 - 20.0)
        .color(rgb(0.8, 0.8, 0.9))
        .font_size(16);

    draw.to_frame(app, &frame).unwrap();
}

// Stub implementation when feature is disabled
#[cfg(not(feature = "animation"))]
#[derive(Clone, Default)]
pub struct NetworkAnimation;

#[cfg(not(feature = "animation"))]
impl NetworkAnimation {
    pub fn new(_layer_sizes: &[usize]) -> Self {
        Self
    }
    pub fn layout(&mut self, _width: f32, _height: f32) {}
    pub fn set_weights(&mut self, _weights: &[Vec<Vec<f32>>]) {}
    pub fn update_neurons(&mut self, _membranes: &[Vec<f32>], _spikes: &[Vec<bool>], _dt: f32) {}
    pub fn reset_for_sample(&mut self, _sample_idx: usize, _label: u8, _image: &[f32]) {}
}

#[cfg(not(feature = "animation"))]
pub type SharedAnimation = std::sync::Arc<std::sync::Mutex<NetworkAnimation>>;

#[cfg(not(feature = "animation"))]
pub fn create_shared_animation(_layer_sizes: &[usize]) -> SharedAnimation {
    std::sync::Arc::new(std::sync::Mutex::new(NetworkAnimation))
}

#[cfg(not(feature = "animation"))]
pub fn run_animation(_animation: SharedAnimation) {}
