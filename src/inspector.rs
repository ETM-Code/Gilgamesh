//! Visual test runner for inspecting single samples
//!
//! Provides an interactive egui window for viewing individual test samples,
//! running inference, and visualizing output neuron activations.
//!
//! Enable with the `dashboard` feature:
//! ```toml
//! gilgamesh = { version = "0.1", features = ["dashboard"] }
//! ```

#[cfg(feature = "dashboard")]
use eframe::egui;

#[cfg(feature = "dashboard")]
use egui_plot::{Bar, BarChart, GridMark, Plot};

#[cfg(feature = "dashboard")]
use rand::SeedableRng;

#[cfg(feature = "dashboard")]
use rand_xoshiro::Xoshiro256PlusPlus;

#[cfg(feature = "dashboard")]
use crate::checkpoint::Checkpoint;
#[cfg(feature = "dashboard")]
use crate::data::MnistDataset;
#[cfg(feature = "dashboard")]
use crate::network::Network;

/// Result of inspecting a single sample
#[cfg(feature = "dashboard")]
#[derive(Clone, Debug)]
pub struct InspectionResult {
    /// Input image pixels [49] for 7x7
    pub input_pixels: Vec<f32>,
    /// Output spike counts per class [10]
    pub spike_counts: Vec<f32>,
    /// Predicted class (argmax of spike_counts)
    pub predicted: usize,
    /// Ground truth label
    pub ground_truth: usize,
    /// Whether prediction was correct
    pub correct: bool,
    /// Sample index in test set
    pub sample_index: usize,
}

/// The egui inspector application
#[cfg(feature = "dashboard")]
pub struct InspectorApp {
    /// Loaded network
    network: Network,
    /// Test dataset
    dataset: MnistDataset,
    /// Number of timesteps for inference
    num_steps: usize,
    /// Current inspection result
    current_result: Option<InspectionResult>,
    /// RNG for random sample selection
    rng: Xoshiro256PlusPlus,
    /// Statistics
    total_inspected: usize,
    correct_count: usize,
}

#[cfg(feature = "dashboard")]
impl InspectorApp {
    /// Create a new inspector app
    pub fn new(network: Network, dataset: MnistDataset, num_steps: usize, seed: u64) -> Self {
        Self {
            network,
            dataset,
            num_steps,
            current_result: None,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            total_inspected: 0,
            correct_count: 0,
        }
    }

    /// Load from checkpoint file
    pub fn from_checkpoint(
        checkpoint_path: &str,
        data_dir: &str,
        num_steps: usize,
        seed: u64,
    ) -> anyhow::Result<Self> {
        use anyhow::Context;

        let checkpoint = Checkpoint::load(checkpoint_path)
            .with_context(|| format!("Failed to load checkpoint from {}", checkpoint_path))?;

        let network = checkpoint.to_network()
            .with_context(|| "Failed to reconstruct network from checkpoint")?;

        let dataset = MnistDataset::load(data_dir)
            .with_context(|| format!("Failed to load dataset from {}", data_dir))?;

        Ok(Self::new(network, dataset, num_steps, seed))
    }

    /// Run the inspector as a native application
    pub fn run(self) -> eframe::Result<()> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([900.0, 600.0])
                .with_title("gilgamesh Inspector"),
            ..Default::default()
        };

        eframe::run_native(
            "gilgamesh Inspector",
            options,
            Box::new(|cc| {
                cc.egui_ctx.set_visuals(egui::Visuals::dark());
                Ok(Box::new(self))
            }),
        )
    }

    /// Inspect a random sample from the test set
    fn inspect_random(&mut self) {
        use rand::Rng;

        let sample_index = self.rng.gen_range(0..self.dataset.test_len());
        self.inspect_sample(sample_index);
    }

    /// Inspect a specific sample
    fn inspect_sample(&mut self, sample_index: usize) {

        // Get sample from dataset
        let (images, labels) = self.dataset.get_test_batch(&[sample_index]);
        let input = images.row(0).to_owned();
        let ground_truth = labels[0];

        // Run forward pass
        let input_2d = input.clone().insert_axis(ndarray::Axis(0));
        let (spike_count, _, _) = self.network.forward(&input_2d, self.num_steps);

        // Get prediction
        let spike_counts: Vec<f32> = spike_count.row(0).to_vec();
        let predicted = spike_counts
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let correct = predicted == ground_truth;

        // Update statistics
        self.total_inspected += 1;
        if correct {
            self.correct_count += 1;
        }

        self.current_result = Some(InspectionResult {
            input_pixels: input.to_vec(),
            spike_counts,
            predicted,
            ground_truth,
            correct,
            sample_index,
        });
    }

    /// Render the input image as a texture
    fn render_image(&self, ui: &mut egui::Ui, pixels: &[f32]) {
        let size = 7;
        let scale = 20; // Scale up for visibility

        // Create color image
        let mut pixels_u8 = Vec::with_capacity(size * size);
        for &p in pixels {
            // Denormalize: MNIST was normalized with mean=0.1307, std=0.3081
            let denorm = (p * 0.3081 + 0.1307).clamp(0.0, 1.0);
            let gray = (denorm * 255.0) as u8;
            pixels_u8.push(egui::Color32::from_gray(gray));
        }

        // Draw as colored rectangles
        let (response, painter) = ui.allocate_painter(
            egui::Vec2::new((size * scale) as f32, (size * scale) as f32),
            egui::Sense::hover(),
        );

        let rect = response.rect;
        let cell_size = rect.width() / size as f32;

        for y in 0..size {
            for x in 0..size {
                let idx = y * size + x;
                let color = pixels_u8[idx];
                let cell_rect = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.min.x + x as f32 * cell_size,
                        rect.min.y + y as f32 * cell_size,
                    ),
                    egui::Vec2::splat(cell_size),
                );
                painter.rect_filled(cell_rect, 0.0, color);
            }
        }
    }
}

#[cfg(feature = "dashboard")]
impl eframe::App for InspectorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top panel with controls
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("gilgamesh Inspector");
                ui.separator();

                if ui.button("Next Sample").clicked() {
                    self.inspect_random();
                }

                ui.separator();

                ui.label(format!(
                    "Accuracy: {}/{} ({:.1}%)",
                    self.correct_count,
                    self.total_inspected,
                    if self.total_inspected > 0 {
                        100.0 * self.correct_count as f64 / self.total_inspected as f64
                    } else {
                        0.0
                    }
                ));
            });
        });

        // Main content
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref result) = self.current_result {
                ui.horizontal(|ui| {
                    // Left side: Image
                    ui.vertical(|ui| {
                        ui.heading("Input Image");
                        ui.add_space(10.0);

                        self.render_image(ui, &result.input_pixels);

                        ui.add_space(10.0);
                        ui.label(format!("Sample #{}", result.sample_index));
                        ui.label(format!("True Label: {}", result.ground_truth));

                        if result.correct {
                            ui.colored_label(
                                egui::Color32::GREEN,
                                format!("Predicted: {} ✓", result.predicted),
                            );
                        } else {
                            ui.colored_label(
                                egui::Color32::RED,
                                format!("Predicted: {} ✗", result.predicted),
                            );
                        }
                    });

                    ui.separator();

                    // Right side: Bar chart of spike counts
                    ui.vertical(|ui| {
                        ui.heading("Output Spike Counts");
                        ui.add_space(10.0);

                        // Create bars with colors
                        let bars: Vec<Bar> = result
                            .spike_counts
                            .iter()
                            .enumerate()
                            .map(|(i, &count)| {
                                let color = if i == result.predicted && result.correct {
                                    // Correct prediction - green
                                    egui::Color32::from_rgb(80, 220, 100)
                                } else if i == result.predicted && !result.correct {
                                    // Wrong prediction - red
                                    egui::Color32::from_rgb(255, 80, 80)
                                } else if i == result.ground_truth {
                                    // True label (missed) - blue
                                    egui::Color32::from_rgb(100, 180, 255)
                                } else {
                                    // Other - gray
                                    egui::Color32::from_rgb(128, 128, 128)
                                };

                                Bar::new(i as f64, count as f64)
                                    .width(0.8)
                                    .fill(color)
                            })
                            .collect();

                        let chart = BarChart::new(bars).name("Spike Counts");

                        // Custom grid spacer to show marks at each digit 0-9
                        let x_grid_spacer = |_input: egui_plot::GridInput| {
                            (0..=9)
                                .map(|i| GridMark { value: i as f64, step_size: 1.0 })
                                .collect()
                        };

                        // Custom formatter to show digit labels
                        let x_formatter = |mark: GridMark, _range: &std::ops::RangeInclusive<f64>| {
                            let digit = mark.value.round() as i32;
                            if digit >= 0 && digit <= 9 {
                                format!("{}", digit)
                            } else {
                                String::new()
                            }
                        };

                        Plot::new("spike_counts")
                            .height(280.0)
                            .allow_zoom(false)
                            .allow_drag(false)
                            .allow_scroll(false)
                            .show_axes([true, true])
                            .include_x(-0.5)
                            .include_x(9.5)
                            .include_y(0.0)
                            .x_grid_spacer(x_grid_spacer)
                            .x_axis_formatter(x_formatter)
                            .show(ui, |plot_ui| {
                                plot_ui.bar_chart(chart);
                            });

                        ui.add_space(10.0);

                        // Legend
                        ui.horizontal(|ui| {
                            ui.colored_label(egui::Color32::from_rgb(80, 220, 100), "■");
                            ui.label("Correct prediction");
                        });
                        ui.horizontal(|ui| {
                            ui.colored_label(egui::Color32::from_rgb(255, 80, 80), "■");
                            ui.label("Wrong prediction");
                        });
                        ui.horizontal(|ui| {
                            ui.colored_label(egui::Color32::from_rgb(100, 180, 255), "■");
                            ui.label("True label (when missed)");
                        });

                        ui.add_space(20.0);

                        // Probability table
                        ui.heading("Spike Count Details");
                        egui::Grid::new("spike_grid")
                            .num_columns(2)
                            .spacing([40.0, 4.0])
                            .show(ui, |ui| {
                                for (i, &count) in result.spike_counts.iter().enumerate() {
                                    let label = if i == result.predicted {
                                        format!("Digit {} (predicted)", i)
                                    } else if i == result.ground_truth {
                                        format!("Digit {} (true)", i)
                                    } else {
                                        format!("Digit {}", i)
                                    };
                                    ui.label(label);
                                    ui.label(format!("{:.0} spikes", count));
                                    ui.end_row();
                                }
                            });
                    });
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.heading("Click 'Next Sample' to start inspecting");
                });
            }
        });

        // Auto-inspect first sample
        if self.current_result.is_none() {
            self.inspect_random();
        }
    }
}

// Stub for when feature is not enabled
#[cfg(not(feature = "dashboard"))]
pub struct InspectorApp;

#[cfg(not(feature = "dashboard"))]
impl InspectorApp {
    pub fn from_checkpoint(
        _checkpoint_path: &str,
        _data_dir: &str,
        _num_steps: usize,
        _seed: u64,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("Inspector requires the 'dashboard' feature. Rebuild with --features dashboard")
    }

    pub fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        Err("Inspector requires the 'dashboard' feature".into())
    }
}
