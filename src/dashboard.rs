//! Interactive training dashboard using egui
//!
//! Provides a native GUI for monitoring and controlling training.
//!
//! Enable with the `dashboard` feature:
//! ```toml
//! gilgamesh = { version = "0.1", features = ["dashboard"] }
//! ```

#[cfg(feature = "dashboard")]
use eframe::egui;

#[cfg(feature = "dashboard")]
use egui_plot::{Line, Plot, PlotPoints};

#[cfg(feature = "dashboard")]
use std::sync::{Arc, Mutex};

/// Training metrics collected during training
#[cfg(feature = "dashboard")]
#[derive(Clone, Default)]
pub struct TrainingMetrics {
    /// Loss per epoch
    pub losses: Vec<f64>,
    /// Training accuracy per epoch
    pub train_accuracies: Vec<f64>,
    /// Test accuracy per epoch
    pub test_accuracies: Vec<f64>,
    /// Learning rate per epoch
    pub learning_rates: Vec<f64>,
    /// Current epoch
    pub current_epoch: usize,
    /// Total epochs
    pub total_epochs: usize,
    /// Is training running?
    pub is_training: bool,
    /// Is training complete?
    pub is_complete: bool,
    /// Network architecture description
    pub architecture: String,
    /// Best test accuracy so far
    pub best_test_acc: f64,
}

#[cfg(feature = "dashboard")]
impl TrainingMetrics {
    pub fn new(total_epochs: usize, architecture: String) -> Self {
        Self {
            total_epochs,
            architecture,
            ..Default::default()
        }
    }

    pub fn record_epoch(&mut self, loss: f64, train_acc: f64, test_acc: f64, lr: f64) {
        self.current_epoch += 1;
        self.losses.push(loss);
        self.train_accuracies.push(train_acc);
        self.test_accuracies.push(test_acc);
        self.learning_rates.push(lr);
        if test_acc > self.best_test_acc {
            self.best_test_acc = test_acc;
        }
    }
}

/// Shared state between training thread and dashboard
#[cfg(feature = "dashboard")]
pub type SharedMetrics = Arc<Mutex<TrainingMetrics>>;

/// Create shared metrics for dashboard
#[cfg(feature = "dashboard")]
pub fn create_shared_metrics(total_epochs: usize, architecture: String) -> SharedMetrics {
    Arc::new(Mutex::new(TrainingMetrics::new(total_epochs, architecture)))
}

/// The egui dashboard application
#[cfg(feature = "dashboard")]
pub struct DashboardApp {
    metrics: SharedMetrics,
}

#[cfg(feature = "dashboard")]
impl DashboardApp {
    pub fn new(metrics: SharedMetrics) -> Self {
        Self { metrics }
    }

    /// Run the dashboard as a native application
    pub fn run(metrics: SharedMetrics) -> eframe::Result<()> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1200.0, 800.0])
                .with_title("gilgamesh Training Dashboard"),
            ..Default::default()
        };

        eframe::run_native(
            "gilgamesh Dashboard",
            options,
            Box::new(|cc| {
                // Enable dark mode
                cc.egui_ctx.set_visuals(egui::Visuals::dark());
                Ok(Box::new(DashboardApp::new(metrics)))
            }),
        )
    }
}

#[cfg(feature = "dashboard")]
impl eframe::App for DashboardApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request repaint to update with new training data
        ctx.request_repaint_after(std::time::Duration::from_millis(100));

        let metrics = self.metrics.lock().unwrap().clone();

        // Top panel with status
        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("gilgamesh Training Dashboard");
                ui.separator();

                if metrics.is_complete {
                    ui.colored_label(egui::Color32::GREEN, "✓ Training Complete");
                } else if metrics.is_training {
                    ui.colored_label(egui::Color32::YELLOW, "● Training...");
                    ui.spinner();
                } else {
                    ui.colored_label(egui::Color32::GRAY, "○ Waiting");
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!(
                        "Epoch {}/{}",
                        metrics.current_epoch, metrics.total_epochs
                    ));
                });
            });
        });

        // Left panel with metrics
        egui::SidePanel::left("metrics_panel")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Metrics");
                ui.separator();

                egui::Grid::new("metrics_grid")
                    .num_columns(2)
                    .spacing([20.0, 8.0])
                    .show(ui, |ui| {
                        ui.label("Architecture:");
                        ui.label(&metrics.architecture);
                        ui.end_row();

                        ui.label("Current Epoch:");
                        ui.label(format!(
                            "{}/{}",
                            metrics.current_epoch, metrics.total_epochs
                        ));
                        ui.end_row();

                        if let Some(&loss) = metrics.losses.last() {
                            ui.label("Loss:");
                            ui.label(format!("{:.4}", loss));
                            ui.end_row();
                        }

                        if let Some(&acc) = metrics.train_accuracies.last() {
                            ui.label("Train Accuracy:");
                            ui.colored_label(
                                egui::Color32::from_rgb(80, 220, 100),
                                format!("{:.2}%", acc),
                            );
                            ui.end_row();
                        }

                        if let Some(&acc) = metrics.test_accuracies.last() {
                            ui.label("Test Accuracy:");
                            ui.colored_label(
                                egui::Color32::from_rgb(100, 180, 255),
                                format!("{:.2}%", acc),
                            );
                            ui.end_row();
                        }

                        ui.label("Best Test Acc:");
                        ui.colored_label(
                            egui::Color32::GOLD,
                            format!("{:.2}%", metrics.best_test_acc),
                        );
                        ui.end_row();

                        if let Some(&lr) = metrics.learning_rates.last() {
                            ui.label("Learning Rate:");
                            ui.label(format!("{:.6}", lr));
                            ui.end_row();
                        }
                    });

                ui.separator();
                ui.heading("Progress");

                let progress = if metrics.total_epochs > 0 {
                    metrics.current_epoch as f32 / metrics.total_epochs as f32
                } else {
                    0.0
                };

                let progress_bar = egui::ProgressBar::new(progress)
                    .text(format!("{:.0}%", progress * 100.0))
                    .animate(metrics.is_training);
                ui.add(progress_bar);
            });

        // Central panel with plots
        egui::CentralPanel::default().show(ctx, |ui| {
            // Use a grid layout for plots
            egui::Grid::new("plots_grid")
                .num_columns(2)
                .spacing([10.0, 10.0])
                .show(ui, |ui| {
                    // Loss plot
                    ui.group(|ui| {
                        ui.heading("Loss");
                        let loss_points: PlotPoints = metrics
                            .losses
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64 + 1.0, v])
                            .collect();

                        Plot::new("loss_plot")
                            .height(300.0)
                            .width(500.0)
                            .allow_zoom(true)
                            .allow_drag(true)
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(loss_points)
                                        .color(egui::Color32::from_rgb(255, 120, 120))
                                        .width(2.0)
                                        .name("Loss"),
                                );
                            });
                    });

                    // Accuracy plot
                    ui.group(|ui| {
                        ui.heading("Accuracy");
                        let train_points: PlotPoints = metrics
                            .train_accuracies
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64 + 1.0, v])
                            .collect();

                        let test_points: PlotPoints = metrics
                            .test_accuracies
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64 + 1.0, v])
                            .collect();

                        Plot::new("accuracy_plot")
                            .height(300.0)
                            .width(500.0)
                            .allow_zoom(true)
                            .allow_drag(true)
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(train_points)
                                        .color(egui::Color32::from_rgb(80, 220, 100))
                                        .width(2.0)
                                        .name("Train"),
                                );
                                plot_ui.line(
                                    Line::new(test_points)
                                        .color(egui::Color32::from_rgb(100, 180, 255))
                                        .width(2.0)
                                        .name("Test"),
                                );
                            });
                    });
                    ui.end_row();

                    // Learning rate plot
                    ui.group(|ui| {
                        ui.heading("Learning Rate");
                        let lr_points: PlotPoints = metrics
                            .learning_rates
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64 + 1.0, v])
                            .collect();

                        Plot::new("lr_plot")
                            .height(250.0)
                            .width(500.0)
                            .allow_zoom(true)
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(lr_points)
                                        .color(egui::Color32::from_rgb(220, 160, 255))
                                        .width(2.0)
                                        .name("LR"),
                                );
                            });
                    });

                    // Info panel
                    ui.group(|ui| {
                        ui.heading("Training Info");
                        ui.separator();

                        if metrics.losses.len() >= 2 {
                            let loss_delta = metrics.losses.last().unwrap()
                                - metrics.losses[metrics.losses.len() - 2];
                            let color = if loss_delta < 0.0 {
                                egui::Color32::GREEN
                            } else {
                                egui::Color32::RED
                            };
                            ui.horizontal(|ui| {
                                ui.label("Loss Δ:");
                                ui.colored_label(color, format!("{:+.4}", loss_delta));
                            });
                        }

                        if metrics.test_accuracies.len() >= 2 {
                            let acc_delta = metrics.test_accuracies.last().unwrap()
                                - metrics.test_accuracies[metrics.test_accuracies.len() - 2];
                            let color = if acc_delta > 0.0 {
                                egui::Color32::GREEN
                            } else {
                                egui::Color32::RED
                            };
                            ui.horizontal(|ui| {
                                ui.label("Test Acc Δ:");
                                ui.colored_label(color, format!("{:+.2}%", acc_delta));
                            });
                        }

                        ui.separator();
                        ui.label("Controls:");
                        ui.horizontal(|ui| {
                            ui.label("Zoom: Scroll");
                            ui.label("Pan: Drag");
                        });
                    });
                });
        });
    }
}

// Stub implementation when feature is disabled
#[cfg(not(feature = "dashboard"))]
pub struct TrainingMetrics;

#[cfg(not(feature = "dashboard"))]
impl TrainingMetrics {
    pub fn new(_total_epochs: usize, _architecture: String) -> Self {
        Self
    }
    pub fn record_epoch(&mut self, _loss: f64, _train_acc: f64, _test_acc: f64, _lr: f64) {}
}

#[cfg(not(feature = "dashboard"))]
impl Default for TrainingMetrics {
    fn default() -> Self {
        Self
    }
}

#[cfg(not(feature = "dashboard"))]
impl Clone for TrainingMetrics {
    fn clone(&self) -> Self {
        Self
    }
}
