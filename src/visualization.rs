//! Visualization module for gilgamesh using Rerun.io
//!
//! Provides logging of training metrics, spike rasters, and weight matrices
//! to Rerun for interactive exploration.
//!
//! Enable with the `visualization` feature:
//! ```toml
//! gilgamesh = { version = "0.1", features = ["visualization"] }
//! ```

#[cfg(feature = "visualization")]
use ndarray::Array2;

#[cfg(feature = "visualization")]
use rerun::{RecordingStream, RecordingStreamBuilder};

/// Training recorder that logs to Rerun
#[cfg(feature = "visualization")]
pub struct TrainingRecorder {
    /// Rerun recording stream
    rec: RecordingStream,
    /// Current epoch
    epoch: usize,
    /// Current batch within epoch
    batch: usize,
    /// Whether to log per-batch metrics
    log_batches: bool,
    /// Stride for spike logging (log every N timesteps)
    spike_stride: usize,
}

#[cfg(feature = "visualization")]
impl TrainingRecorder {
    /// Create a new training recorder
    ///
    /// # Arguments
    /// * `app_name` - Application name for Rerun viewer
    /// * `log_batches` - Whether to log metrics for each batch (vs just epochs)
    /// * `spike_stride` - Log spike rasters every N timesteps (1 = all)
    pub fn new(app_name: &str, log_batches: bool, spike_stride: usize) -> anyhow::Result<Self> {
        let rec = RecordingStreamBuilder::new(app_name)
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to create Rerun stream: {}", e))?;

        Ok(Self {
            rec,
            epoch: 0,
            batch: 0,
            log_batches,
            spike_stride: spike_stride.max(1),
        })
    }

    /// Create a recorder that saves to a file instead of spawning viewer
    pub fn to_file(
        app_name: &str,
        path: &str,
        log_batches: bool,
        spike_stride: usize,
    ) -> anyhow::Result<Self> {
        let rec = RecordingStreamBuilder::new(app_name)
            .save(path)
            .map_err(|e| anyhow::anyhow!("Failed to create Rerun file: {}", e))?;

        Ok(Self {
            rec,
            epoch: 0,
            batch: 0,
            log_batches,
            spike_stride: spike_stride.max(1),
        })
    }

    /// Set the current epoch
    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
        self.batch = 0;
        self.rec.set_time_sequence("epoch", epoch as i64);
    }

    /// Log training loss for current batch
    pub fn log_batch_loss(&mut self, loss: f32) -> anyhow::Result<()> {
        if !self.log_batches {
            return Ok(());
        }

        self.batch += 1;
        let global_step = self.epoch * 1000 + self.batch; // Rough global step
        self.rec.set_time_sequence("batch", global_step as i64);

        self.rec
            .log("training/batch_loss", &rerun::Scalar::new(loss as f64))
            .map_err(|e| anyhow::anyhow!("Failed to log batch loss: {}", e))?;

        Ok(())
    }

    /// Log epoch-level metrics
    pub fn log_epoch_metrics(
        &mut self,
        train_loss: f32,
        train_acc: f32,
        test_acc: f32,
        learning_rate: f32,
    ) -> anyhow::Result<()> {
        self.rec.set_time_sequence("epoch", self.epoch as i64);

        self.rec
            .log("training/loss", &rerun::Scalar::new(train_loss as f64))
            .map_err(|e| anyhow::anyhow!("Failed to log loss: {}", e))?;

        self.rec
            .log(
                "training/train_accuracy",
                &rerun::Scalar::new(train_acc as f64),
            )
            .map_err(|e| anyhow::anyhow!("Failed to log train acc: {}", e))?;

        self.rec
            .log(
                "training/test_accuracy",
                &rerun::Scalar::new(test_acc as f64),
            )
            .map_err(|e| anyhow::anyhow!("Failed to log test acc: {}", e))?;

        self.rec
            .log(
                "training/learning_rate",
                &rerun::Scalar::new(learning_rate as f64),
            )
            .map_err(|e| anyhow::anyhow!("Failed to log lr: {}", e))?;

        Ok(())
    }

    /// Log weight matrix as heatmap image
    pub fn log_weights(&self, name: &str, weights: &Array2<f32>) -> anyhow::Result<()> {
        let (rows, cols) = weights.dim();

        // Normalize weights to 0-255 for visualization
        let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_w - min_w).max(1e-6);

        // Create grayscale image data
        let pixels: Vec<u8> = weights
            .iter()
            .map(|&w| ((w - min_w) / range * 255.0) as u8)
            .collect();

        // Log as tensor (simpler than Image for grayscale data)
        let tensor = rerun::TensorData::new(
            vec![
                rerun::TensorDimension::height(rows as u64),
                rerun::TensorDimension::width(cols as u64),
            ],
            rerun::TensorBuffer::U8(pixels.into()),
        );

        self.rec
            .log(format!("weights/{}", name), &rerun::Tensor::new(tensor))
            .map_err(|e| anyhow::anyhow!("Failed to log weights: {}", e))?;

        // Also log weight statistics
        self.rec
            .log(
                format!("weights/{}/mean", name),
                &rerun::Scalar::new(weights.mean().unwrap_or(0.0) as f64),
            )
            .ok();
        self.rec
            .log(
                format!("weights/{}/std", name),
                &rerun::Scalar::new(weights.std(0.0) as f64),
            )
            .ok();
        self.rec
            .log(
                format!("weights/{}/max", name),
                &rerun::Scalar::new(max_w as f64),
            )
            .ok();
        self.rec
            .log(
                format!("weights/{}/min", name),
                &rerun::Scalar::new(min_w as f64),
            )
            .ok();

        Ok(())
    }

    /// Log spike raster for a batch
    ///
    /// # Arguments
    /// * `name` - Layer name (e.g., "hidden", "output")
    /// * `spikes` - Spike tensor [batch, neurons] accumulated over timesteps
    /// * `sample_idx` - Which sample in the batch to visualize (default 0)
    pub fn log_spike_counts(
        &self,
        name: &str,
        spikes: &Array2<f32>,
        sample_idx: usize,
    ) -> anyhow::Result<()> {
        if sample_idx >= spikes.shape()[0] {
            return Ok(());
        }

        let spike_counts: Vec<f64> = spikes.row(sample_idx).iter().map(|&s| s as f64).collect();

        // Log as bar chart (using individual scalars per neuron)
        for (i, &count) in spike_counts.iter().enumerate() {
            self.rec
                .log(
                    format!("spikes/{}/neuron_{:03}", name, i),
                    &rerun::Scalar::new(count),
                )
                .ok();
        }

        Ok(())
    }

    /// Log spike raster over time (more detailed)
    ///
    /// # Arguments
    /// * `name` - Layer name
    /// * `spike_history` - Vec of [batch, neurons] arrays, one per timestep
    /// * `sample_idx` - Which sample to visualize
    pub fn log_spike_raster(
        &self,
        name: &str,
        spike_history: &[Array2<f32>],
        sample_idx: usize,
    ) -> anyhow::Result<()> {
        if spike_history.is_empty() {
            return Ok(());
        }

        let num_neurons = spike_history[0].shape()[1];
        let num_steps = spike_history.len();

        // Create binary image: rows = neurons, cols = time
        let mut raster: Vec<u8> = vec![0; num_neurons * num_steps];

        for (t, spikes) in spike_history.iter().enumerate() {
            if t % self.spike_stride != 0 {
                continue;
            }
            if sample_idx >= spikes.shape()[0] {
                continue;
            }

            for (n, &spike) in spikes.row(sample_idx).iter().enumerate() {
                if spike > 0.5 {
                    raster[n * num_steps + t] = 255;
                }
            }
        }

        // Log as tensor (rows = neurons, cols = time)
        let tensor = rerun::TensorData::new(
            vec![
                rerun::TensorDimension::height(num_neurons as u64),
                rerun::TensorDimension::width(num_steps as u64),
            ],
            rerun::TensorBuffer::U8(raster.into()),
        );

        self.rec
            .log(
                format!("spikes/{}_raster", name),
                &rerun::Tensor::new(tensor),
            )
            .map_err(|e| anyhow::anyhow!("Failed to log spike raster: {}", e))?;

        Ok(())
    }

    /// Log membrane potential trace for a neuron
    pub fn log_membrane_trace(
        &self,
        name: &str,
        membrane_history: &[f32],
        neuron_idx: usize,
    ) -> anyhow::Result<()> {
        for (t, &mem) in membrane_history.iter().enumerate() {
            self.rec.set_time_sequence("timestep", t as i64);
            self.rec
                .log(
                    format!("membrane/{}/neuron_{:03}", name, neuron_idx),
                    &rerun::Scalar::new(mem as f64),
                )
                .ok();
        }
        Ok(())
    }

    /// Log a text annotation (useful for marking events)
    pub fn log_text(&self, path: &str, text: &str) -> anyhow::Result<()> {
        self.rec
            .log(path, &rerun::TextLog::new(text))
            .map_err(|e| anyhow::anyhow!("Failed to log text: {}", e))?;
        Ok(())
    }

    /// Log network architecture info
    pub fn log_architecture(
        &self,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        mode: &str,
    ) -> anyhow::Result<()> {
        let arch_text = format!(
            "Architecture: {} -> {} -> {} ({})",
            input_size, hidden_size, output_size, mode
        );
        self.log_text("architecture", &arch_text)
    }
}

/// No-op recorder for when visualization is disabled
#[cfg(not(feature = "visualization"))]
pub struct TrainingRecorder;

#[cfg(not(feature = "visualization"))]
impl TrainingRecorder {
    pub fn new(_app_name: &str, _log_batches: bool, _spike_stride: usize) -> anyhow::Result<Self> {
        Ok(Self)
    }
    pub fn to_file(
        _app_name: &str,
        _path: &str,
        _log_batches: bool,
        _spike_stride: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self)
    }
    pub fn set_epoch(&mut self, _epoch: usize) {}
    pub fn log_batch_loss(&mut self, _loss: f32) -> anyhow::Result<()> {
        Ok(())
    }
    pub fn log_epoch_metrics(
        &mut self,
        _train_loss: f32,
        _train_acc: f32,
        _test_acc: f32,
        _learning_rate: f32,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    pub fn log_weights(&self, _name: &str, _weights: &ndarray::Array2<f32>) -> anyhow::Result<()> {
        Ok(())
    }
    pub fn log_spike_counts(
        &self,
        _name: &str,
        _spikes: &ndarray::Array2<f32>,
        _sample_idx: usize,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    pub fn log_spike_raster(
        &self,
        _name: &str,
        _spike_history: &[ndarray::Array2<f32>],
        _sample_idx: usize,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    pub fn log_membrane_trace(
        &self,
        _name: &str,
        _membrane_history: &[f32],
        _neuron_idx: usize,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    pub fn log_text(&self, _path: &str, _text: &str) -> anyhow::Result<()> {
        Ok(())
    }
    pub fn log_architecture(
        &self,
        _input_size: usize,
        _hidden_size: usize,
        _output_size: usize,
        _mode: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "visualization")]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_weight_normalization() {
        // Test that weight normalization works correctly
        let weights =
            Array2::from_shape_vec((3, 3), vec![-1.0, 0.0, 1.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
                .unwrap();

        let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        assert!((min_w - (-1.0)).abs() < 1e-6);
        assert!((max_w - 1.0).abs() < 1e-6);
    }
}
