//! Data loading utilities for MNIST
//!
//! Loads MNIST dataset and provides batching utilities.
//! Also provides input encoding for SNN (rate-coded vs temporal).

use anyhow::Result;
use mnist::MnistBuilder;
use ndarray::Array2;

/// MNIST dataset with configurable downsampling
pub struct MnistDataset {
    pub train_images: Array2<f32>,
    pub train_labels: Vec<usize>,
    pub test_images: Array2<f32>,
    pub test_labels: Vec<usize>,
    pub image_width: usize,
    pub image_height: usize,
}

impl MnistDataset {
    /// Load MNIST dataset from the specified directory
    ///
    /// Downloads if not present. Downsamples to 6x6 by default.
    pub fn load(data_dir: &str) -> Result<Self> {
        Self::load_with_size(data_dir, 6)
    }

    /// Load with custom square image size
    pub fn load_with_size(data_dir: &str, target_size: usize) -> Result<Self> {
        Self::load_with_dimensions(data_dir, target_size, target_size)
    }

    /// Load with custom width x height (non-square supported)
    pub fn load_with_dimensions(data_dir: &str, width: usize, height: usize) -> Result<Self> {
        // Create data directory if it doesn't exist
        std::fs::create_dir_all(data_dir)?;

        let mnist = MnistBuilder::new()
            .base_path(data_dir)
            .label_format_digit()
            .training_set_length(60_000)
            .test_set_length(10_000)
            .finalize();

        // Convert and normalize training images
        let train_images = Self::process_images(&mnist.trn_img, 60_000, 28, width, height)?;
        let train_labels: Vec<usize> = mnist.trn_lbl.iter().map(|&l| l as usize).collect();

        // Convert and normalize test images
        let test_images = Self::process_images(&mnist.tst_img, 10_000, 28, width, height)?;
        let test_labels: Vec<usize> = mnist.tst_lbl.iter().map(|&l| l as usize).collect();

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            image_width: width,
            image_height: height,
        })
    }

    /// Process raw images: normalize and downsample to target width x height
    fn process_images(
        raw: &[u8],
        num_images: usize,
        original_size: usize,
        target_width: usize,
        target_height: usize,
    ) -> Result<Array2<f32>> {
        let pixels_per_image = original_size * original_size;
        let target_pixels = target_width * target_height;

        let mut images = Array2::zeros((num_images, target_pixels));

        // Calculate scale factors for each dimension
        let scale_x = original_size as f32 / target_width as f32;
        let scale_y = original_size as f32 / target_height as f32;

        for i in 0..num_images {
            let start = i * pixels_per_image;
            let img_data = &raw[start..start + pixels_per_image];

            // Downsample by averaging (supports non-integer scales)
            for target_y in 0..target_height {
                for target_x in 0..target_width {
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    // Source region bounds
                    let source_y_start = (target_y as f32 * scale_y) as usize;
                    let source_y_end = ((target_y + 1) as f32 * scale_y).ceil() as usize;
                    let source_x_start = (target_x as f32 * scale_x) as usize;
                    let source_x_end = ((target_x + 1) as f32 * scale_x).ceil() as usize;

                    for source_y in source_y_start..source_y_end.min(original_size) {
                        for source_x in source_x_start..source_x_end.min(original_size) {
                            sum += img_data[source_y * original_size + source_x] as f32;
                            count += 1;
                        }
                    }

                    let pixel_idx = target_y * target_width + target_x;
                    // Normalize to [0, 1] then apply MNIST normalization
                    let normalized = if count > 0 {
                        (sum / count as f32) / 255.0
                    } else {
                        0.0
                    };
                    // MNIST normalization: (x - 0.1307) / 0.3081
                    images[[i, pixel_idx]] = (normalized - 0.1307) / 0.3081;
                }
            }
        }

        Ok(images)
    }

    /// Get a batch of training data
    pub fn get_train_batch(&self, indices: &[usize]) -> (Array2<f32>, Vec<usize>) {
        let batch_size = indices.len();
        let features = self.feature_dim();

        let mut batch_images = Array2::zeros((batch_size, features));
        let mut batch_labels = Vec::with_capacity(batch_size);

        for (i, &idx) in indices.iter().enumerate() {
            batch_images.row_mut(i).assign(&self.train_images.row(idx));
            batch_labels.push(self.train_labels[idx]);
        }

        (batch_images, batch_labels)
    }

    /// Get a batch of test data
    pub fn get_test_batch(&self, indices: &[usize]) -> (Array2<f32>, Vec<usize>) {
        let batch_size = indices.len();
        let features = self.feature_dim();

        let mut batch_images = Array2::zeros((batch_size, features));
        let mut batch_labels = Vec::with_capacity(batch_size);

        for (i, &idx) in indices.iter().enumerate() {
            batch_images.row_mut(i).assign(&self.test_images.row(idx));
            batch_labels.push(self.test_labels[idx]);
        }

        (batch_images, batch_labels)
    }

    /// Number of training samples
    pub fn train_len(&self) -> usize {
        self.train_labels.len()
    }

    /// Number of test samples
    pub fn test_len(&self) -> usize {
        self.test_labels.len()
    }

    /// Feature dimension (width × height)
    pub fn feature_dim(&self) -> usize {
        self.image_width * self.image_height
    }

    /// Image size for square images (returns width, assumes square)
    pub fn image_size(&self) -> usize {
        self.image_width
    }
}

/// Batch iterator for training
pub struct BatchIterator<'a> {
    dataset: &'a MnistDataset,
    indices: Vec<usize>,
    batch_size: usize,
    current: usize,
    is_train: bool,
}

impl<'a> BatchIterator<'a> {
    pub fn new(
        dataset: &'a MnistDataset,
        batch_size: usize,
        shuffle: bool,
        is_train: bool,
    ) -> Self {
        let len = if is_train {
            dataset.train_len()
        } else {
            dataset.test_len()
        };
        let mut indices: Vec<usize> = (0..len).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            indices.shuffle(&mut thread_rng());
        }

        Self {
            dataset,
            indices,
            batch_size,
            current: 0,
            is_train,
        }
    }

    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Array2<f32>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current..end];
        self.current = end;

        if self.is_train {
            Some(self.dataset.get_train_batch(batch_indices))
        } else {
            Some(self.dataset.get_test_batch(batch_indices))
        }
    }
}

/// Input encoding type for SNN
#[derive(Clone, Debug, PartialEq)]
pub enum InputEncodingType {
    /// Rate-coded: same input presented at all timesteps (snnTorch-style)
    RateCoded,
    /// Temporal: rows presented sequentially (hardware-like)
    Temporal {
        /// Time per row in seconds
        row_spacing: f32,
        /// Pulse width as fraction of row spacing (0.0-1.0)
        pulse_width: f32,
    },
}

impl Default for InputEncodingType {
    fn default() -> Self {
        InputEncodingType::RateCoded
    }
}

/// Input encoder for converting static images to time-varying SNN input
pub struct InputEncoder {
    /// Encoding type
    encoding_type: InputEncodingType,
    /// Image dimensions (assumed square: image_size x image_size)
    image_size: usize,
    /// Integration timestep (for temporal encoding timing)
    dt: f32,
}

impl InputEncoder {
    /// Create a new input encoder
    ///
    /// Args:
    ///   encoding_type: Rate-coded or temporal encoding
    ///   image_size: Side length of square image (e.g., 7 for 7x7)
    ///   dt: Integration timestep in seconds
    pub fn new(encoding_type: InputEncodingType, image_size: usize, dt: f32) -> Self {
        Self {
            encoding_type,
            image_size,
            dt,
        }
    }

    /// Create a rate-coded encoder (default snnTorch behavior)
    pub fn rate_coded(image_size: usize) -> Self {
        Self::new(InputEncodingType::RateCoded, image_size, 0.001)
    }

    /// Create a temporal encoder with row-by-row presentation
    pub fn temporal(image_size: usize, row_spacing: f32, pulse_width: f32, dt: f32) -> Self {
        Self::new(
            InputEncodingType::Temporal {
                row_spacing,
                pulse_width,
            },
            image_size,
            dt,
        )
    }

    /// Get the number of timesteps needed for this encoding
    ///
    /// For rate-coded: returns the requested num_steps
    /// For temporal: returns enough steps to present all rows
    pub fn timesteps_needed(&self, requested_steps: usize) -> usize {
        match &self.encoding_type {
            InputEncodingType::RateCoded => requested_steps,
            InputEncodingType::Temporal { row_spacing, .. } => {
                // Need enough time to present all rows
                let total_time = row_spacing * self.image_size as f32;
                let steps = (total_time / self.dt).ceil() as usize;
                steps.max(requested_steps)
            }
        }
    }

    /// Get the output dimension for this encoder
    ///
    /// For rate-coded: returns image_size^2 (all pixels)
    /// For temporal: returns image_size (one row of pixels)
    pub fn output_dim(&self) -> usize {
        match &self.encoding_type {
            InputEncodingType::RateCoded => self.image_size * self.image_size,
            InputEncodingType::Temporal { .. } => self.image_size,
        }
    }

    /// Encode input for a specific timestep
    ///
    /// Args:
    ///   input: Static input [batch, features] where features = image_size^2
    ///   timestep: Current timestep (0-indexed)
    ///
    /// Returns:
    ///   For rate-coded: [batch, image_size^2] - full image
    ///   For temporal: [batch, image_size] - just the active row (6 pixels)
    pub fn encode_timestep(&self, input: &Array2<f32>, timestep: usize) -> Array2<f32> {
        match &self.encoding_type {
            InputEncodingType::RateCoded => {
                // Same input at every timestep
                input.clone()
            }
            InputEncodingType::Temporal {
                row_spacing,
                pulse_width,
            } => {
                // Determine which row is active at this timestep
                let current_time = timestep as f32 * self.dt;
                let row_idx = (current_time / row_spacing).floor() as usize;

                // Check if we're within the pulse window for this row
                let time_in_row = current_time - (row_idx as f32 * row_spacing);
                let pulse_duration = row_spacing * pulse_width;
                let is_active = time_in_row < pulse_duration;

                let batch_size = input.shape()[0];
                // Output is just image_size (6 pixels for one row)
                let mut encoded = Array2::zeros((batch_size, self.image_size));

                if row_idx < self.image_size && is_active {
                    // Extract the active row (6 pixels)
                    let row_start = row_idx * self.image_size;

                    for b in 0..batch_size {
                        for col in 0..self.image_size {
                            let src_idx = row_start + col;
                            if src_idx < input.shape()[1] {
                                encoded[[b, col]] = input[[b, src_idx]];
                            }
                        }
                    }
                }
                // When no row is active, return zeros

                encoded
            }
        }
    }

    /// Check if encoding is temporal
    pub fn is_temporal(&self) -> bool {
        matches!(self.encoding_type, InputEncodingType::Temporal { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require MNIST data to be downloaded
    // They are marked as ignored by default

    #[test]
    #[ignore]
    fn test_mnist_load() {
        let dataset = MnistDataset::load("./data").expect("Failed to load MNIST");

        assert_eq!(dataset.train_len(), 60_000);
        assert_eq!(dataset.test_len(), 10_000);
        assert_eq!(dataset.feature_dim(), 36); // 6x6
    }

    #[test]
    #[ignore]
    fn test_batch_iterator() {
        let dataset = MnistDataset::load("./data").expect("Failed to load MNIST");

        let mut iter = BatchIterator::new(&dataset, 128, true, true);
        let (batch, labels) = iter.next().expect("Should have at least one batch");

        assert_eq!(batch.shape()[0], 128);
        assert_eq!(batch.shape()[1], 36);
        assert_eq!(labels.len(), 128);
    }

    #[test]
    fn test_rate_coded_encoder() {
        use ndarray::array;

        let encoder = InputEncoder::rate_coded(3); // 3x3 image

        // Create a test input [batch=1, features=9]
        let input = array![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]];

        // Rate-coded should return same input at every timestep
        let encoded_t0 = encoder.encode_timestep(&input, 0);
        let encoded_t5 = encoder.encode_timestep(&input, 5);
        let encoded_t100 = encoder.encode_timestep(&input, 100);

        assert_eq!(encoded_t0, input);
        assert_eq!(encoded_t5, input);
        assert_eq!(encoded_t100, input);
    }

    #[test]
    fn test_temporal_encoder() {
        use ndarray::array;

        // 3x3 image, row_spacing=2ms, pulse_width=90%, dt=1ms
        let encoder = InputEncoder::temporal(3, 0.002, 0.9, 0.001);

        // Verify output dimension is 3 (one row at a time)
        assert_eq!(
            encoder.output_dim(),
            3,
            "Temporal encoder should output 3 values (one row)"
        );

        // Create a test input [batch=1, features=9]
        // Row 0: [1, 2, 3], Row 1: [4, 5, 6], Row 2: [7, 8, 9]
        let input = array![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]];

        // Timestep 0: row 0 active (time=0ms, row=0)
        // Output should be [batch=1, 3] with row 0's values
        let t0 = encoder.encode_timestep(&input, 0);
        assert_eq!(t0.shape(), &[1, 3], "Output should be [1, 3]");
        assert_eq!(t0[[0, 0]], 1.0, "First col of row 0");
        assert_eq!(t0[[0, 1]], 2.0, "Second col of row 0");
        assert_eq!(t0[[0, 2]], 3.0, "Third col of row 0");

        // Timestep 1: still row 0 active (time=1ms, row=0, within pulse)
        let t1 = encoder.encode_timestep(&input, 1);
        assert_eq!(t1[[0, 0]], 1.0, "Row 0 should still be active");

        // Timestep 2: row 1 active (time=2ms, row=1)
        // Output should have row 1's values: [4, 5, 6]
        let t2 = encoder.encode_timestep(&input, 2);
        assert_eq!(t2[[0, 0]], 4.0, "First col of row 1");
        assert_eq!(t2[[0, 1]], 5.0, "Second col of row 1");
        assert_eq!(t2[[0, 2]], 6.0, "Third col of row 1");

        // Timestep 4: row 2 active (time=4ms, row=2)
        // Output should have row 2's values: [7, 8, 9]
        let t4 = encoder.encode_timestep(&input, 4);
        assert_eq!(t4[[0, 0]], 7.0, "First col of row 2");
        assert_eq!(t4[[0, 1]], 8.0, "Second col of row 2");
        assert_eq!(t4[[0, 2]], 9.0, "Third col of row 2");

        // Timestep 6: all rows done (time=6ms, row=3 > image_size)
        let t6 = encoder.encode_timestep(&input, 6);
        assert!(
            t6.iter().all(|&v| v == 0.0),
            "All should be zero after all rows"
        );
    }

    #[test]
    fn test_temporal_timesteps_needed() {
        // 7x7 image, row_spacing=1.5ms, dt=1ms
        let encoder = InputEncoder::temporal(7, 0.0015, 0.9, 0.001);

        // Total time = 7 rows * 1.5ms = 10.5ms = 11 timesteps at dt=1ms
        let steps = encoder.timesteps_needed(5);
        assert!(
            steps >= 11,
            "Should need at least 11 steps for temporal encoding"
        );
    }
}
