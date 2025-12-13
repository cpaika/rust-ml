use crate::{DataLoader, Sample};
use common::Digit;

/// Wrapper around Digit that implements the Data trait
#[derive(Clone)]
pub struct MnistSample {
    digit: Digit,
}

impl MnistSample {
    pub fn new(digit: Digit) -> Self {
        Self { digit }
    }

    pub fn label(&self) -> u8 {
        self.digit.label()
    }

    pub fn pixels(&self) -> &[u8] {
        self.digit.pixels()
    }

    /// Returns normalized pixel values (0.0 - 1.0)
    pub fn normalized_pixels(&self) -> Vec<f64> {
        self.digit.pixels().iter().map(|&p| p as f64 / 255.0).collect()
    }
}

impl Sample for MnistSample {}

/// Loads MNIST data and provides iteration over batches
pub struct MnistLoader {
    samples: Vec<MnistSample>,
    batch_size: usize,
    current_index: usize,
}

impl MnistLoader {
    /// Create a new loader from raw CSV bytes
    pub fn from_bytes(data: &[u8], batch_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let digits = common::parse_digits_from_bytes(data)?;
        let samples = digits.into_iter().map(MnistSample::new).collect();
        Ok(Self {
            samples,
            batch_size,
            current_index: 0,
        })
    }

    /// Create a new loader from a file path
    pub fn from_file<P: AsRef<std::path::Path>>(path: P, batch_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let digits = common::parse_digits(path)?;
        let samples = digits.into_iter().map(MnistSample::new).collect();
        Ok(Self {
            samples,
            batch_size,
            current_index: 0,
        })
    }

    /// Total number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Shuffle the samples (useful between epochs)
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.samples.shuffle(&mut rng);
    }
}

impl Iterator for MnistLoader {
    type Item = Vec<MnistSample>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.samples.len() {
            return None;
        }

        let end = std::cmp::min(self.current_index + self.batch_size, self.samples.len());
        let batch: Vec<MnistSample> = self.samples[self.current_index..end]
            .iter()
            .cloned()
            .collect();
        self.current_index = end;

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

impl DataLoader for MnistLoader {
    type SampleType = MnistSample;

    fn reset(&mut self) {
        self.current_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalized_pixels() {
        let loader = MnistLoader::from_file(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../perceptron/digit-recognizer/train.csv"),
            1
        ).unwrap();

        let sample = &loader.samples[0];
        let normalized = sample.normalized_pixels();

        // Should have 784 pixels (28x28)
        assert_eq!(normalized.len(), 784);

        // All values should be between 0.0 and 1.0
        for &pixel in &normalized {
            assert!(pixel >= 0.0 && pixel <= 1.0);
        }

        // Verify normalization: raw pixel / 255.0 = normalized
        let raw_pixels = sample.pixels();
        for (i, &raw) in raw_pixels.iter().enumerate() {
            let expected = raw as f64 / 255.0;
            assert!((normalized[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_input_array_for_training() {
        let loader = MnistLoader::from_file(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../perceptron/digit-recognizer/train.csv"),
            1
        ).unwrap();

        let sample = &loader.samples[0];
        let input = sample.normalized_pixels();

        // Input layer expects 784 normalized values
        assert_eq!(input.len(), 784);

        // Label should be 0-9
        assert!(sample.label() <= 9);
    }
}