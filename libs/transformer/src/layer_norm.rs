//! Layer normalization

use crate::TransformerConfig;
use serde::{Deserialize, Serialize};

/// Layer normalization
///
/// Normalizes across the feature dimension for each position:
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    /// Scale parameter (gamma) [d_model]
    pub gamma: Vec<f32>,
    /// Shift parameter (beta) [d_model]
    pub beta: Vec<f32>,
    /// Feature dimension
    pub d_model: usize,
    /// Epsilon for numerical stability
    pub eps: f32,

    // Cached values for backpropagation
    #[serde(skip)]
    cached_input: Vec<f32>,
    #[serde(skip)]
    cached_normalized: Vec<f32>,
    #[serde(skip)]
    cached_mean: Vec<f32>,
    #[serde(skip)]
    cached_var: Vec<f32>,
}

impl LayerNorm {
    /// Create new layer normalization
    pub fn new(d_model: usize, eps: f32) -> Self {
        Self {
            gamma: vec![1.0; d_model],
            beta: vec![0.0; d_model],
            d_model,
            eps,
            cached_input: Vec::new(),
            cached_normalized: Vec::new(),
            cached_mean: Vec::new(),
            cached_var: Vec::new(),
        }
    }

    /// Create from config
    pub fn from_config(config: &TransformerConfig) -> Self {
        Self::new(config.d_model, config.layer_norm_eps)
    }

    /// Forward pass
    ///
    /// Input: [seq_len, d_model] as flat vector
    /// Output: [seq_len, d_model] as flat vector
    pub fn forward(&mut self, x: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(x.len(), seq_len * self.d_model);

        let mut output = vec![0.0f32; x.len()];
        let mut means = vec![0.0f32; seq_len];
        let mut vars = vec![0.0f32; seq_len];
        let mut normalized = vec![0.0f32; x.len()];

        // Compute mean and variance for each position
        for pos in 0..seq_len {
            let start = pos * self.d_model;
            let end = start + self.d_model;
            let slice = &x[start..end];

            // Mean
            let mean: f32 = slice.iter().sum::<f32>() / self.d_model as f32;
            means[pos] = mean;

            // Variance
            let var: f32 = slice.iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f32>() / self.d_model as f32;
            vars[pos] = var;

            // Normalize and scale
            let std_inv = 1.0 / (var + self.eps).sqrt();
            for dim in 0..self.d_model {
                let x_normalized = (slice[dim] - mean) * std_inv;
                normalized[start + dim] = x_normalized;
                output[start + dim] = self.gamma[dim] * x_normalized + self.beta[dim];
            }
        }

        // Cache for backward pass
        self.cached_input = x.to_vec();
        self.cached_normalized = normalized;
        self.cached_mean = means;
        self.cached_var = vars;

        output
    }

    /// Forward pass for a single position (useful for incremental inference)
    pub fn forward_single(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.d_model);

        let mean: f32 = x.iter().sum::<f32>() / self.d_model as f32;
        let var: f32 = x.iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>() / self.d_model as f32;

        let std_inv = 1.0 / (var + self.eps).sqrt();

        x.iter()
            .enumerate()
            .map(|(dim, &v)| {
                let x_norm = (v - mean) * std_inv;
                self.gamma[dim] * x_norm + self.beta[dim]
            })
            .collect()
    }

    /// Backward pass with parameter updates
    ///
    /// Computes gradient w.r.t. input and updates gamma/beta parameters.
    ///
    /// # Arguments
    /// * `grad_output` - gradient w.r.t. output [seq_len, d_model]
    /// * `seq_len` - sequence length
    /// * `learning_rate` - learning rate for gamma/beta updates
    ///
    /// # Returns
    /// Gradient w.r.t. input [seq_len, d_model]
    ///
    /// # Mathematical Details
    ///
    /// Forward: y = gamma * (x - mean) / std + beta
    /// where: normalized = (x - mean) / std
    ///
    /// Gradients:
    /// - grad_gamma = sum over positions of (grad_output * normalized)
    /// - grad_beta = sum over positions of (grad_output)
    /// - grad_input uses chain rule through normalization
    pub fn backward(&mut self, grad_output: &[f32], seq_len: usize, learning_rate: f32) -> Vec<f32> {
        assert_eq!(grad_output.len(), seq_len * self.d_model);

        let mut grad_input = vec![0.0f32; grad_output.len()];
        let mut grad_gamma = vec![0.0f32; self.d_model];
        let mut grad_beta = vec![0.0f32; self.d_model];

        for pos in 0..seq_len {
            let start = pos * self.d_model;

            let mean = self.cached_mean[pos];
            let var = self.cached_var[pos];
            let std_inv = 1.0 / (var + self.eps).sqrt();

            // Accumulate gradients for gamma and beta
            // grad_gamma[d] += grad_output[pos, d] * normalized[pos, d]
            // grad_beta[d] += grad_output[pos, d]
            for dim in 0..self.d_model {
                grad_gamma[dim] += grad_output[start + dim] * self.cached_normalized[start + dim];
                grad_beta[dim] += grad_output[start + dim];
            }

            // Gradient w.r.t. normalized input
            let grad_normalized: Vec<f32> = (0..self.d_model)
                .map(|dim| grad_output[start + dim] * self.gamma[dim])
                .collect();

            // Gradient w.r.t. variance
            let x_minus_mean: Vec<f32> = (0..self.d_model)
                .map(|dim| self.cached_input[start + dim] - mean)
                .collect();

            let grad_var: f32 = grad_normalized.iter()
                .zip(x_minus_mean.iter())
                .map(|(&gn, &xm)| gn * xm * -0.5 * std_inv.powi(3))
                .sum();

            // Gradient w.r.t. mean
            let grad_mean: f32 = grad_normalized.iter().map(|&gn| gn * -std_inv).sum::<f32>()
                + grad_var * x_minus_mean.iter().map(|&xm| -2.0 * xm).sum::<f32>() / self.d_model as f32;

            // Gradient w.r.t. input
            for dim in 0..self.d_model {
                grad_input[start + dim] = grad_normalized[dim] * std_inv
                    + grad_var * 2.0 * x_minus_mean[dim] / self.d_model as f32
                    + grad_mean / self.d_model as f32;
            }
        }

        // Apply gradient updates to gamma and beta
        let batch_lr = learning_rate / seq_len as f32;
        for d in 0..self.d_model {
            self.gamma[d] -= batch_lr * grad_gamma[d];
            self.beta[d] -= batch_lr * grad_beta[d];
        }

        grad_input
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        2 * self.d_model // gamma + beta
    }

    /// Get gamma parameters
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// Get beta parameters
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }

    /// Get mutable gamma
    pub fn gamma_mut(&mut self) -> &mut [f32] {
        &mut self.gamma
    }

    /// Get mutable beta
    pub fn beta_mut(&mut self) -> &mut [f32] {
        &mut self.beta
    }
}

/// RMS Layer Normalization (used in LLaMA and other modern architectures)
///
/// y = x / sqrt(mean(x^2) + eps) * gamma
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNorm {
    /// Scale parameter (gamma) [d_model]
    pub gamma: Vec<f32>,
    /// Feature dimension
    pub d_model: usize,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl RMSNorm {
    /// Create new RMS normalization
    pub fn new(d_model: usize, eps: f32) -> Self {
        Self {
            gamma: vec![1.0; d_model],
            d_model,
            eps,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(x.len(), seq_len * self.d_model);

        let mut output = vec![0.0f32; x.len()];

        for pos in 0..seq_len {
            let start = pos * self.d_model;
            let end = start + self.d_model;
            let slice = &x[start..end];

            // RMS
            let rms: f32 = (slice.iter().map(|&v| v * v).sum::<f32>() / self.d_model as f32 + self.eps).sqrt();

            // Normalize and scale
            for dim in 0..self.d_model {
                output[start + dim] = slice[dim] / rms * self.gamma[dim];
            }
        }

        output
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.d_model // Only gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let ln = LayerNorm::new(64, 1e-5);
        assert_eq!(ln.d_model, 64);
        assert_eq!(ln.gamma.len(), 64);
        assert_eq!(ln.beta.len(), 64);
        // Gamma initialized to 1
        assert!(ln.gamma.iter().all(|&g| (g - 1.0).abs() < 1e-5));
        // Beta initialized to 0
        assert!(ln.beta.iter().all(|&b| b.abs() < 1e-5));
    }

    #[test]
    fn test_layer_norm_forward_shape() {
        let mut ln = LayerNorm::new(32, 1e-5);
        let seq_len = 4;
        let x = vec![1.0f32; seq_len * 32];

        let output = ln.forward(&x, seq_len);
        assert_eq!(output.len(), seq_len * 32);
    }

    #[test]
    fn test_layer_norm_normalized_output() {
        let mut ln = LayerNorm::new(32, 1e-5);
        let seq_len = 1;

        // Create input with non-zero mean and non-unit variance
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();

        let output = ln.forward(&x, seq_len);

        // With gamma=1, beta=0, output should have mean≈0, var≈1
        let mean: f32 = output.iter().sum::<f32>() / 32.0;
        let var: f32 = output.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 32.0;

        assert!(mean.abs() < 1e-4, "Mean should be ~0, got {}", mean);
        assert!((var - 1.0).abs() < 1e-4, "Variance should be ~1, got {}", var);
    }

    #[test]
    fn test_layer_norm_forward_single() {
        let ln = LayerNorm::new(32, 1e-5);
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();

        let output = ln.forward_single(&x);
        assert_eq!(output.len(), 32);

        // Check normalized
        let mean: f32 = output.iter().sum::<f32>() / 32.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_with_gamma_beta() {
        let mut ln = LayerNorm::new(4, 1e-5);
        ln.gamma = vec![2.0, 2.0, 2.0, 2.0];
        ln.beta = vec![1.0, 1.0, 1.0, 1.0];

        let x = vec![0.0, 1.0, 2.0, 3.0];
        let output = ln.forward(&x, 1);

        // Mean of output should be beta = 1 (approximately)
        // Variance should be gamma^2 = 4 (approximately)
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!((mean - 1.0).abs() < 0.1, "Mean with beta=1 should be ~1, got {}", mean);
    }

    #[test]
    fn test_layer_norm_multiple_positions() {
        let mut ln = LayerNorm::new(4, 1e-5);
        let seq_len = 3;

        // Different values at each position
        let x = vec![
            0.0, 1.0, 2.0, 3.0,  // pos 0
            4.0, 5.0, 6.0, 7.0,  // pos 1
            8.0, 9.0, 10.0, 11.0, // pos 2
        ];

        let output = ln.forward(&x, seq_len);

        // Each position should be independently normalized
        for pos in 0..seq_len {
            let slice = &output[pos * 4..(pos + 1) * 4];
            let mean: f32 = slice.iter().sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "Position {} mean should be ~0, got {}", pos, mean);
        }
    }

    #[test]
    fn test_layer_norm_num_parameters() {
        let ln = LayerNorm::new(64, 1e-5);
        assert_eq!(ln.num_parameters(), 128); // gamma + beta
    }

    #[test]
    fn test_layer_norm_from_config() {
        let config = TransformerConfig::tiny(1000);
        let ln = LayerNorm::from_config(&config);
        assert_eq!(ln.d_model, config.d_model);
        assert_eq!(ln.eps, config.layer_norm_eps);
    }

    #[test]
    fn test_rms_norm_creation() {
        let rms = RMSNorm::new(64, 1e-5);
        assert_eq!(rms.d_model, 64);
        assert_eq!(rms.gamma.len(), 64);
    }

    #[test]
    fn test_rms_norm_forward() {
        let rms = RMSNorm::new(32, 1e-5);
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();

        let output = rms.forward(&x, 1);
        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_rms_norm_num_parameters() {
        let rms = RMSNorm::new(64, 1e-5);
        assert_eq!(rms.num_parameters(), 64); // Only gamma
    }

    #[test]
    fn test_layer_norm_backward_shape() {
        let mut ln = LayerNorm::new(32, 1e-5);
        let seq_len = 4;
        let x = vec![1.0f32; seq_len * 32];

        // Forward pass to populate cache
        ln.forward(&x, seq_len);

        // Backward pass
        let grad_output = vec![0.1f32; seq_len * 32];
        let grad_input = ln.backward(&grad_output, seq_len, 0.0);

        assert_eq!(grad_input.len(), seq_len * 32);
    }

    #[test]
    fn test_layer_norm_backward_gamma_beta_update() {
        let mut ln = LayerNorm::new(8, 1e-5);
        let seq_len = 2;
        let x: Vec<f32> = (0..seq_len * 8).map(|i| i as f32 * 0.1).collect();

        let gamma_before = ln.gamma.clone();
        let beta_before = ln.beta.clone();

        ln.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * 8).map(|i| (i + 1) as f32 * 0.01).collect();
        ln.backward(&grad_output, seq_len, 0.1);

        // Gamma and beta should have changed
        let gamma_changed = ln.gamma.iter().zip(gamma_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        let beta_changed = ln.beta.iter().zip(beta_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(gamma_changed, "gamma should be updated");
        assert!(beta_changed, "beta should be updated");
    }

    #[test]
    fn test_layer_norm_backward_no_update_with_zero_lr() {
        let mut ln = LayerNorm::new(8, 1e-5);
        let seq_len = 2;
        let x: Vec<f32> = (0..seq_len * 8).map(|i| i as f32 * 0.1).collect();

        let gamma_before = ln.gamma.clone();

        ln.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * 8).map(|i| (i + 1) as f32 * 0.01).collect();
        ln.backward(&grad_output, seq_len, 0.0); // Zero learning rate

        // Gamma should NOT have changed
        let gamma_same = ln.gamma.iter().zip(gamma_before.iter()).all(|(a, b)| (a - b).abs() < 1e-10);
        assert!(gamma_same, "gamma should NOT be updated with lr=0");
    }

    #[test]
    fn test_layer_norm_backward_grad_input_nonzero() {
        let mut ln = LayerNorm::new(8, 1e-5);
        let seq_len = 2;
        let x: Vec<f32> = (0..seq_len * 8).map(|i| i as f32 * 0.1).collect();

        ln.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * 8).map(|i| (i + 1) as f32 * 0.01).collect();
        let grad_input = ln.backward(&grad_output, seq_len, 0.0);

        // Gradient input should have non-zero values
        let has_nonzero = grad_input.iter().any(|&g| g.abs() > 1e-10);
        assert!(has_nonzero, "Gradient input should have non-zero values");
    }

    #[test]
    fn test_layer_norm_deterministic() {
        let mut ln1 = LayerNorm::new(32, 1e-5);
        let mut ln2 = ln1.clone();

        let x = vec![0.5f32; 4 * 32];

        let out1 = ln1.forward(&x, 4);
        let out2 = ln2.forward(&x, 4);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_layer_norm_gamma_beta_access() {
        let mut ln = LayerNorm::new(4, 1e-5);

        ln.gamma_mut()[0] = 2.0;
        ln.beta_mut()[0] = 0.5;

        assert_eq!(ln.gamma()[0], 2.0);
        assert_eq!(ln.beta()[0], 0.5);
    }

    #[test]
    fn test_rms_norm_output_scale() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = vec![1.0, 1.0, 1.0, 1.0];

        let output = rms.forward(&x, 1);

        // With all 1s, RMS = 1, so output should equal input (with gamma=1)
        for (a, b) in output.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
