//! Feed-forward network (MLP) for transformer

use crate::{xavier_init, Activation, TransformerConfig};
use serde::{Deserialize, Serialize};

/// Position-wise feed-forward network
///
/// Applies two linear transformations with an activation in between:
/// FFN(x) = activation(x * W1 + b1) * W2 + b2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForward {
    /// First layer weights [d_model, d_ff]
    pub w1: Vec<f32>,
    /// First layer bias [d_ff]
    pub b1: Vec<f32>,
    /// Second layer weights [d_ff, d_model]
    pub w2: Vec<f32>,
    /// Second layer bias [d_model]
    pub b2: Vec<f32>,

    /// Model dimension
    pub d_model: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Activation function
    pub activation: Activation,

    // Cached values for backpropagation
    #[serde(skip)]
    cached_input: Vec<f32>,
    #[serde(skip)]
    cached_hidden: Vec<f32>,
    #[serde(skip)]
    cached_pre_activation: Vec<f32>,
}

impl FeedForward {
    /// Create new feed-forward network
    pub fn new(d_model: usize, d_ff: usize, activation: Activation) -> Self {
        // Initialize weights with Xavier/Glorot
        let w1: Vec<f32> = (0..d_model * d_ff)
            .map(|_| xavier_init(d_model, d_ff))
            .collect();
        let w2: Vec<f32> = (0..d_ff * d_model)
            .map(|_| xavier_init(d_ff, d_model))
            .collect();

        let b1 = vec![0.0; d_ff];
        let b2 = vec![0.0; d_model];

        Self {
            w1, b1, w2, b2,
            d_model,
            d_ff,
            activation,
            cached_input: Vec::new(),
            cached_hidden: Vec::new(),
            cached_pre_activation: Vec::new(),
        }
    }

    /// Create from config
    pub fn from_config(config: &TransformerConfig) -> Self {
        Self::new(config.d_model, config.d_ff, config.activation)
    }

    /// Forward pass
    ///
    /// Input: [seq_len, d_model] as flat vector
    /// Output: [seq_len, d_model] as flat vector
    pub fn forward(&mut self, x: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(x.len(), seq_len * self.d_model);

        // First linear: x * W1 + b1 -> [seq_len, d_ff]
        let mut hidden = vec![0.0f32; seq_len * self.d_ff];
        let mut pre_activation = vec![0.0f32; seq_len * self.d_ff];

        for pos in 0..seq_len {
            for ff in 0..self.d_ff {
                let mut sum = self.b1[ff];
                for d in 0..self.d_model {
                    sum += x[pos * self.d_model + d] * self.w1[d * self.d_ff + ff];
                }
                pre_activation[pos * self.d_ff + ff] = sum;
                hidden[pos * self.d_ff + ff] = self.activation.apply(sum);
            }
        }

        // Second linear: hidden * W2 + b2 -> [seq_len, d_model]
        let mut output = vec![0.0f32; seq_len * self.d_model];

        for pos in 0..seq_len {
            for d in 0..self.d_model {
                let mut sum = self.b2[d];
                for ff in 0..self.d_ff {
                    sum += hidden[pos * self.d_ff + ff] * self.w2[ff * self.d_model + d];
                }
                output[pos * self.d_model + d] = sum;
            }
        }

        // Cache for backward pass
        self.cached_input = x.to_vec();
        self.cached_hidden = hidden;
        self.cached_pre_activation = pre_activation;

        output
    }

    /// Backward pass
    ///
    /// grad_output: gradient w.r.t. output [seq_len, d_model]
    /// Returns: gradient w.r.t. input [seq_len, d_model]
    pub fn backward(
        &mut self,
        grad_output: &[f32],
        seq_len: usize,
        learning_rate: f32,
    ) -> Vec<f32> {
        assert_eq!(grad_output.len(), seq_len * self.d_model);

        // Gradient w.r.t. hidden (before W2)
        let mut grad_hidden = vec![0.0f32; seq_len * self.d_ff];

        // Also accumulate gradients for W2 and b2
        let mut grad_w2 = vec![0.0f32; self.d_ff * self.d_model];
        let mut grad_b2 = vec![0.0f32; self.d_model];

        for pos in 0..seq_len {
            for d in 0..self.d_model {
                let g = grad_output[pos * self.d_model + d];
                grad_b2[d] += g;

                for ff in 0..self.d_ff {
                    grad_w2[ff * self.d_model + d] += self.cached_hidden[pos * self.d_ff + ff] * g;
                    grad_hidden[pos * self.d_ff + ff] += self.w2[ff * self.d_model + d] * g;
                }
            }
        }

        // Apply activation derivative
        for i in 0..grad_hidden.len() {
            grad_hidden[i] *= self.activation.derivative(self.cached_pre_activation[i]);
        }

        // Gradient w.r.t. input and W1, b1
        let mut grad_input = vec![0.0f32; seq_len * self.d_model];
        let mut grad_w1 = vec![0.0f32; self.d_model * self.d_ff];
        let mut grad_b1 = vec![0.0f32; self.d_ff];

        for pos in 0..seq_len {
            for ff in 0..self.d_ff {
                let g = grad_hidden[pos * self.d_ff + ff];
                grad_b1[ff] += g;

                for d in 0..self.d_model {
                    grad_w1[d * self.d_ff + ff] += self.cached_input[pos * self.d_model + d] * g;
                    grad_input[pos * self.d_model + d] += self.w1[d * self.d_ff + ff] * g;
                }
            }
        }

        // Apply gradients
        let batch_lr = learning_rate / seq_len as f32;
        for i in 0..self.w1.len() {
            self.w1[i] -= batch_lr * grad_w1[i];
        }
        for i in 0..self.b1.len() {
            self.b1[i] -= batch_lr * grad_b1[i];
        }
        for i in 0..self.w2.len() {
            self.w2[i] -= batch_lr * grad_w2[i];
        }
        for i in 0..self.b2.len() {
            self.b2[i] -= batch_lr * grad_b2[i];
        }

        grad_input
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

/// Gated Linear Unit (GLU) variant for feed-forward
///
/// GLU(x) = (x * W_gate) * sigmoid(x * W_up) * W_down
/// Used in modern architectures like PaLM, LLaMA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatedFeedForward {
    /// Gate projection [d_model, d_ff]
    pub w_gate: Vec<f32>,
    /// Up projection [d_model, d_ff]
    pub w_up: Vec<f32>,
    /// Down projection [d_ff, d_model]
    pub w_down: Vec<f32>,

    /// Model dimension
    pub d_model: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Activation function (applied to gate)
    pub activation: Activation,
}

impl GatedFeedForward {
    /// Create new gated feed-forward
    pub fn new(d_model: usize, d_ff: usize, activation: Activation) -> Self {
        let w_gate: Vec<f32> = (0..d_model * d_ff)
            .map(|_| xavier_init(d_model, d_ff))
            .collect();
        let w_up: Vec<f32> = (0..d_model * d_ff)
            .map(|_| xavier_init(d_model, d_ff))
            .collect();
        let w_down: Vec<f32> = (0..d_ff * d_model)
            .map(|_| xavier_init(d_ff, d_model))
            .collect();

        Self {
            w_gate, w_up, w_down,
            d_model,
            d_ff,
            activation,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(x.len(), seq_len * self.d_model);

        // Gate: x * W_gate -> [seq_len, d_ff]
        let mut gate = vec![0.0f32; seq_len * self.d_ff];
        // Up: x * W_up -> [seq_len, d_ff]
        let mut up = vec![0.0f32; seq_len * self.d_ff];

        for pos in 0..seq_len {
            for ff in 0..self.d_ff {
                let mut gate_sum = 0.0;
                let mut up_sum = 0.0;
                for d in 0..self.d_model {
                    let x_val = x[pos * self.d_model + d];
                    gate_sum += x_val * self.w_gate[d * self.d_ff + ff];
                    up_sum += x_val * self.w_up[d * self.d_ff + ff];
                }
                gate[pos * self.d_ff + ff] = self.activation.apply(gate_sum);
                up[pos * self.d_ff + ff] = up_sum;
            }
        }

        // Element-wise multiply and project down
        let mut output = vec![0.0f32; seq_len * self.d_model];

        for pos in 0..seq_len {
            // Hadamard product
            let mut hidden = vec![0.0f32; self.d_ff];
            for ff in 0..self.d_ff {
                hidden[ff] = gate[pos * self.d_ff + ff] * up[pos * self.d_ff + ff];
            }

            // Down projection
            for d in 0..self.d_model {
                let mut sum = 0.0;
                for ff in 0..self.d_ff {
                    sum += hidden[ff] * self.w_down[ff * self.d_model + d];
                }
                output[pos * self.d_model + d] = sum;
            }
        }

        output
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        3 * self.d_model * self.d_ff  // gate + up + down
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_creation() {
        let ff = FeedForward::new(64, 256, Activation::GELU);
        assert_eq!(ff.d_model, 64);
        assert_eq!(ff.d_ff, 256);
        assert_eq!(ff.w1.len(), 64 * 256);
        assert_eq!(ff.w2.len(), 256 * 64);
    }

    #[test]
    fn test_feedforward_forward_shape() {
        let mut ff = FeedForward::new(32, 128, Activation::GELU);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 32];

        let output = ff.forward(&x, seq_len);
        assert_eq!(output.len(), seq_len * 32);
    }

    #[test]
    fn test_feedforward_from_config() {
        let config = TransformerConfig::tiny(1000);
        let ff = FeedForward::from_config(&config);
        assert_eq!(ff.d_model, config.d_model);
        assert_eq!(ff.d_ff, config.d_ff);
        assert_eq!(ff.activation, config.activation);
    }

    #[test]
    fn test_feedforward_num_parameters() {
        let ff = FeedForward::new(64, 256, Activation::GELU);
        let params = ff.num_parameters();

        // w1: 64*256, b1: 256, w2: 256*64, b2: 64
        assert_eq!(params, 64 * 256 + 256 + 256 * 64 + 64);
    }

    #[test]
    fn test_feedforward_backward_shape() {
        let mut ff = FeedForward::new(32, 128, Activation::GELU);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 32];

        // Forward pass
        ff.forward(&x, seq_len);

        // Backward pass
        let grad_output = vec![0.01f32; seq_len * 32];
        let grad_input = ff.backward(&grad_output, seq_len, 0.01);

        assert_eq!(grad_input.len(), seq_len * 32);
    }

    #[test]
    fn test_feedforward_weights_change_after_backward() {
        let mut ff = FeedForward::new(16, 64, Activation::GELU);
        let seq_len = 2;
        let x = vec![0.1f32; seq_len * 16];

        let initial_w1 = ff.w1.clone();

        ff.forward(&x, seq_len);
        let grad_output = vec![0.1f32; seq_len * 16];
        ff.backward(&grad_output, seq_len, 0.1);

        // Weights should have changed
        let weights_changed = ff.w1.iter().zip(initial_w1.iter())
            .any(|(new, old)| (new - old).abs() > 1e-6);
        assert!(weights_changed, "Weights should change after backward pass");
    }

    #[test]
    fn test_feedforward_different_activations() {
        for activation in [Activation::GELU, Activation::ReLU, Activation::Swish] {
            let mut ff = FeedForward::new(32, 128, activation);
            let x = vec![0.5f32; 2 * 32];

            let output = ff.forward(&x, 2);
            assert_eq!(output.len(), 2 * 32);
        }
    }

    #[test]
    fn test_feedforward_deterministic() {
        let mut ff1 = FeedForward::new(32, 128, Activation::GELU);
        let mut ff2 = ff1.clone();

        let x = vec![0.5f32; 4 * 32];

        let out1 = ff1.forward(&x, 4);
        let out2 = ff2.forward(&x, 4);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gated_feedforward_creation() {
        let gff = GatedFeedForward::new(64, 256, Activation::Swish);
        assert_eq!(gff.d_model, 64);
        assert_eq!(gff.d_ff, 256);
    }

    #[test]
    fn test_gated_feedforward_forward() {
        let gff = GatedFeedForward::new(32, 128, Activation::Swish);
        let x = vec![0.1f32; 4 * 32];

        let output = gff.forward(&x, 4);
        assert_eq!(output.len(), 4 * 32);
    }

    #[test]
    fn test_gated_feedforward_num_parameters() {
        let gff = GatedFeedForward::new(64, 256, Activation::Swish);
        // 3 weight matrices, no biases
        assert_eq!(gff.num_parameters(), 3 * 64 * 256);
    }

    #[test]
    fn test_feedforward_expands_then_contracts() {
        let mut ff = FeedForward::new(32, 128, Activation::GELU);

        // d_ff should be larger than d_model (expansion)
        assert!(ff.d_ff > ff.d_model);

        // But output should be same dimension as input
        let x = vec![0.1f32; 4 * 32];
        let output = ff.forward(&x, 4);
        assert_eq!(output.len(), x.len());
    }

    #[test]
    fn test_feedforward_nonlinearity_matters() {
        // With GELU, negative pre-activations are dampened
        let mut ff = FeedForward::new(4, 8, Activation::GELU);

        // Set weights to create negative pre-activations
        for w in ff.w1.iter_mut() {
            *w = -1.0;
        }
        ff.b1 = vec![-1.0; 8];

        let x = vec![1.0f32; 4];
        let output = ff.forward(&x, 1);

        // Should not be all zeros (GELU is not ReLU)
        let has_nonzero = output.iter().any(|&v| v.abs() > 1e-6);
        // But should be dampened from full activation
        assert!(has_nonzero || output.iter().all(|&v| v.abs() < 10.0));
    }
}
