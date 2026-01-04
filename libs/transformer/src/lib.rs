//! Transformer Architecture for Language Modeling
//!
//! A GPT-style decoder-only transformer implemented from scratch
//! with GPU acceleration via WebGPU compute shaders.

use rand::Rng;
#[allow(unused_imports)]
use rand::distr::Distribution;
use serde::{Deserialize, Serialize};

pub mod config;
pub mod embedding;
pub mod attention;
pub mod layer_norm;
pub mod feed_forward;
pub mod block;
pub mod model;
pub mod checkpoint;
pub mod training;
pub mod gpu;
pub mod grad_check;
pub mod optimizer;
pub mod eval;

pub use config::TransformerConfig;
pub use model::Transformer;
pub use checkpoint::{Checkpoint, CheckpointManager};

/// Activation function types for feed-forward layers
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// Gaussian Error Linear Unit (smoother than ReLU)
    GELU,
    /// Rectified Linear Unit
    ReLU,
    /// Swish activation (x * sigmoid(x))
    Swish,
}

impl Activation {
    /// Apply the activation function element-wise
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::GELU => {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let sqrt_2_pi = 0.7978845608028654;
                let cdf = 0.5 * (1.0 + (sqrt_2_pi * (x + 0.044715 * x * x * x)).tanh());
                x * cdf
            }
            Activation::ReLU => x.max(0.0),
            Activation::Swish => x * sigmoid(x),
        }
    }

    /// Derivative of the activation function
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Activation::GELU => {
                // Approximate GELU derivative
                let sqrt_2_pi = 0.7978845608028654;
                let x3 = x * x * x;
                let inner = sqrt_2_pi * (x + 0.044715 * x3);
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;
                let cdf = 0.5 * (1.0 + tanh_inner);
                let pdf = 0.5 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x) * sech2;
                cdf + x * pdf
            }
            Activation::ReLU => {
                if x > 0.0 { 1.0 } else { 0.0 }
            }
            Activation::Swish => {
                let sig = sigmoid(x);
                sig + x * sig * (1.0 - sig)
            }
        }
    }
}

/// Sigmoid function
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Xavier/Glorot initialization for weights
pub fn xavier_init(fan_in: usize, fan_out: usize) -> f32 {
    let mut rng = rand::rng();
    let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
    rng.random_range(-scale..scale)
}

/// Kaiming/He initialization for ReLU networks
pub fn kaiming_init(fan_in: usize) -> f32 {
    let mut rng = rand::rng();
    let scale = (2.0 / fan_in as f32).sqrt();
    rng.random_range(-scale..scale)
}

/// Standard normal initialization
pub fn normal_init(std: f32) -> f32 {
    let mut rng = rand::rng();
    // Box-Muller transform for normal distribution
    let u1: f32 = rng.random();
    let u2: f32 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    z * std
}

/// Softmax over a slice (for attention weights)
///
/// Optimized implementation that:
/// 1. Finds max in first pass
/// 2. Computes exp(x - max) and sum in second pass
/// 3. Normalizes in third pass
///
/// This is numerically stable and reasonably efficient for small arrays.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Pass 1: Find max for numerical stability
    let mut max_val = x[0];
    for &v in x.iter().skip(1) {
        if v > max_val {
            max_val = v;
        }
    }

    // Pass 2: Compute exp(x - max) and accumulate sum
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Pass 3: Normalize
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Fused softmax that returns log probabilities
///
/// More numerically stable for cross-entropy loss computation:
/// log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max)))
///
/// This avoids computing exp() then log(), which loses precision.
pub fn log_softmax(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    if x.is_empty() {
        return;
    }

    // Find max
    let mut max_val = x[0];
    for &v in x.iter().skip(1) {
        if v > max_val {
            max_val = v;
        }
    }

    // Compute sum of exp(x - max) using log-sum-exp trick
    let mut sum = 0.0f32;
    for &v in x.iter() {
        sum += (v - max_val).exp();
    }
    let log_sum = sum.ln();

    // Compute log probabilities: x_i - max - log(sum)
    let shift = max_val + log_sum;
    for (i, &v) in x.iter().enumerate() {
        output[i] = v - shift;
    }
}

/// Compute softmax and cross-entropy loss in a single fused operation
///
/// This is more efficient than separate softmax + loss computation because:
/// 1. We only need the probability at the target index for the loss
/// 2. We can use log_softmax for numerical stability
///
/// Returns (loss, probabilities)
pub fn softmax_cross_entropy(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let mut probs = logits.to_vec();
    softmax(&mut probs);
    let loss = -probs[target].max(1e-7).ln();
    (loss, probs)
}

/// Compute cross-entropy loss using log-softmax (more stable)
///
/// This avoids the exp() then ln() round-trip that loses precision.
pub fn stable_cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }

    // Find max for numerical stability
    let mut max_val = logits[0];
    for &v in logits.iter().skip(1) {
        if v > max_val {
            max_val = v;
        }
    }

    // Compute log(sum(exp(x - max)))
    let mut sum = 0.0f32;
    for &v in logits.iter() {
        sum += (v - max_val).exp();
    }
    let log_sum = sum.ln();

    // Loss = -log_prob[target] = -(logits[target] - max - log_sum)
    -(logits[target] - max_val - log_sum)
}

/// Cross-entropy loss for language modeling
pub fn cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    // Apply softmax to get probabilities
    let mut probs = logits.to_vec();
    softmax(&mut probs);

    // Return negative log probability of target
    -probs[target].max(1e-7).ln()
}

/// Compute perplexity from average loss
pub fn perplexity(avg_loss: f32) -> f32 {
    avg_loss.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_activation() {
        let act = Activation::GELU;

        // GELU(0) should be 0
        assert!((act.apply(0.0) - 0.0).abs() < 1e-5);

        // GELU is approximately x for large positive x
        let large = 3.0;
        assert!((act.apply(large) - large).abs() < 0.1);

        // GELU is approximately 0 for large negative x
        let neg = -3.0;
        assert!(act.apply(neg).abs() < 0.01);
    }

    #[test]
    fn test_relu_activation() {
        let act = Activation::ReLU;

        assert_eq!(act.apply(-1.0), 0.0);
        assert_eq!(act.apply(0.0), 0.0);
        assert_eq!(act.apply(1.0), 1.0);
        assert_eq!(act.apply(5.0), 5.0);
    }

    #[test]
    fn test_swish_activation() {
        let act = Activation::Swish;

        // Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((act.apply(0.0) - 0.0).abs() < 1e-5);

        // Swish approaches x for large positive x
        let large = 5.0;
        assert!((act.apply(large) - large).abs() < 0.1);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-5);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax(&mut logits);

        // Should sum to 1
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Higher logit -> higher probability
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Very large values shouldn't cause overflow
        let mut logits = vec![1000.0, 1001.0, 1002.0];
        softmax(&mut logits);

        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let logits = vec![1.0, 2.0, 3.0];

        // Loss should be lowest when target has highest logit
        let loss_target_2 = cross_entropy_loss(&logits, 2);
        let loss_target_0 = cross_entropy_loss(&logits, 0);

        assert!(loss_target_2 < loss_target_0);
    }

    #[test]
    fn test_perplexity() {
        // Perplexity of 0 loss is 1 (perfect prediction)
        assert!((perplexity(0.0) - 1.0).abs() < 1e-5);

        // Perplexity of ln(V) is V (random prediction for vocab size V)
        let vocab_size = 100.0f32;
        assert!((perplexity(vocab_size.ln()) - vocab_size).abs() < 1e-3);
    }

    #[test]
    fn test_xavier_init_range() {
        let fan_in = 512;
        let fan_out = 512;
        let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();

        // Generate many samples and check they're in range
        for _ in 0..100 {
            let w = xavier_init(fan_in, fan_out);
            assert!(w.abs() <= scale);
        }
    }

    #[test]
    fn test_kaiming_init_range() {
        let fan_in = 512;
        let scale = (2.0 / fan_in as f32).sqrt();

        for _ in 0..100 {
            let w = kaiming_init(fan_in);
            assert!(w.abs() <= scale);
        }
    }

    #[test]
    fn test_activation_derivatives() {
        // Test that derivatives are approximately correct using finite differences
        let eps = 1e-4;

        for act in [Activation::GELU, Activation::ReLU, Activation::Swish] {
            for x in [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
                if act == Activation::ReLU && x == 0.0 {
                    continue; // ReLU is not differentiable at 0
                }

                let numerical_deriv = (act.apply(x + eps) - act.apply(x - eps)) / (2.0 * eps);
                let analytical_deriv = act.derivative(x);

                assert!(
                    (numerical_deriv - analytical_deriv).abs() < 0.01,
                    "Derivative mismatch for {:?} at x={}: numerical={}, analytical={}",
                    act, x, numerical_deriv, analytical_deriv
                );
            }
        }
    }

    #[test]
    fn test_log_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut log_probs = vec![0.0; 3];
        log_softmax(&logits, &mut log_probs);

        // Log probabilities should be negative (probs < 1)
        assert!(log_probs.iter().all(|&lp| lp <= 0.0));

        // exp(log_probs) should sum to 1
        let sum: f32 = log_probs.iter().map(|&lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Higher logit -> higher log prob
        assert!(log_probs[2] > log_probs[1]);
        assert!(log_probs[1] > log_probs[0]);
    }

    #[test]
    fn test_stable_cross_entropy_matches_regular() {
        let logits = vec![1.0, 2.0, 3.0, 0.5, -1.0];

        for target in 0..logits.len() {
            let regular = cross_entropy_loss(&logits, target);
            let stable = stable_cross_entropy_loss(&logits, target);

            assert!(
                (regular - stable).abs() < 1e-4,
                "Mismatch for target {}: regular={}, stable={}",
                target,
                regular,
                stable
            );
        }
    }

    #[test]
    fn test_stable_cross_entropy_with_large_values() {
        // Large values that might cause overflow in naive implementation
        let logits = vec![100.0, 101.0, 102.0];

        for target in 0..3 {
            let loss = stable_cross_entropy_loss(&logits, target);
            assert!(loss.is_finite());
            assert!(loss >= 0.0);
        }
    }

    #[test]
    fn test_softmax_cross_entropy_fused() {
        let logits = vec![1.0, 2.0, 3.0];
        let target = 2;

        let (loss, probs) = softmax_cross_entropy(&logits, target);

        // Loss should match cross_entropy_loss
        let expected_loss = cross_entropy_loss(&logits, target);
        assert!((loss - expected_loss).abs() < 1e-5);

        // Probs should be valid softmax output
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_empty() {
        let mut empty: Vec<f32> = vec![];
        softmax(&mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_softmax_single() {
        let mut single = vec![5.0];
        softmax(&mut single);
        assert!((single[0] - 1.0).abs() < 1e-5);
    }
}
