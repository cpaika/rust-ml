//! Optimizers for transformer training
//!
//! This module provides production-grade optimizers including:
//! - SGD with momentum
//! - Adam (Adaptive Moment Estimation)
//! - AdamW (Adam with decoupled weight decay)
//!
//! The Adam optimizer is recommended for most use cases as it provides:
//! - Faster convergence through adaptive learning rates
//! - Better handling of sparse gradients
//! - Built-in momentum for smoother updates

use serde::{Deserialize, Serialize};

/// Optimizer state for a single parameter tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamState {
    /// First moment estimate (momentum)
    pub m: Vec<f32>,
    /// Second moment estimate (RMSprop-like)
    pub v: Vec<f32>,
    /// Number of updates (for bias correction)
    pub t: u64,
}

impl AdamState {
    /// Create new Adam state for a parameter of given size
    pub fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            t: 0,
        }
    }

    /// Reset state (e.g., for fine-tuning)
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.t = 0;
    }
}

/// Adam optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamConfig {
    /// Learning rate (default: 1e-3)
    pub lr: f32,
    /// Exponential decay rate for first moment (default: 0.9)
    pub beta1: f32,
    /// Exponential decay rate for second moment (default: 0.999)
    pub beta2: f32,
    /// Small constant for numerical stability (default: 1e-8)
    pub eps: f32,
    /// Weight decay coefficient (default: 0.01 for AdamW, 0 for Adam)
    pub weight_decay: f32,
    /// Whether to use decoupled weight decay (AdamW style)
    pub decoupled_weight_decay: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_weight_decay: true,
        }
    }
}

impl AdamConfig {
    /// Create AdamW config (with decoupled weight decay)
    pub fn adamw(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            decoupled_weight_decay: true,
            ..Default::default()
        }
    }

    /// Create standard Adam config (no weight decay)
    pub fn adam(lr: f32) -> Self {
        Self {
            lr,
            weight_decay: 0.0,
            decoupled_weight_decay: false,
            ..Default::default()
        }
    }

    /// Create config optimized for transformers
    pub fn transformer_default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.98, // Lower for transformers (more stable)
            eps: 1e-9,   // Smaller for better precision
            weight_decay: 0.01,
            decoupled_weight_decay: true,
        }
    }

    /// Create config for fine-tuning (lower LR)
    pub fn fine_tune(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            decoupled_weight_decay: true,
        }
    }
}

/// Adam/AdamW optimizer
///
/// Implements the Adam algorithm with optional decoupled weight decay (AdamW).
///
/// Adam update rule:
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
/// ```
///
/// For AdamW, weight decay is applied separately:
/// ```text
/// theta = theta - lr * weight_decay * theta
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam {
    /// Configuration
    pub config: AdamConfig,
}

impl Adam {
    /// Create new Adam optimizer
    pub fn new(config: AdamConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(AdamConfig::default())
    }

    /// Perform one optimization step on a parameter
    ///
    /// # Arguments
    /// * `params` - Parameter values (modified in place)
    /// * `grads` - Gradients for the parameters
    /// * `state` - Adam state for this parameter (modified in place)
    /// * `lr_scale` - Optional learning rate multiplier (e.g., for warmup)
    pub fn step(&self, params: &mut [f32], grads: &[f32], state: &mut AdamState, lr_scale: f32) {
        assert_eq!(params.len(), grads.len());
        assert_eq!(params.len(), state.m.len());

        // Increment timestep
        state.t += 1;

        let lr = self.config.lr * lr_scale;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;

        // Bias correction terms
        let bias_correction1 = 1.0 - beta1.powi(state.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(state.t as i32);

        // Precompute the effective learning rate with bias correction
        let step_size = lr / bias_correction1;
        let bias2_sqrt = bias_correction2.sqrt();

        for i in 0..params.len() {
            let g = grads[i];

            // Update first moment (momentum)
            state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g;

            // Update second moment (RMSprop-like)
            state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g;

            // Compute update with bias correction
            // Equivalent to: m_hat / (sqrt(v_hat) + eps)
            let denom = (state.v[i].sqrt() / bias2_sqrt) + eps;
            let update = step_size * state.m[i] / denom;

            // Apply weight decay
            if self.config.decoupled_weight_decay && self.config.weight_decay > 0.0 {
                // AdamW: decoupled weight decay
                params[i] -= lr * self.config.weight_decay * params[i];
            } else if self.config.weight_decay > 0.0 {
                // L2 regularization (add to gradient)
                params[i] -= lr * self.config.weight_decay * params[i];
            }

            // Apply Adam update
            params[i] -= update;
        }
    }

    /// Perform step with gradient clipping
    pub fn step_with_clip(
        &self,
        params: &mut [f32],
        grads: &mut [f32],
        state: &mut AdamState,
        lr_scale: f32,
        max_grad_norm: f32,
    ) {
        // Clip gradients by global norm
        clip_grad_norm(grads, max_grad_norm);

        // Regular Adam step
        self.step(params, grads, state, lr_scale);
    }
}

/// Clip gradients by global L2 norm
///
/// If the L2 norm of gradients exceeds max_norm, scales them down proportionally.
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    // Compute L2 norm
    let norm_sq: f32 = grads.iter().map(|&g| g * g).sum();
    let norm = norm_sq.sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }

    norm
}

/// Compute global gradient norm across multiple parameter tensors
pub fn compute_global_grad_norm(grad_tensors: &[&[f32]]) -> f32 {
    let norm_sq: f32 = grad_tensors
        .iter()
        .flat_map(|g| g.iter())
        .map(|&v| v * v)
        .sum();
    norm_sq.sqrt()
}

/// SGD optimizer with optional momentum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGD {
    /// Learning rate
    pub lr: f32,
    /// Momentum coefficient (0 = no momentum)
    pub momentum: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Nesterov momentum
    pub nesterov: bool,
}

/// State for SGD with momentum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGDState {
    /// Velocity buffer (for momentum)
    pub velocity: Vec<f32>,
}

impl SGDState {
    pub fn new(size: usize) -> Self {
        Self {
            velocity: vec![0.0; size],
        }
    }
}

impl SGD {
    /// Create SGD without momentum
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// Create SGD with momentum
    pub fn with_momentum(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// Create SGD with Nesterov momentum
    pub fn nesterov(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay: 0.0,
            nesterov: true,
        }
    }

    /// Perform one optimization step
    pub fn step(&self, params: &mut [f32], grads: &[f32], state: Option<&mut SGDState>) {
        if self.momentum > 0.0 {
            let state = state.expect("SGD with momentum requires state");
            for i in 0..params.len() {
                let mut g = grads[i];

                // Add weight decay
                if self.weight_decay > 0.0 {
                    g += self.weight_decay * params[i];
                }

                // Update velocity
                state.velocity[i] = self.momentum * state.velocity[i] + g;

                // Apply update
                if self.nesterov {
                    params[i] -= self.lr * (g + self.momentum * state.velocity[i]);
                } else {
                    params[i] -= self.lr * state.velocity[i];
                }
            }
        } else {
            // Simple SGD without momentum
            for i in 0..params.len() {
                let mut g = grads[i];

                if self.weight_decay > 0.0 {
                    g += self.weight_decay * params[i];
                }

                params[i] -= self.lr * g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_state_creation() {
        let state = AdamState::new(100);
        assert_eq!(state.m.len(), 100);
        assert_eq!(state.v.len(), 100);
        assert_eq!(state.t, 0);
        assert!(state.m.iter().all(|&x| x == 0.0));
        assert!(state.v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_adam_config_defaults() {
        let config = AdamConfig::default();
        assert!((config.lr - 1e-3).abs() < 1e-6);
        assert!((config.beta1 - 0.9).abs() < 1e-6);
        assert!((config.beta2 - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_adam_step_decreases_loss() {
        // Simple quadratic: f(x) = x^2, gradient = 2x
        let config = AdamConfig::adam(0.5); // Higher LR for faster convergence
        let optimizer = Adam::new(config);

        let mut params = vec![5.0]; // Start at x=5
        let mut state = AdamState::new(1);
        let initial = params[0];

        // Take several steps toward minimum at x=0
        for _ in 0..200 {
            let grads = vec![2.0 * params[0]]; // gradient of x^2
            optimizer.step(&mut params, &grads, &mut state, 1.0);
        }

        // Should be much closer to 0 than initial
        assert!(
            params[0].abs() < initial.abs() * 0.1,
            "Expected params much smaller than initial, got {} (started at {})",
            params[0],
            initial
        );
    }

    #[test]
    fn test_adam_momentum_accumulation() {
        let config = AdamConfig::adam(0.001);
        let optimizer = Adam::new(config);

        let mut params = vec![1.0];
        let mut state = AdamState::new(1);
        let grads = vec![1.0];

        // Take a step
        optimizer.step(&mut params, &grads, &mut state, 1.0);

        // Momentum should be non-zero
        assert!(state.m[0] > 0.0);
        assert!(state.v[0] > 0.0);
        assert_eq!(state.t, 1);
    }

    #[test]
    fn test_adamw_weight_decay() {
        let config = AdamConfig::adamw(0.1, 0.1);
        let optimizer = Adam::new(config);

        let mut params = vec![1.0];
        let mut state = AdamState::new(1);
        let grads = vec![0.0]; // No gradient, only weight decay

        let initial = params[0];
        optimizer.step(&mut params, &grads, &mut state, 1.0);

        // Weight should decrease due to decay
        assert!(params[0] < initial);
    }

    #[test]
    fn test_clip_grad_norm() {
        let mut grads = vec![3.0, 4.0]; // norm = 5
        let norm = clip_grad_norm(&mut grads, 2.5);

        assert!((norm - 5.0).abs() < 1e-5);

        // Check gradients were scaled
        let new_norm: f32 = grads.iter().map(|&g| g * g).sum::<f32>().sqrt();
        assert!((new_norm - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_basic() {
        let sgd = SGD::new(0.1);
        let mut params = vec![5.0];
        let grads = vec![2.0];

        sgd.step(&mut params, &grads, None);

        // params should decrease by lr * grad = 0.1 * 2 = 0.2
        assert!((params[0] - 4.8).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_momentum() {
        let sgd = SGD::with_momentum(0.1, 0.9);
        let mut params = vec![5.0];
        let grads = vec![1.0];
        let mut state = SGDState::new(1);

        // First step
        sgd.step(&mut params, &grads, Some(&mut state));
        let after_first = params[0];

        // Second step (momentum should accelerate)
        sgd.step(&mut params, &grads, Some(&mut state));
        let diff_second = after_first - params[0];

        // With momentum, second step should be larger
        assert!(diff_second > 0.1); // More than just lr * grad
    }

    #[test]
    fn test_transformer_config() {
        let config = AdamConfig::transformer_default();
        assert!((config.lr - 3e-4).abs() < 1e-6);
        assert!((config.beta2 - 0.98).abs() < 1e-6);
        assert!(config.decoupled_weight_decay);
    }
}
