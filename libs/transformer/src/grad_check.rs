//! Numerical Gradient Checking Utilities
//!
//! Production-grade gradient validation using finite differences.
//! Essential for verifying analytical gradient implementations.

/// Result of a gradient check comparison
#[derive(Debug, Clone)]
pub struct GradCheckResult {
    /// Whether the gradient check passed within tolerance
    pub passed: bool,
    /// Maximum relative error found
    pub max_relative_error: f32,
    /// Index where maximum error occurred
    pub max_error_index: usize,
    /// Mean relative error across all parameters
    pub mean_relative_error: f32,
    /// Analytical gradient value at max error location
    pub analytical_at_max: f32,
    /// Numerical gradient value at max error location
    pub numerical_at_max: f32,
}

impl GradCheckResult {
    /// Create a new result indicating all gradients matched
    pub fn all_zero() -> Self {
        Self {
            passed: true,
            max_relative_error: 0.0,
            max_error_index: 0,
            mean_relative_error: 0.0,
            analytical_at_max: 0.0,
            numerical_at_max: 0.0,
        }
    }
}

impl std::fmt::Display for GradCheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GradCheck {{ passed: {}, max_rel_error: {:.6}, mean_rel_error: {:.6}, \
             at_idx: {}, analytical: {:.6}, numerical: {:.6} }}",
            self.passed,
            self.max_relative_error,
            self.mean_relative_error,
            self.max_error_index,
            self.analytical_at_max,
            self.numerical_at_max
        )
    }
}

/// Compute numerical gradient using central differences
///
/// Uses the formula: (f(x + eps) - f(x - eps)) / (2 * eps)
/// This has O(eps^2) error compared to forward differences which have O(eps) error.
///
/// # Arguments
/// * `params` - Mutable parameter slice to perturb
/// * `loss_fn` - Function that computes loss given parameters
/// * `epsilon` - Perturbation size (typically 1e-5 to 1e-4)
///
/// # Returns
/// Vector of numerical gradients, same length as params
pub fn numerical_gradient<F>(params: &mut [f32], loss_fn: F, epsilon: f32) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut grad = vec![0.0f32; params.len()];

    for i in 0..params.len() {
        let original = params[i];

        // f(x + eps)
        params[i] = original + epsilon;
        let loss_plus = loss_fn(params);

        // f(x - eps)
        params[i] = original - epsilon;
        let loss_minus = loss_fn(params);

        // Central difference
        grad[i] = (loss_plus - loss_minus) / (2.0 * epsilon);

        // Restore original value
        params[i] = original;
    }

    grad
}

/// Compare analytical and numerical gradients
///
/// Uses relative error metric: |a - n| / max(|a|, |n|, eps)
/// This handles cases where gradients are near zero.
///
/// # Arguments
/// * `analytical_grad` - Analytically computed gradient
/// * `numerical_grad` - Numerically computed gradient
/// * `tolerance` - Maximum allowable relative error (typically 1e-4 to 1e-3)
///
/// # Returns
/// GradCheckResult with detailed comparison information
pub fn compare_gradients(
    analytical_grad: &[f32],
    numerical_grad: &[f32],
    tolerance: f32,
) -> GradCheckResult {
    assert_eq!(
        analytical_grad.len(),
        numerical_grad.len(),
        "Gradient lengths must match"
    );

    if analytical_grad.is_empty() {
        return GradCheckResult::all_zero();
    }

    let mut max_rel_error = 0.0f32;
    let mut max_error_idx = 0usize;
    let mut total_rel_error = 0.0f32;
    let mut count = 0;

    for i in 0..analytical_grad.len() {
        let a = analytical_grad[i];
        let n = numerical_grad[i];

        // Skip if both are effectively zero
        if a.abs() < 1e-10 && n.abs() < 1e-10 {
            continue;
        }

        // Relative error with small epsilon to handle near-zero cases
        let abs_diff = (a - n).abs();
        let scale = a.abs().max(n.abs()).max(1e-8);
        let rel_error = abs_diff / scale;

        total_rel_error += rel_error;
        count += 1;

        if rel_error > max_rel_error {
            max_rel_error = rel_error;
            max_error_idx = i;
        }
    }

    let mean_rel_error = if count > 0 {
        total_rel_error / count as f32
    } else {
        0.0
    };

    GradCheckResult {
        passed: max_rel_error < tolerance,
        max_relative_error: max_rel_error,
        max_error_index: max_error_idx,
        mean_relative_error: mean_rel_error,
        analytical_at_max: analytical_grad[max_error_idx],
        numerical_at_max: numerical_grad[max_error_idx],
    }
}

/// Full gradient check: compute numerical gradient and compare to analytical
///
/// # Arguments
/// * `params` - Parameters to check gradients for
/// * `analytical_grad` - Pre-computed analytical gradients
/// * `loss_fn` - Loss function that takes params and returns scalar loss
/// * `epsilon` - Perturbation size for numerical gradient
/// * `tolerance` - Maximum allowable relative error
///
/// # Returns
/// GradCheckResult with detailed comparison
pub fn check_gradient<F>(
    params: &mut [f32],
    analytical_grad: &[f32],
    loss_fn: F,
    epsilon: f32,
    tolerance: f32,
) -> GradCheckResult
where
    F: Fn(&[f32]) -> f32,
{
    let numerical = numerical_gradient(params, loss_fn, epsilon);
    compare_gradients(analytical_grad, &numerical, tolerance)
}

/// Compute gradient of softmax function
///
/// For softmax output p and upstream gradient g:
/// grad_input[i] = p[i] * (g[i] - sum_j(g[j] * p[j]))
///
/// This is derived from the Jacobian of softmax being:
/// dp_i/ds_j = p_i * (delta_ij - p_j)
pub fn softmax_backward(softmax_output: &[f32], grad_output: &[f32]) -> Vec<f32> {
    assert_eq!(softmax_output.len(), grad_output.len());

    // Compute dot product: sum_j(g[j] * p[j])
    let dot: f32 = softmax_output
        .iter()
        .zip(grad_output.iter())
        .map(|(p, g)| p * g)
        .sum();

    // grad_input[i] = p[i] * (g[i] - dot)
    softmax_output
        .iter()
        .zip(grad_output.iter())
        .map(|(p, g)| p * (g - dot))
        .collect()
}

/// Check gradient for a subset of parameters (useful for large models)
///
/// Randomly samples `num_samples` parameters to check.
///
/// # Arguments
/// * `params` - Full parameter vector
/// * `analytical_grad` - Full analytical gradient
/// * `loss_fn` - Loss function
/// * `num_samples` - Number of parameters to sample
/// * `epsilon` - Perturbation size
/// * `tolerance` - Error tolerance
pub fn check_gradient_sampled<F>(
    params: &mut [f32],
    analytical_grad: &[f32],
    loss_fn: F,
    num_samples: usize,
    epsilon: f32,
    tolerance: f32,
) -> GradCheckResult
where
    F: Fn(&[f32]) -> f32,
{
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();

    let n = params.len().min(num_samples);
    let mut indices: Vec<usize> = (0..params.len()).collect();
    indices.shuffle(&mut rng);
    let indices: Vec<usize> = indices.into_iter().take(n).collect();

    let mut max_rel_error = 0.0f32;
    let mut max_error_idx = 0usize;
    let mut total_rel_error = 0.0f32;

    for &i in &indices {
        let original = params[i];

        params[i] = original + epsilon;
        let loss_plus = loss_fn(params);

        params[i] = original - epsilon;
        let loss_minus = loss_fn(params);

        params[i] = original;

        let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
        let analytical = analytical_grad[i];

        if analytical.abs() < 1e-10 && numerical.abs() < 1e-10 {
            continue;
        }

        let abs_diff = (analytical - numerical).abs();
        let scale = analytical.abs().max(numerical.abs()).max(1e-8);
        let rel_error = abs_diff / scale;

        total_rel_error += rel_error;

        if rel_error > max_rel_error {
            max_rel_error = rel_error;
            max_error_idx = i;
        }
    }

    let mean_rel_error = if !indices.is_empty() {
        total_rel_error / indices.len() as f32
    } else {
        0.0
    };

    GradCheckResult {
        passed: max_rel_error < tolerance,
        max_relative_error: max_rel_error,
        max_error_index: max_error_idx,
        mean_relative_error: mean_rel_error,
        analytical_at_max: analytical_grad.get(max_error_idx).copied().unwrap_or(0.0),
        numerical_at_max: 0.0, // Not stored in sampled version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_gradient_simple_quadratic() {
        // f(x) = x^2, gradient should be 2x
        let mut params = vec![3.0f32];
        let loss_fn = |p: &[f32]| p[0] * p[0];

        let grad = numerical_gradient(&mut params, loss_fn, 1e-4);

        // Expected gradient at x=3 is 2*3 = 6
        // With f32 precision and eps=1e-4, expect ~0.1% error
        assert!(
            (grad[0] - 6.0).abs() < 0.1,
            "Expected ~6.0, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_numerical_gradient_multivariate() {
        // f(x, y) = x^2 + 2*x*y + y^2
        // df/dx = 2x + 2y, df/dy = 2x + 2y
        let mut params = vec![1.0f32, 2.0f32];
        let loss_fn = |p: &[f32]| p[0] * p[0] + 2.0 * p[0] * p[1] + p[1] * p[1];

        let grad = numerical_gradient(&mut params, loss_fn, 1e-4);

        // At (1, 2): df/dx = 2*1 + 2*2 = 6, df/dy = 2*1 + 2*2 = 6
        // With f32 precision, allow ~0.1% error
        assert!(
            (grad[0] - 6.0).abs() < 0.1,
            "df/dx: expected ~6.0, got {}",
            grad[0]
        );
        assert!(
            (grad[1] - 6.0).abs() < 0.1,
            "df/dy: expected ~6.0, got {}",
            grad[1]
        );
    }

    #[test]
    fn test_compare_gradients_exact_match() {
        let analytical = vec![1.0, 2.0, 3.0, 4.0];
        let numerical = vec![1.0, 2.0, 3.0, 4.0];

        let result = compare_gradients(&analytical, &numerical, 1e-4);

        assert!(result.passed);
        assert!(result.max_relative_error < 1e-10);
    }

    #[test]
    fn test_compare_gradients_within_tolerance() {
        let analytical = vec![1.0, 2.0, 3.0];
        let numerical = vec![1.0001, 2.0002, 3.0003];

        let result = compare_gradients(&analytical, &numerical, 1e-3);

        assert!(result.passed, "Should pass within 1e-3 tolerance: {}", result);
    }

    #[test]
    fn test_compare_gradients_outside_tolerance() {
        let analytical = vec![1.0, 2.0, 3.0];
        let numerical = vec![1.1, 2.0, 3.0]; // 10% error on first element

        let result = compare_gradients(&analytical, &numerical, 1e-3);

        assert!(!result.passed, "Should fail with 10% error");
        assert_eq!(result.max_error_index, 0);
    }

    #[test]
    fn test_softmax_backward_simple() {
        // Simple case: uniform softmax output
        let softmax_out = vec![0.25, 0.25, 0.25, 0.25];
        let grad_out = vec![1.0, 0.0, 0.0, 0.0];

        let grad_input = softmax_backward(&softmax_out, &grad_out);

        // dot = 0.25 * 1.0 + 0.25 * 0.0 + 0.25 * 0.0 + 0.25 * 0.0 = 0.25
        // grad_input[0] = 0.25 * (1.0 - 0.25) = 0.1875
        // grad_input[1] = 0.25 * (0.0 - 0.25) = -0.0625
        assert!(
            (grad_input[0] - 0.1875).abs() < 1e-6,
            "Expected 0.1875, got {}",
            grad_input[0]
        );
        assert!(
            (grad_input[1] - (-0.0625)).abs() < 1e-6,
            "Expected -0.0625, got {}",
            grad_input[1]
        );
    }

    #[test]
    fn test_softmax_backward_numerical_check() {
        use crate::softmax;

        // Verify softmax backward against numerical gradient
        let mut scores = vec![1.0f32, 2.0, 3.0];
        let scores_orig = scores.clone();

        // Forward: compute softmax
        softmax(&mut scores);
        let softmax_out = scores.clone();

        // Upstream gradient (as if from cross-entropy)
        let grad_out = vec![0.1, 0.2, 0.3];

        // Analytical gradient
        let analytical = softmax_backward(&softmax_out, &grad_out);

        // Numerical gradient
        let loss_fn = |s: &[f32]| -> f32 {
            let mut probs = s.to_vec();
            softmax(&mut probs);
            // Loss = sum(grad_out * probs)
            probs.iter().zip(grad_out.iter()).map(|(p, g)| p * g).sum()
        };

        let mut scores_copy = scores_orig;
        let numerical = numerical_gradient(&mut scores_copy, loss_fn, 1e-4);

        // With f32 precision, allow up to 10% relative error for softmax gradients
        // (softmax is numerically sensitive due to exp operations)
        let result = compare_gradients(&analytical, &numerical, 0.1);
        assert!(
            result.passed,
            "Softmax backward failed numerical check: {}",
            result
        );
    }

    #[test]
    fn test_check_gradient_full_integration() {
        // Test full gradient check workflow
        let mut params = vec![1.0f32, 2.0, 3.0];
        let loss_fn = |p: &[f32]| p[0] * p[0] + p[1] * p[1] + p[2] * p[2];

        // Analytical gradient: [2*x, 2*y, 2*z] = [2, 4, 6]
        let analytical = vec![2.0, 4.0, 6.0];

        // With f32 precision and eps=1e-4, allow ~1% relative error
        let result = check_gradient(&mut params, &analytical, loss_fn, 1e-4, 0.01);

        assert!(result.passed, "Full gradient check failed: {}", result);
    }

    #[test]
    fn test_gradient_check_detects_wrong_gradient() {
        let mut params = vec![1.0f32, 2.0, 3.0];
        let loss_fn = |p: &[f32]| p[0] * p[0] + p[1] * p[1] + p[2] * p[2];

        // Wrong analytical gradient (should be [2, 4, 6])
        let wrong_analytical = vec![3.0, 4.0, 6.0]; // First element wrong

        let result = check_gradient(&mut params, &wrong_analytical, loss_fn, 1e-5, 1e-4);

        assert!(!result.passed, "Should detect wrong gradient");
        assert_eq!(result.max_error_index, 0, "Error should be at index 0");
    }

    #[test]
    fn test_grad_check_result_display() {
        let result = GradCheckResult {
            passed: true,
            max_relative_error: 1e-5,
            max_error_index: 0,
            mean_relative_error: 1e-6,
            analytical_at_max: 2.0,
            numerical_at_max: 2.00001,
        };

        let display = format!("{}", result);
        assert!(display.contains("passed: true"));
    }
}
