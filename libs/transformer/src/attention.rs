//! Multi-head self-attention mechanism

use crate::{softmax, xavier_init, TransformerConfig};
use serde::{Deserialize, Serialize};

/// Multi-head self-attention layer
///
/// Implements scaled dot-product attention with multiple heads.
/// For decoder-only transformers, uses causal (masked) attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    /// Query projection weights [d_model, d_model]
    pub w_q: Vec<f32>,
    /// Key projection weights [d_model, d_model]
    pub w_k: Vec<f32>,
    /// Value projection weights [d_model, d_model]
    pub w_v: Vec<f32>,
    /// Output projection weights [d_model, d_model]
    pub w_o: Vec<f32>,

    /// Query bias [d_model]
    pub b_q: Vec<f32>,
    /// Key bias [d_model]
    pub b_k: Vec<f32>,
    /// Value bias [d_model]
    pub b_v: Vec<f32>,
    /// Output bias [d_model]
    pub b_o: Vec<f32>,

    /// Number of attention heads
    pub n_heads: usize,
    /// Model dimension
    pub d_model: usize,
    /// Dimension per head
    pub d_head: usize,
    /// Scaling factor (1/sqrt(d_head))
    pub scale: f32,

    // Cached values for backpropagation
    #[serde(skip)]
    cached_input: Vec<f32>,
    #[serde(skip)]
    cached_q: Vec<f32>,
    #[serde(skip)]
    cached_k: Vec<f32>,
    #[serde(skip)]
    cached_v: Vec<f32>,
    #[serde(skip)]
    cached_attn_weights: Vec<f32>,
    #[serde(skip)]
    cached_attn_output: Vec<f32>,
}

impl MultiHeadAttention {
    /// Create new multi-head attention layer
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        let d_head = d_model / n_heads;
        let scale = 1.0 / (d_head as f32).sqrt();

        // Initialize weights with Xavier initialization
        let w_q: Vec<f32> = (0..d_model * d_model)
            .map(|_| xavier_init(d_model, d_model))
            .collect();
        let w_k: Vec<f32> = (0..d_model * d_model)
            .map(|_| xavier_init(d_model, d_model))
            .collect();
        let w_v: Vec<f32> = (0..d_model * d_model)
            .map(|_| xavier_init(d_model, d_model))
            .collect();
        let w_o: Vec<f32> = (0..d_model * d_model)
            .map(|_| xavier_init(d_model, d_model))
            .collect();

        // Initialize biases to zero
        let b_q = vec![0.0; d_model];
        let b_k = vec![0.0; d_model];
        let b_v = vec![0.0; d_model];
        let b_o = vec![0.0; d_model];

        Self {
            w_q, w_k, w_v, w_o,
            b_q, b_k, b_v, b_o,
            n_heads,
            d_model,
            d_head,
            scale,
            cached_input: Vec::new(),
            cached_q: Vec::new(),
            cached_k: Vec::new(),
            cached_v: Vec::new(),
            cached_attn_weights: Vec::new(),
            cached_attn_output: Vec::new(),
        }
    }

    /// Create from config
    pub fn from_config(config: &TransformerConfig) -> Self {
        Self::new(config.d_model, config.n_heads)
    }

    /// Forward pass with causal masking
    ///
    /// Input: [seq_len, d_model] as flat vector
    /// Output: [seq_len, d_model] as flat vector
    pub fn forward(&mut self, x: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(x.len(), seq_len * self.d_model);

        // Cache input for backward pass
        self.cached_input = x.to_vec();

        // Compute Q, K, V projections
        let q = self.linear(x, &self.w_q, &self.b_q, seq_len);
        let k = self.linear(x, &self.w_k, &self.b_k, seq_len);
        let v = self.linear(x, &self.w_v, &self.b_v, seq_len);

        // Compute attention
        let attn_output = self.scaled_dot_product_attention(&q, &k, &v, seq_len, true);

        // Output projection
        let output = self.linear(&attn_output, &self.w_o, &self.b_o, seq_len);

        // Cache for backward pass
        self.cached_q = q;
        self.cached_k = k;
        self.cached_v = v;
        self.cached_attn_output = attn_output;

        output
    }

    /// Linear projection: y = x * W^T + b
    fn linear(&self, x: &[f32], w: &[f32], b: &[f32], seq_len: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * self.d_model];

        for pos in 0..seq_len {
            for out_dim in 0..self.d_model {
                let mut sum = b[out_dim];
                for in_dim in 0..self.d_model {
                    // W is stored as [d_model, d_model] row-major
                    // y[pos, out_dim] = sum(x[pos, in_dim] * W[out_dim, in_dim])
                    sum += x[pos * self.d_model + in_dim] * w[out_dim * self.d_model + in_dim];
                }
                output[pos * self.d_model + out_dim] = sum;
            }
        }

        output
    }

    /// Scaled dot-product attention with optional causal masking
    ///
    /// Q, K, V: [seq_len, d_model]
    /// Output: [seq_len, d_model]
    fn scaled_dot_product_attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        causal: bool,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * self.d_model];
        let mut all_attn_weights = vec![0.0f32; self.n_heads * seq_len * seq_len];

        // Process each head
        for head in 0..self.n_heads {
            let head_offset = head * self.d_head;

            // Compute attention scores for this head
            // scores[i, j] = Q[i, head] dot K[j, head] * scale
            for i in 0..seq_len {
                let mut scores = vec![0.0f32; seq_len];

                for j in 0..seq_len {
                    // Apply causal mask: can only attend to positions <= current
                    if causal && j > i {
                        scores[j] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0;
                        for d in 0..self.d_head {
                            let q_idx = i * self.d_model + head_offset + d;
                            let k_idx = j * self.d_model + head_offset + d;
                            dot += q[q_idx] * k[k_idx];
                        }
                        scores[j] = dot * self.scale;
                    }
                }

                // Softmax over scores
                softmax(&mut scores);

                // Store attention weights for caching/visualization
                for j in 0..seq_len {
                    all_attn_weights[head * seq_len * seq_len + i * seq_len + j] = scores[j];
                }

                // Weighted sum of values
                for d in 0..self.d_head {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        let v_idx = j * self.d_model + head_offset + d;
                        sum += scores[j] * v[v_idx];
                    }
                    output[i * self.d_model + head_offset + d] = sum;
                }
            }
        }

        self.cached_attn_weights = all_attn_weights;
        output
    }

    /// Get cached attention weights for visualization
    /// Returns [n_heads, seq_len, seq_len] as flat vector
    pub fn get_attention_weights(&self) -> &[f32] {
        &self.cached_attn_weights
    }

    /// Get attention weights for a specific head
    /// Returns [seq_len, seq_len] as flat vector
    pub fn get_head_attention(&self, head: usize, seq_len: usize) -> Vec<f32> {
        let start = head * seq_len * seq_len;
        let end = start + seq_len * seq_len;
        self.cached_attn_weights[start..end].to_vec()
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        // 4 weight matrices + 4 bias vectors
        4 * self.d_model * self.d_model + 4 * self.d_model
    }

    /// Get all weights as a flat slice for serialization
    pub fn all_weights(&self) -> Vec<&[f32]> {
        vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o]
    }

    /// Get all biases as a flat slice
    pub fn all_biases(&self) -> Vec<&[f32]> {
        vec![&self.b_q, &self.b_k, &self.b_v, &self.b_o]
    }

    /// Backward pass through multi-head attention
    ///
    /// Computes gradients for all weights and biases, and returns gradient w.r.t. input.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient w.r.t. output [seq_len, d_model]
    /// * `seq_len` - Sequence length
    /// * `learning_rate` - Learning rate for weight updates
    ///
    /// # Returns
    /// Gradient w.r.t. input [seq_len, d_model]
    ///
    /// # Mathematical Details
    ///
    /// Forward pass:
    /// 1. Q = x @ W_q^T + b_q
    /// 2. K = x @ W_k^T + b_k
    /// 3. V = x @ W_v^T + b_v
    /// 4. Scores = Q @ K^T * scale (per head, with causal mask)
    /// 5. Attn = softmax(Scores)
    /// 6. Context = Attn @ V
    /// 7. Output = Context @ W_o^T + b_o
    ///
    /// Backward pass reverses each step, computing gradients via chain rule.
    pub fn backward(
        &mut self,
        grad_output: &[f32],
        seq_len: usize,
        learning_rate: f32,
    ) -> Vec<f32> {
        assert_eq!(grad_output.len(), seq_len * self.d_model);
        assert!(!self.cached_input.is_empty(), "Must call forward() before backward()");

        // Step 1: Backward through output projection
        // Output = Context @ W_o^T + b_o
        // grad_context = grad_output @ W_o
        // grad_W_o = grad_output^T @ Context (in row-major: Context^T @ grad_output)
        // grad_b_o = sum(grad_output, axis=0)
        let (grad_context, grad_w_o, grad_b_o) =
            self.backward_output_projection(grad_output, seq_len);

        // Step 2: Backward through attention computation (per head)
        // Context[h] = Attn[h] @ V[h]
        // Attn[h] = softmax(Q[h] @ K[h]^T * scale)
        let (grad_q, grad_k, grad_v) = self.backward_attention(&grad_context, seq_len);

        // Step 3: Backward through QKV projections
        // Q = x @ W_q^T + b_q => grad_x_q = grad_Q @ W_q, grad_W_q = grad_Q^T @ x
        let grad_input = self.backward_qkv_projections(&grad_q, &grad_k, &grad_v, seq_len);

        // Step 4: Apply gradient updates
        self.apply_gradients(
            learning_rate,
            seq_len,
            &grad_q,
            &grad_k,
            &grad_v,
            &grad_w_o,
            &grad_b_o,
        );

        grad_input
    }

    /// Backward through output projection: Output = Context @ W_o^T + b_o
    fn backward_output_projection(
        &self,
        grad_output: &[f32],
        seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut grad_context = vec![0.0f32; seq_len * self.d_model];
        let mut grad_w_o = vec![0.0f32; self.d_model * self.d_model];
        let mut grad_b_o = vec![0.0f32; self.d_model];

        // grad_context = grad_output @ W_o
        // W_o is stored as [out_dim, in_dim] = [d_model, d_model]
        // So grad_context[pos, in_dim] = sum_out(grad_output[pos, out_dim] * W_o[out_dim, in_dim])
        for pos in 0..seq_len {
            for in_dim in 0..self.d_model {
                let mut sum = 0.0;
                for out_dim in 0..self.d_model {
                    sum += grad_output[pos * self.d_model + out_dim]
                        * self.w_o[out_dim * self.d_model + in_dim];
                }
                grad_context[pos * self.d_model + in_dim] = sum;
            }
        }

        // grad_W_o[out_dim, in_dim] = sum_pos(grad_output[pos, out_dim] * context[pos, in_dim])
        for out_dim in 0..self.d_model {
            for in_dim in 0..self.d_model {
                let mut sum = 0.0;
                for pos in 0..seq_len {
                    sum += grad_output[pos * self.d_model + out_dim]
                        * self.cached_attn_output[pos * self.d_model + in_dim];
                }
                grad_w_o[out_dim * self.d_model + in_dim] = sum;
            }
        }

        // grad_b_o = sum over positions
        for pos in 0..seq_len {
            for d in 0..self.d_model {
                grad_b_o[d] += grad_output[pos * self.d_model + d];
            }
        }

        (grad_context, grad_w_o, grad_b_o)
    }

    /// Backward through multi-head attention computation
    ///
    /// For each head h:
    /// - Context[h] = Attn[h] @ V[h]
    /// - Attn[h] = softmax(Scores[h])
    /// - Scores[h] = Q[h] @ K[h]^T * scale
    fn backward_attention(
        &self,
        grad_context: &[f32],
        seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut grad_q = vec![0.0f32; seq_len * self.d_model];
        let mut grad_k = vec![0.0f32; seq_len * self.d_model];
        let mut grad_v = vec![0.0f32; seq_len * self.d_model];

        // Process each head independently
        for head in 0..self.n_heads {
            let head_offset = head * self.d_head;

            // For each query position
            for i in 0..seq_len {
                // Get attention weights for this position (row i)
                let attn_start = head * seq_len * seq_len + i * seq_len;
                let attn_row = &self.cached_attn_weights[attn_start..attn_start + seq_len];

                // Get grad_context for this head and position
                let gc_start = i * self.d_model + head_offset;
                let grad_context_h: Vec<f32> =
                    grad_context[gc_start..gc_start + self.d_head].to_vec();

                // Step 1: Backward through Context = Attn @ V
                // grad_Attn[i, :] = grad_Context[i] @ V^T
                // grad_V[j] += Attn[i, j] * grad_Context[i]
                let mut grad_attn_row = vec![0.0f32; seq_len];

                for j in 0..=i {
                    // Only positions we attended to (causal)
                    // grad_V[j] += attn[i,j] * grad_context[i]
                    for d in 0..self.d_head {
                        grad_v[j * self.d_model + head_offset + d] +=
                            attn_row[j] * grad_context_h[d];
                    }

                    // grad_attn[i,j] = grad_context[i] dot V[j]
                    for d in 0..self.d_head {
                        grad_attn_row[j] +=
                            grad_context_h[d] * self.cached_v[j * self.d_model + head_offset + d];
                    }
                }

                // Step 2: Backward through softmax
                // grad_scores = attn * (grad_attn - dot(grad_attn, attn))
                let dot_product: f32 =
                    (0..=i).map(|j| grad_attn_row[j] * attn_row[j]).sum();

                let mut grad_scores_row = vec![0.0f32; seq_len];
                for j in 0..=i {
                    grad_scores_row[j] = attn_row[j] * (grad_attn_row[j] - dot_product);
                }

                // Step 3: Backward through Scores = Q @ K^T * scale
                // grad_Q[i] = sum_j(grad_scores[i,j] * K[j]) * scale
                // grad_K[j] += grad_scores[i,j] * Q[i] * scale
                for j in 0..=i {
                    let scaled_grad = grad_scores_row[j] * self.scale;

                    for d in 0..self.d_head {
                        // grad_Q[i] += grad_scores[i,j] * K[j] * scale
                        grad_q[i * self.d_model + head_offset + d] +=
                            scaled_grad * self.cached_k[j * self.d_model + head_offset + d];

                        // grad_K[j] += grad_scores[i,j] * Q[i] * scale
                        grad_k[j * self.d_model + head_offset + d] +=
                            scaled_grad * self.cached_q[i * self.d_model + head_offset + d];
                    }
                }
            }
        }

        (grad_q, grad_k, grad_v)
    }

    /// Backward through QKV projections
    ///
    /// Q = x @ W_q^T + b_q
    /// grad_x_q = grad_Q @ W_q
    /// grad_W_q = grad_Q^T @ x (accumulated externally)
    fn backward_qkv_projections(
        &self,
        grad_q: &[f32],
        grad_k: &[f32],
        grad_v: &[f32],
        seq_len: usize,
    ) -> Vec<f32> {
        let mut grad_input = vec![0.0f32; seq_len * self.d_model];

        // For each projection (Q, K, V), compute grad_input contribution
        for (grad_proj, w) in [
            (grad_q, &self.w_q),
            (grad_k, &self.w_k),
            (grad_v, &self.w_v),
        ] {
            // grad_input += grad_proj @ W
            // W is stored as [out_dim, in_dim]
            for pos in 0..seq_len {
                for in_dim in 0..self.d_model {
                    let mut sum = 0.0;
                    for out_dim in 0..self.d_model {
                        sum +=
                            grad_proj[pos * self.d_model + out_dim] * w[out_dim * self.d_model + in_dim];
                    }
                    grad_input[pos * self.d_model + in_dim] += sum;
                }
            }
        }

        grad_input
    }

    /// Apply gradient updates to all weights
    fn apply_gradients(
        &mut self,
        learning_rate: f32,
        seq_len: usize,
        grad_q: &[f32],
        grad_k: &[f32],
        grad_v: &[f32],
        grad_w_o: &[f32],
        grad_b_o: &[f32],
    ) {
        let batch_lr = learning_rate / seq_len as f32;

        // Compute and apply gradients for W_q, W_k, W_v
        for (grad_proj, w, b) in [
            (grad_q, &mut self.w_q, &mut self.b_q),
            (grad_k, &mut self.w_k, &mut self.b_k),
            (grad_v, &mut self.w_v, &mut self.b_v),
        ] {
            // grad_W[out_dim, in_dim] = sum_pos(grad_proj[pos, out_dim] * input[pos, in_dim])
            for out_dim in 0..self.d_model {
                for in_dim in 0..self.d_model {
                    let mut grad_sum = 0.0;
                    for pos in 0..seq_len {
                        grad_sum += grad_proj[pos * self.d_model + out_dim]
                            * self.cached_input[pos * self.d_model + in_dim];
                    }
                    w[out_dim * self.d_model + in_dim] -= batch_lr * grad_sum;
                }
            }

            // grad_b = sum over positions
            for d in 0..self.d_model {
                let mut grad_sum = 0.0;
                for pos in 0..seq_len {
                    grad_sum += grad_proj[pos * self.d_model + d];
                }
                b[d] -= batch_lr * grad_sum;
            }
        }

        // Apply gradients for W_o, b_o
        for i in 0..self.w_o.len() {
            self.w_o[i] -= batch_lr * grad_w_o[i];
        }
        for i in 0..self.b_o.len() {
            self.b_o[i] -= batch_lr * grad_b_o[i];
        }
    }
}

/// Attention pattern types for visualization
#[derive(Debug, Clone, Copy)]
pub enum AttentionPattern {
    /// Causal (autoregressive) - can only attend to past
    Causal,
    /// Full attention - can attend to all positions
    Full,
    /// Sliding window attention
    SlidingWindow { window_size: usize },
}

/// Create a causal attention mask
/// Returns [seq_len, seq_len] where mask[i][j] = true if position i can attend to j
pub fn create_causal_mask(seq_len: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = true;
        }
    }
    mask
}

/// Create a sliding window attention mask
pub fn create_sliding_window_mask(seq_len: usize, window_size: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * seq_len];
    for i in 0..seq_len {
        let start = i.saturating_sub(window_size);
        for j in start..=i {
            mask[i * seq_len + j] = true;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let attn = MultiHeadAttention::new(64, 4);
        assert_eq!(attn.d_model, 64);
        assert_eq!(attn.n_heads, 4);
        assert_eq!(attn.d_head, 16);
    }

    #[test]
    fn test_attention_from_config() {
        let config = TransformerConfig::tiny(1000);
        let attn = MultiHeadAttention::from_config(&config);
        assert_eq!(attn.d_model, config.d_model);
        assert_eq!(attn.n_heads, config.n_heads);
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn test_attention_invalid_heads() {
        MultiHeadAttention::new(64, 5); // 64 not divisible by 5
    }

    #[test]
    fn test_attention_forward() {
        let mut attn = MultiHeadAttention::new(32, 2);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 32];

        let output = attn.forward(&x, seq_len);
        assert_eq!(output.len(), seq_len * 32);
    }

    #[test]
    fn test_attention_forward_output_shape() {
        let mut attn = MultiHeadAttention::new(64, 4);
        let seq_len = 8;
        let x: Vec<f32> = (0..seq_len * 64).map(|i| i as f32 * 0.01).collect();

        let output = attn.forward(&x, seq_len);
        assert_eq!(output.len(), seq_len * 64);
    }

    #[test]
    fn test_attention_weights_cached() {
        let mut attn = MultiHeadAttention::new(32, 2);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 32];

        attn.forward(&x, seq_len);

        let weights = attn.get_attention_weights();
        assert_eq!(weights.len(), 2 * seq_len * seq_len); // n_heads * seq_len * seq_len
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let mut attn = MultiHeadAttention::new(32, 2);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 32];

        attn.forward(&x, seq_len);

        // Check that attention weights sum to 1 for each query position
        for head in 0..2 {
            for i in 0..seq_len {
                let start = head * seq_len * seq_len + i * seq_len;
                let row_sum: f32 = attn.cached_attn_weights[start..start + seq_len].iter().sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "Attention weights should sum to 1, got {}",
                    row_sum
                );
            }
        }
    }

    #[test]
    fn test_causal_masking() {
        let mut attn = MultiHeadAttention::new(32, 2);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 32];

        attn.forward(&x, seq_len);

        // Check causal masking: position i should not attend to j > i
        for head in 0..2 {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    let idx = head * seq_len * seq_len + i * seq_len + j;
                    assert!(
                        attn.cached_attn_weights[idx] < 1e-6,
                        "Causal mask violated: position {} attended to {}",
                        i, j
                    );
                }
            }
        }
    }

    #[test]
    fn test_get_head_attention() {
        let mut attn = MultiHeadAttention::new(32, 2);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 32];

        attn.forward(&x, seq_len);

        let head0 = attn.get_head_attention(0, seq_len);
        let head1 = attn.get_head_attention(1, seq_len);

        assert_eq!(head0.len(), seq_len * seq_len);
        assert_eq!(head1.len(), seq_len * seq_len);
    }

    #[test]
    fn test_num_parameters() {
        let attn = MultiHeadAttention::new(64, 4);
        let params = attn.num_parameters();

        // 4 * (64*64) + 4 * 64 = 16384 + 256 = 16640
        assert_eq!(params, 4 * 64 * 64 + 4 * 64);
    }

    #[test]
    fn test_linear_projection() {
        let attn = MultiHeadAttention::new(32, 2);
        let seq_len = 2;
        let x = vec![1.0f32; seq_len * 32];

        let result = attn.linear(&x, &attn.w_q, &attn.b_q, seq_len);
        assert_eq!(result.len(), seq_len * 32);
    }

    #[test]
    fn test_create_causal_mask() {
        let mask = create_causal_mask(4);

        // Row 0: can only see position 0
        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(!mask[2]);
        assert!(!mask[3]);

        // Row 3: can see positions 0, 1, 2, 3
        assert!(mask[12]);
        assert!(mask[13]);
        assert!(mask[14]);
        assert!(mask[15]);
    }

    #[test]
    fn test_create_sliding_window_mask() {
        let mask = create_sliding_window_mask(6, 2);

        // Position 3 with window 2: can see positions 1, 2, 3
        let row3_start = 3 * 6;
        assert!(!mask[row3_start + 0]); // Can't see position 0
        assert!(mask[row3_start + 1]);  // Can see position 1
        assert!(mask[row3_start + 2]);  // Can see position 2
        assert!(mask[row3_start + 3]);  // Can see position 3 (self)
        assert!(!mask[row3_start + 4]); // Future
        assert!(!mask[row3_start + 5]); // Future
    }

    #[test]
    fn test_scale_factor() {
        let attn = MultiHeadAttention::new(64, 4);
        // d_head = 64/4 = 16, scale = 1/sqrt(16) = 0.25
        assert!((attn.scale - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_attention_deterministic_with_fixed_input() {
        let mut attn1 = MultiHeadAttention::new(32, 2);
        let mut attn2 = attn1.clone();

        let x = vec![0.5f32; 4 * 32];

        let out1 = attn1.forward(&x, 4);
        let out2 = attn2.forward(&x, 4);

        // Same weights + same input should give same output
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_different_positions_different_outputs() {
        let mut attn = MultiHeadAttention::new(32, 2);

        // Create input where different positions have different values
        let mut x = vec![0.0f32; 4 * 32];
        for i in 0..4 {
            for j in 0..32 {
                x[i * 32 + j] = (i * 32 + j) as f32 * 0.01;
            }
        }

        let output = attn.forward(&x, 4);

        // First and last positions should have different outputs
        let pos0: Vec<f32> = output[0..32].to_vec();
        let pos3: Vec<f32> = output[3 * 32..4 * 32].to_vec();

        let all_same = pos0.iter().zip(pos3.iter()).all(|(a, b)| (a - b).abs() < 1e-5);
        assert!(!all_same, "Different positions should produce different outputs");
    }

    // ==================== BACKWARD PASS TESTS ====================

    #[test]
    fn test_attention_backward_output_shape() {
        let mut attn = MultiHeadAttention::new(32, 2);
        let seq_len = 4;
        let x = vec![0.1f32; seq_len * 32];

        attn.forward(&x, seq_len);

        let grad_output = vec![0.01f32; seq_len * 32];
        let grad_input = attn.backward(&grad_output, seq_len, 0.0);

        assert_eq!(grad_input.len(), seq_len * 32);
    }

    #[test]
    fn test_attention_backward_weights_update() {
        let mut attn = MultiHeadAttention::new(16, 2);
        let seq_len = 3;
        let x: Vec<f32> = (0..seq_len * 16).map(|i| i as f32 * 0.01).collect();

        let w_q_before = attn.w_q.clone();
        let w_k_before = attn.w_k.clone();
        let w_v_before = attn.w_v.clone();
        let w_o_before = attn.w_o.clone();

        attn.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * 16).map(|i| i as f32 * 0.001).collect();
        attn.backward(&grad_output, seq_len, 0.1);

        // All weights should have changed
        let w_q_changed = attn.w_q.iter().zip(w_q_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        let w_k_changed = attn.w_k.iter().zip(w_k_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        let w_v_changed = attn.w_v.iter().zip(w_v_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        let w_o_changed = attn.w_o.iter().zip(w_o_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(w_q_changed, "W_q should be updated");
        assert!(w_k_changed, "W_k should be updated");
        assert!(w_v_changed, "W_v should be updated");
        assert!(w_o_changed, "W_o should be updated");
    }

    #[test]
    fn test_attention_backward_biases_update() {
        let mut attn = MultiHeadAttention::new(16, 2);
        let seq_len = 3;
        let x: Vec<f32> = (0..seq_len * 16).map(|i| i as f32 * 0.01).collect();

        let b_q_before = attn.b_q.clone();
        let b_o_before = attn.b_o.clone();

        attn.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * 16).map(|i| i as f32 * 0.001).collect();
        attn.backward(&grad_output, seq_len, 0.1);

        // Biases should have changed
        let b_q_changed = attn.b_q.iter().zip(b_q_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        let b_o_changed = attn.b_o.iter().zip(b_o_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(b_q_changed, "b_q should be updated");
        assert!(b_o_changed, "b_o should be updated");
    }

    #[test]
    fn test_attention_backward_no_update_with_zero_lr() {
        let mut attn = MultiHeadAttention::new(16, 2);
        let seq_len = 3;
        let x: Vec<f32> = (0..seq_len * 16).map(|i| i as f32 * 0.01).collect();

        let w_q_before = attn.w_q.clone();

        attn.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * 16).map(|i| i as f32 * 0.001).collect();
        attn.backward(&grad_output, seq_len, 0.0); // Zero learning rate

        // Weights should NOT have changed
        let w_q_same = attn.w_q.iter().zip(w_q_before.iter()).all(|(a, b)| (a - b).abs() < 1e-10);
        assert!(w_q_same, "W_q should NOT be updated with lr=0");
    }

    #[test]
    fn test_attention_backward_grad_input_nonzero() {
        let mut attn = MultiHeadAttention::new(16, 2);
        let seq_len = 3;
        let x: Vec<f32> = (0..seq_len * 16).map(|i| i as f32 * 0.01).collect();

        attn.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * 16).map(|i| (i + 1) as f32 * 0.01).collect();
        let grad_input = attn.backward(&grad_output, seq_len, 0.0);

        // Gradient input should have non-zero values
        let has_nonzero = grad_input.iter().any(|&g| g.abs() > 1e-10);
        assert!(has_nonzero, "Gradient input should have non-zero values");
    }

    #[test]
    fn test_attention_backward_multi_head_gradients() {
        let mut attn = MultiHeadAttention::new(32, 4); // 4 heads
        let seq_len = 4;
        let x: Vec<f32> = (0..seq_len * 32).map(|i| i as f32 * 0.001).collect();

        attn.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * 32).map(|i| i as f32 * 0.0001).collect();
        let grad_input = attn.backward(&grad_output, seq_len, 0.0);

        // Each head should contribute to the gradient
        let d_head = 32 / 4;
        for h in 0..4 {
            let head_grad: f32 = (0..seq_len)
                .map(|pos| {
                    (0..d_head)
                        .map(|d| grad_input[pos * 32 + h * d_head + d].abs())
                        .sum::<f32>()
                })
                .sum();
            assert!(head_grad > 1e-10, "Head {} should have non-zero gradient", h);
        }
    }

    #[test]
    fn test_attention_backward_numerical_gradient_w_o() {
        use crate::grad_check::{check_gradient_sampled, GradCheckResult};

        let d_model = 8;
        let n_heads = 2;
        let seq_len = 2;

        let mut attn = MultiHeadAttention::new(d_model, n_heads);
        let x: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32) * 0.1).collect();
        let grad_output: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32) * 0.01).collect();

        // Compute analytical gradient for w_o
        attn.forward(&x, seq_len);

        // For W_o, we need to compute grad_W_o analytically
        let (_, grad_w_o, _) = attn.backward_output_projection(&grad_output, seq_len);

        // Numerical gradient
        let loss_fn = |w: &[f32]| -> f32 {
            let mut attn_copy = attn.clone();
            attn_copy.w_o = w.to_vec();
            let output = attn_copy.forward(&x, seq_len);
            // Loss = grad_output dot output
            output.iter().zip(grad_output.iter()).map(|(o, g)| o * g).sum()
        };

        let result = check_gradient_sampled(
            &mut attn.w_o.clone(),
            &grad_w_o,
            loss_fn,
            20,  // Sample 20 parameters
            1e-4,
            0.1, // 10% tolerance for f32
        );

        assert!(
            result.passed,
            "W_o numerical gradient check failed: max_error={:.4}, mean_error={:.4}",
            result.max_relative_error,
            result.mean_relative_error
        );
    }
}
