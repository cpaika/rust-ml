//! GPU-accelerated forward pass for transformer inference
//!
//! Provides a complete forward pass using GPU compute shaders

use super::ops::GpuOps;
use super::GpuContext;
use crate::TransformerConfig;
use std::sync::Arc;

/// GPU-accelerated transformer for inference
pub struct GpuTransformer {
    ops: GpuOps,
    config: TransformerConfig,

    // Weights (stored on CPU, uploaded to GPU per operation)
    token_embeddings: Vec<f32>,     // [vocab_size, d_model]
    position_embeddings: Vec<f32>,  // [max_seq_len, d_model]

    // Per-layer weights
    layer_weights: Vec<LayerWeights>,

    // Final layer norm and projection
    final_ln_gamma: Vec<f32>,
    final_ln_beta: Vec<f32>,
    output_projection: Vec<f32>,    // [d_model, vocab_size]
}

/// Weights for a single transformer layer
pub struct LayerWeights {
    // Attention
    q_weight: Vec<f32>,     // [d_model, d_model]
    k_weight: Vec<f32>,     // [d_model, d_model]
    v_weight: Vec<f32>,     // [d_model, d_model]
    o_weight: Vec<f32>,     // [d_model, d_model]

    // Pre-attention layer norm
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,

    // FFN
    ffn_up: Vec<f32>,       // [d_model, d_ff]
    ffn_down: Vec<f32>,     // [d_ff, d_model]

    // Pre-FFN layer norm
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
}

impl GpuTransformer {
    /// Create a GPU transformer with random weights
    pub fn new(ctx: Arc<GpuContext>, config: TransformerConfig) -> Self {
        let ops = GpuOps::new(ctx);

        let d_model = config.d_model;
        let d_ff = config.d_ff;
        let vocab_size = config.vocab_size;
        let max_seq_len = config.max_seq_len;
        let n_layers = config.n_layers;

        // Initialize embeddings
        let token_embeddings = Self::random_weights(vocab_size * d_model, d_model);
        let position_embeddings = Self::random_weights(max_seq_len * d_model, d_model);

        // Initialize layer weights
        let layer_weights: Vec<_> = (0..n_layers)
            .map(|_| LayerWeights {
                q_weight: Self::random_weights(d_model * d_model, d_model),
                k_weight: Self::random_weights(d_model * d_model, d_model),
                v_weight: Self::random_weights(d_model * d_model, d_model),
                o_weight: Self::random_weights(d_model * d_model, d_model),
                ln1_gamma: vec![1.0; d_model],
                ln1_beta: vec![0.0; d_model],
                ffn_up: Self::random_weights(d_model * d_ff, d_model),
                ffn_down: Self::random_weights(d_ff * d_model, d_ff),
                ln2_gamma: vec![1.0; d_model],
                ln2_beta: vec![0.0; d_model],
            })
            .collect();

        let final_ln_gamma = vec![1.0; d_model];
        let final_ln_beta = vec![0.0; d_model];
        let output_projection = Self::random_weights(d_model * vocab_size, d_model);

        Self {
            ops,
            config,
            token_embeddings,
            position_embeddings,
            layer_weights,
            final_ln_gamma,
            final_ln_beta,
            output_projection,
        }
    }

    /// Generate random weights using Xavier initialization
    fn random_weights(size: usize, fan_in: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::rng();
        let scale = (2.0 / fan_in as f32).sqrt();
        (0..size).map(|_| rng.random_range(-scale..scale)).collect()
    }

    /// Forward pass: token_ids -> logits
    /// Returns logits for the last position only
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let seq_len = token_ids.len() as u32;
        let d_model = self.config.d_model as u32;
        let vocab_size = self.config.vocab_size as u32;

        // 1. Token embeddings + position embeddings
        let mut hidden = self.ops.embedding_lookup(
            token_ids,
            &self.token_embeddings,
            seq_len,
            d_model,
            vocab_size,
        );

        // Add position embeddings
        let positions: Vec<u32> = (0..seq_len).collect();
        let pos_emb = self.ops.embedding_lookup(
            &positions,
            &self.position_embeddings,
            seq_len,
            d_model,
            self.config.max_seq_len as u32,
        );
        hidden = self.ops.add(&hidden, &pos_emb);

        // 2. Process each layer
        for layer in &self.layer_weights {
            hidden = self.forward_layer(&hidden, layer, seq_len, d_model);
        }

        // 3. Final layer norm
        hidden = self.ops.layer_norm(
            &hidden,
            &self.final_ln_gamma,
            &self.final_ln_beta,
            seq_len,
            d_model,
            1e-5,
        );

        // 4. Get last position's hidden state
        let last_pos_start = ((seq_len - 1) * d_model) as usize;
        let last_hidden = hidden[last_pos_start..last_pos_start + d_model as usize].to_vec();

        // 5. Project to vocabulary
        self.ops.matmul(
            &last_hidden,
            &self.output_projection,
            1,
            d_model,
            vocab_size,
        )
    }

    /// Forward pass through a single transformer layer
    fn forward_layer(
        &self,
        hidden: &[f32],
        layer: &LayerWeights,
        seq_len: u32,
        d_model: u32,
    ) -> Vec<f32> {
        // Pre-attention layer norm
        let normed = self.ops.layer_norm(
            hidden,
            &layer.ln1_gamma,
            &layer.ln1_beta,
            seq_len,
            d_model,
            1e-5,
        );

        // Compute Q, K, V projections
        let q = self.ops.matmul(&normed, &layer.q_weight, seq_len, d_model, d_model);
        let k = self.ops.matmul(&normed, &layer.k_weight, seq_len, d_model, d_model);
        let v = self.ops.matmul(&normed, &layer.v_weight, seq_len, d_model, d_model);

        // Attention (simplified single-head for now)
        let attn_out = self.ops.attention(&q, &k, &v, seq_len, d_model);

        // Output projection
        let projected = self.ops.matmul(&attn_out, &layer.o_weight, seq_len, d_model, d_model);

        // Residual connection
        let after_attn = self.ops.add(hidden, &projected);

        // Pre-FFN layer norm
        let normed2 = self.ops.layer_norm(
            &after_attn,
            &layer.ln2_gamma,
            &layer.ln2_beta,
            seq_len,
            d_model,
            1e-5,
        );

        // FFN: up projection -> GELU -> down projection
        let d_ff = self.config.d_ff as u32;
        let up = self.ops.matmul(&normed2, &layer.ffn_up, seq_len, d_model, d_ff);
        let activated = self.ops.gelu(&up);
        let down = self.ops.matmul(&activated, &layer.ffn_down, seq_len, d_ff, d_model);

        // Final residual connection
        self.ops.add(&after_attn, &down)
    }

    /// Get config
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(Arc<GpuContext>, TransformerConfig)> {
        let ctx = GpuContext::new().ok()?;
        let config = TransformerConfig {
            vocab_size: 100,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            max_seq_len: 32,
            dropout: 0.0,
            ..Default::default()
        };
        Some((Arc::new(ctx), config))
    }

    #[test]
    fn test_gpu_transformer_creation() {
        let Some((ctx, config)) = setup() else { return };
        let transformer = GpuTransformer::new(ctx, config);
        assert_eq!(transformer.config().vocab_size, 100);
        assert_eq!(transformer.layer_weights.len(), 2);
    }

    #[test]
    fn test_gpu_forward_pass() {
        let Some((ctx, config)) = setup() else { return };
        let transformer = GpuTransformer::new(ctx, config);

        let token_ids = vec![1u32, 5, 10, 15];
        let logits = transformer.forward(&token_ids);

        // Should return vocab_size logits
        assert_eq!(logits.len(), 100);

        // Logits should be finite
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_gpu_forward_single_token() {
        let Some((ctx, config)) = setup() else { return };
        let transformer = GpuTransformer::new(ctx, config);

        let token_ids = vec![42u32];
        let logits = transformer.forward(&token_ids);

        assert_eq!(logits.len(), 100);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_gpu_forward_different_inputs_produce_different_outputs() {
        let Some((ctx, config)) = setup() else { return };
        let transformer = GpuTransformer::new(ctx, config);

        let logits1 = transformer.forward(&[1, 2, 3]);
        let logits2 = transformer.forward(&[4, 5, 6]);

        // Different inputs should produce different outputs
        let diff: f32 = logits1.iter()
            .zip(logits2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1, "Different inputs should produce different outputs");
    }
}
