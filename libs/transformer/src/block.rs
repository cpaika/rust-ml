//! Transformer block combining attention, layer norm, and feed-forward

use crate::attention::MultiHeadAttention;
use crate::feed_forward::FeedForward;
use crate::layer_norm::LayerNorm;
use crate::TransformerConfig;
use serde::{Deserialize, Serialize};

/// A single transformer decoder block
///
/// Uses pre-layer normalization (GPT-2 style):
/// x = x + attention(layer_norm1(x))
/// x = x + feed_forward(layer_norm2(x))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerBlock {
    /// First layer normalization (before attention)
    pub ln1: LayerNorm,
    /// Multi-head self-attention
    pub attention: MultiHeadAttention,
    /// Second layer normalization (before feed-forward)
    pub ln2: LayerNorm,
    /// Feed-forward network
    pub ff: FeedForward,
    /// Whether to use pre-norm (before sublayer) or post-norm (after sublayer)
    pub pre_norm: bool,
    /// Model dimension
    pub d_model: usize,
}

impl TransformerBlock {
    /// Create a new transformer block
    pub fn new(config: &TransformerConfig) -> Self {
        Self {
            ln1: LayerNorm::from_config(config),
            attention: MultiHeadAttention::from_config(config),
            ln2: LayerNorm::from_config(config),
            ff: FeedForward::from_config(config),
            pre_norm: config.pre_norm,
            d_model: config.d_model,
        }
    }

    /// Forward pass
    ///
    /// Input: [seq_len, d_model] as flat vector
    /// Output: [seq_len, d_model] as flat vector
    pub fn forward(&mut self, x: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(x.len(), seq_len * self.d_model);

        if self.pre_norm {
            self.forward_pre_norm(x, seq_len)
        } else {
            self.forward_post_norm(x, seq_len)
        }
    }

    /// Pre-layer normalization forward pass (GPT-2 style)
    fn forward_pre_norm(&mut self, x: &[f32], seq_len: usize) -> Vec<f32> {
        // Attention sublayer with residual
        let normalized1 = self.ln1.forward(x, seq_len);
        let attn_out = self.attention.forward(&normalized1, seq_len);
        let mut residual1: Vec<f32> = x.iter()
            .zip(attn_out.iter())
            .map(|(&xi, &ai)| xi + ai)
            .collect();

        // Feed-forward sublayer with residual
        let normalized2 = self.ln2.forward(&residual1, seq_len);
        let ff_out = self.ff.forward(&normalized2, seq_len);
        for (r, &f) in residual1.iter_mut().zip(ff_out.iter()) {
            *r += f;
        }

        residual1
    }

    /// Post-layer normalization forward pass (original transformer)
    fn forward_post_norm(&mut self, x: &[f32], seq_len: usize) -> Vec<f32> {
        // Attention sublayer
        let attn_out = self.attention.forward(x, seq_len);
        let residual1: Vec<f32> = x.iter()
            .zip(attn_out.iter())
            .map(|(&xi, &ai)| xi + ai)
            .collect();
        let normalized1 = self.ln1.forward(&residual1, seq_len);

        // Feed-forward sublayer
        let ff_out = self.ff.forward(&normalized1, seq_len);
        let residual2: Vec<f32> = normalized1.iter()
            .zip(ff_out.iter())
            .map(|(&ni, &fi)| ni + fi)
            .collect();
        let normalized2 = self.ln2.forward(&residual2, seq_len);

        normalized2
    }

    /// Get cached attention weights for visualization
    pub fn get_attention_weights(&self) -> &[f32] {
        self.attention.get_attention_weights()
    }

    /// Get number of parameters in this block
    pub fn num_parameters(&self) -> usize {
        self.ln1.num_parameters()
            + self.attention.num_parameters()
            + self.ln2.num_parameters()
            + self.ff.num_parameters()
    }

    /// Backward pass through transformer block
    ///
    /// Computes gradients for all component layers and returns gradient w.r.t. input.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient w.r.t. block output [seq_len, d_model]
    /// * `seq_len` - Sequence length
    /// * `learning_rate` - Learning rate for parameter updates
    ///
    /// # Returns
    /// Gradient w.r.t. input [seq_len, d_model]
    pub fn backward(
        &mut self,
        grad_output: &[f32],
        seq_len: usize,
        learning_rate: f32,
    ) -> Vec<f32> {
        assert_eq!(grad_output.len(), seq_len * self.d_model);

        if self.pre_norm {
            self.backward_pre_norm(grad_output, seq_len, learning_rate)
        } else {
            self.backward_post_norm(grad_output, seq_len, learning_rate)
        }
    }

    /// Backward for pre-layer normalization (GPT-2 style)
    ///
    /// Forward was:
    /// 1. normalized1 = ln1.forward(x)
    /// 2. attn_out = attention.forward(normalized1)
    /// 3. residual1 = x + attn_out
    /// 4. normalized2 = ln2.forward(residual1)
    /// 5. ff_out = ff.forward(normalized2)
    /// 6. output = residual1 + ff_out
    ///
    /// Backward reverses this order with residual gradient splitting.
    fn backward_pre_norm(
        &mut self,
        grad_output: &[f32],
        seq_len: usize,
        learning_rate: f32,
    ) -> Vec<f32> {
        // Step 6 backward: output = residual1 + ff_out
        // Both paths receive grad_output
        let grad_ff_out = grad_output.to_vec();
        let mut grad_residual1 = grad_output.to_vec();

        // Step 5 backward: ff_out = ff.forward(normalized2)
        let grad_normalized2 = self.ff.backward(&grad_ff_out, seq_len, learning_rate);

        // Step 4 backward: normalized2 = ln2.forward(residual1)
        let grad_residual1_from_ln2 = self.ln2.backward(&grad_normalized2, seq_len, learning_rate);

        // Accumulate gradients at residual1
        for (g, &g2) in grad_residual1.iter_mut().zip(grad_residual1_from_ln2.iter()) {
            *g += g2;
        }

        // Step 3 backward: residual1 = x + attn_out
        // Both paths receive grad_residual1
        let grad_attn_out = grad_residual1.clone();
        let mut grad_x = grad_residual1;

        // Step 2 backward: attn_out = attention.forward(normalized1)
        let grad_normalized1 = self.attention.backward(&grad_attn_out, seq_len, learning_rate);

        // Step 1 backward: normalized1 = ln1.forward(x)
        let grad_x_from_ln1 = self.ln1.backward(&grad_normalized1, seq_len, learning_rate);

        // Accumulate gradients at input
        for (g, &g1) in grad_x.iter_mut().zip(grad_x_from_ln1.iter()) {
            *g += g1;
        }

        grad_x
    }

    /// Backward for post-layer normalization (original transformer)
    ///
    /// Forward was:
    /// 1. attn_out = attention.forward(x)
    /// 2. residual1 = x + attn_out
    /// 3. normalized1 = ln1.forward(residual1)
    /// 4. ff_out = ff.forward(normalized1)
    /// 5. residual2 = normalized1 + ff_out
    /// 6. output = ln2.forward(residual2)
    fn backward_post_norm(
        &mut self,
        grad_output: &[f32],
        seq_len: usize,
        learning_rate: f32,
    ) -> Vec<f32> {
        // Step 6 backward: output = ln2.forward(residual2)
        let grad_residual2 = self.ln2.backward(grad_output, seq_len, learning_rate);

        // Step 5 backward: residual2 = normalized1 + ff_out
        let grad_ff_out = grad_residual2.clone();
        let mut grad_normalized1 = grad_residual2;

        // Step 4 backward: ff_out = ff.forward(normalized1)
        let grad_normalized1_from_ff = self.ff.backward(&grad_ff_out, seq_len, learning_rate);

        // Accumulate gradients at normalized1
        for (g, &gf) in grad_normalized1.iter_mut().zip(grad_normalized1_from_ff.iter()) {
            *g += gf;
        }

        // Step 3 backward: normalized1 = ln1.forward(residual1)
        let grad_residual1 = self.ln1.backward(&grad_normalized1, seq_len, learning_rate);

        // Step 2 backward: residual1 = x + attn_out
        let grad_attn_out = grad_residual1.clone();
        let mut grad_x = grad_residual1;

        // Step 1 backward: attn_out = attention.forward(x)
        let grad_x_from_attn = self.attention.backward(&grad_attn_out, seq_len, learning_rate);

        // Accumulate gradients at input
        for (g, &ga) in grad_x.iter_mut().zip(grad_x_from_attn.iter()) {
            *g += ga;
        }

        grad_x
    }
}

/// Stack of transformer blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerStack {
    /// Individual transformer blocks
    pub blocks: Vec<TransformerBlock>,
    /// Model dimension
    pub d_model: usize,
}

impl TransformerStack {
    /// Create a stack of transformer blocks
    pub fn new(config: &TransformerConfig) -> Self {
        let blocks: Vec<TransformerBlock> = (0..config.n_layers)
            .map(|_| TransformerBlock::new(config))
            .collect();

        Self {
            blocks,
            d_model: config.d_model,
        }
    }

    /// Forward pass through all blocks
    pub fn forward(&mut self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let mut current = x.to_vec();

        for block in &mut self.blocks {
            current = block.forward(&current, seq_len);
        }

        current
    }

    /// Get number of layers
    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Get attention weights from a specific layer
    pub fn get_layer_attention(&self, layer: usize) -> Option<&[f32]> {
        self.blocks.get(layer).map(|b| b.get_attention_weights())
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.blocks.iter().map(|b| b.num_parameters()).sum()
    }

    /// Get mutable reference to a specific block
    pub fn get_block_mut(&mut self, idx: usize) -> Option<&mut TransformerBlock> {
        self.blocks.get_mut(idx)
    }

    /// Get reference to a specific block
    pub fn get_block(&self, idx: usize) -> Option<&TransformerBlock> {
        self.blocks.get(idx)
    }

    /// Backward pass through all transformer blocks
    ///
    /// Propagates gradients in reverse order through all blocks.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient w.r.t. stack output [seq_len, d_model]
    /// * `seq_len` - Sequence length
    /// * `learning_rate` - Learning rate for parameter updates
    ///
    /// # Returns
    /// Gradient w.r.t. input [seq_len, d_model]
    pub fn backward(
        &mut self,
        grad_output: &[f32],
        seq_len: usize,
        learning_rate: f32,
    ) -> Vec<f32> {
        let mut grad = grad_output.to_vec();

        // Backward through blocks in reverse order
        for block in self.blocks.iter_mut().rev() {
            grad = block.backward(&grad, seq_len, learning_rate);
        }

        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig::tiny(1000)
    }

    #[test]
    fn test_block_creation() {
        let config = tiny_config();
        let block = TransformerBlock::new(&config);

        assert_eq!(block.d_model, config.d_model);
        assert_eq!(block.pre_norm, config.pre_norm);
    }

    #[test]
    fn test_block_forward_shape() {
        let config = tiny_config();
        let mut block = TransformerBlock::new(&config);

        let seq_len = 8;
        let x = vec![0.1f32; seq_len * config.d_model];

        let output = block.forward(&x, seq_len);
        assert_eq!(output.len(), seq_len * config.d_model);
    }

    #[test]
    fn test_block_forward_pre_norm() {
        let mut config = tiny_config();
        config.pre_norm = true;
        let mut block = TransformerBlock::new(&config);

        let x = vec![0.1f32; 4 * config.d_model];
        let output = block.forward(&x, 4);

        assert_eq!(output.len(), x.len());
    }

    #[test]
    fn test_block_forward_post_norm() {
        let mut config = tiny_config();
        config.pre_norm = false;
        let mut block = TransformerBlock::new(&config);

        let x = vec![0.1f32; 4 * config.d_model];
        let output = block.forward(&x, 4);

        assert_eq!(output.len(), x.len());
    }

    #[test]
    fn test_block_residual_connection() {
        let config = tiny_config();
        let mut block = TransformerBlock::new(&config);

        // With non-zero input, output should be different from sublayer output alone
        // The residual connection adds the input to the sublayer output
        let x: Vec<f32> = (0..4 * config.d_model).map(|i| (i as f32) * 0.01).collect();
        let output = block.forward(&x, 4);

        // Output should exist and have same shape
        assert_eq!(output.len(), x.len());

        // Output should not be identical to input (sublayer should modify it)
        let all_same = x.iter().zip(output.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(!all_same, "Sublayer should modify the input");
    }

    #[test]
    fn test_block_attention_cached() {
        let config = tiny_config();
        let mut block = TransformerBlock::new(&config);

        let x = vec![0.1f32; 4 * config.d_model];
        block.forward(&x, 4);

        let attn_weights = block.get_attention_weights();
        assert!(!attn_weights.is_empty());
    }

    #[test]
    fn test_block_num_parameters() {
        let config = tiny_config();
        let block = TransformerBlock::new(&config);

        let params = block.num_parameters();
        assert!(params > 0);

        // Should include all components
        let expected = block.ln1.num_parameters()
            + block.attention.num_parameters()
            + block.ln2.num_parameters()
            + block.ff.num_parameters();
        assert_eq!(params, expected);
    }

    #[test]
    fn test_stack_creation() {
        let config = tiny_config();
        let stack = TransformerStack::new(&config);

        assert_eq!(stack.n_layers(), config.n_layers);
        assert_eq!(stack.d_model, config.d_model);
    }

    #[test]
    fn test_stack_forward() {
        let config = tiny_config();
        let mut stack = TransformerStack::new(&config);

        let x = vec![0.1f32; 4 * config.d_model];
        let output = stack.forward(&x, 4);

        assert_eq!(output.len(), x.len());
    }

    #[test]
    fn test_stack_get_layer_attention() {
        let config = tiny_config();
        let mut stack = TransformerStack::new(&config);

        let x = vec![0.1f32; 4 * config.d_model];
        stack.forward(&x, 4);

        let layer0_attn = stack.get_layer_attention(0);
        assert!(layer0_attn.is_some());

        let invalid_layer = stack.get_layer_attention(100);
        assert!(invalid_layer.is_none());
    }

    #[test]
    fn test_stack_num_parameters() {
        let config = tiny_config();
        let stack = TransformerStack::new(&config);

        let params = stack.num_parameters();
        assert!(params > 0);

        // Should be n_layers times single block params
        let single_block_params = stack.blocks[0].num_parameters();
        assert_eq!(params, config.n_layers * single_block_params);
    }

    #[test]
    fn test_stack_get_block() {
        let config = tiny_config();
        let mut stack = TransformerStack::new(&config);

        assert!(stack.get_block(0).is_some());
        assert!(stack.get_block_mut(0).is_some());
        assert!(stack.get_block(100).is_none());
    }

    #[test]
    fn test_block_deterministic() {
        let config = tiny_config();
        let mut block1 = TransformerBlock::new(&config);
        let mut block2 = block1.clone();

        let x = vec![0.5f32; 4 * config.d_model];

        let out1 = block1.forward(&x, 4);
        let out2 = block2.forward(&x, 4);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_stack_depth_matters() {
        let mut config = tiny_config();

        // Create stacks with different depths
        config.n_layers = 1;
        let shallow = TransformerStack::new(&config);

        config.n_layers = 4;
        let deep = TransformerStack::new(&config);

        // Deeper stack should have more parameters
        assert!(deep.num_parameters() > shallow.num_parameters());
    }

    #[test]
    fn test_block_serialization() {
        let config = tiny_config();
        let block = TransformerBlock::new(&config);

        let json = serde_json::to_string(&block).unwrap();
        let loaded: TransformerBlock = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.d_model, block.d_model);
        assert_eq!(loaded.pre_norm, block.pre_norm);
    }

    // ==================== BACKWARD PASS TESTS ====================

    #[test]
    fn test_block_backward_output_shape() {
        let config = tiny_config();
        let mut block = TransformerBlock::new(&config);
        let seq_len = 4;
        let x: Vec<f32> = (0..seq_len * config.d_model).map(|i| i as f32 * 0.01).collect();

        block.forward(&x, seq_len);

        let grad_output = vec![0.01f32; seq_len * config.d_model];
        let grad_input = block.backward(&grad_output, seq_len, 0.0);

        assert_eq!(grad_input.len(), seq_len * config.d_model);
    }

    #[test]
    fn test_block_backward_weights_update() {
        let config = tiny_config();
        let mut block = TransformerBlock::new(&config);
        let seq_len = 3;
        // Use larger input values to ensure non-trivial gradients
        let x: Vec<f32> = (0..seq_len * config.d_model).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Get initial weights
        let w_q_before = block.attention.w_q.clone();
        let w1_before = block.ff.w1.clone();
        let gamma_before = block.ln1.gamma.clone();

        block.forward(&x, seq_len);

        // Use larger gradients and higher learning rate
        let grad_output: Vec<f32> = (0..seq_len * config.d_model).map(|i| (i as f32 + 1.0) * 0.1).collect();
        block.backward(&grad_output, seq_len, 1.0);

        // All components should have updated weights
        let w_q_changed = block.attention.w_q.iter().zip(w_q_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        let w1_changed = block.ff.w1.iter().zip(w1_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        let gamma_changed = block.ln1.gamma.iter().zip(gamma_before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(w_q_changed, "Attention W_q should be updated");
        assert!(w1_changed, "FeedForward W1 should be updated");
        assert!(gamma_changed, "LayerNorm gamma should be updated");
    }

    #[test]
    fn test_block_backward_pre_norm() {
        let mut config = tiny_config();
        config.pre_norm = true;
        let mut block = TransformerBlock::new(&config);
        let seq_len = 3;
        let x: Vec<f32> = (0..seq_len * config.d_model).map(|i| i as f32 * 0.01).collect();

        block.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * config.d_model).map(|i| (i + 1) as f32 * 0.01).collect();
        let grad_input = block.backward(&grad_output, seq_len, 0.0);

        // Gradient should be non-zero
        let has_nonzero = grad_input.iter().any(|&g| g.abs() > 1e-10);
        assert!(has_nonzero, "Gradient input should have non-zero values");
    }

    #[test]
    fn test_block_backward_post_norm() {
        let mut config = tiny_config();
        config.pre_norm = false;
        let mut block = TransformerBlock::new(&config);
        let seq_len = 3;
        let x: Vec<f32> = (0..seq_len * config.d_model).map(|i| i as f32 * 0.01).collect();

        block.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * config.d_model).map(|i| (i + 1) as f32 * 0.01).collect();
        let grad_input = block.backward(&grad_output, seq_len, 0.0);

        // Gradient should be non-zero
        let has_nonzero = grad_input.iter().any(|&g| g.abs() > 1e-10);
        assert!(has_nonzero, "Gradient input should have non-zero values (post-norm)");
    }

    #[test]
    fn test_stack_backward_output_shape() {
        let mut config = tiny_config();
        config.n_layers = 2;
        let mut stack = TransformerStack::new(&config);
        let seq_len = 4;
        let x: Vec<f32> = (0..seq_len * config.d_model).map(|i| i as f32 * 0.01).collect();

        stack.forward(&x, seq_len);

        let grad_output = vec![0.01f32; seq_len * config.d_model];
        let grad_input = stack.backward(&grad_output, seq_len, 0.0);

        assert_eq!(grad_input.len(), seq_len * config.d_model);
    }

    #[test]
    fn test_stack_backward_all_layers_updated() {
        let mut config = tiny_config();
        config.n_layers = 3;
        let mut stack = TransformerStack::new(&config);
        let seq_len = 3;
        // Use larger input values to ensure non-trivial gradients through all layers
        let x: Vec<f32> = (0..seq_len * config.d_model).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Get initial weights from each layer
        let initial_weights: Vec<Vec<f32>> = stack.blocks.iter()
            .map(|b| b.attention.w_q.clone())
            .collect();

        stack.forward(&x, seq_len);

        // Use larger gradients and higher learning rate
        let grad_output: Vec<f32> = (0..seq_len * config.d_model).map(|i| (i as f32 + 1.0) * 0.1).collect();
        stack.backward(&grad_output, seq_len, 1.0);

        // All layers should have updated weights
        for (i, block) in stack.blocks.iter().enumerate() {
            let changed = block.attention.w_q.iter()
                .zip(initial_weights[i].iter())
                .any(|(a, b)| (a - b).abs() > 1e-10);
            assert!(changed, "Layer {} should have updated weights", i);
        }
    }

    #[test]
    fn test_stack_backward_gradient_nonzero() {
        let mut config = tiny_config();
        config.n_layers = 2;
        let mut stack = TransformerStack::new(&config);
        let seq_len = 3;
        let x: Vec<f32> = (0..seq_len * config.d_model).map(|i| i as f32 * 0.01).collect();

        stack.forward(&x, seq_len);

        let grad_output: Vec<f32> = (0..seq_len * config.d_model).map(|i| (i + 1) as f32 * 0.01).collect();
        let grad_input = stack.backward(&grad_output, seq_len, 0.0);

        // Gradient should be non-zero
        let has_nonzero = grad_input.iter().any(|&g| g.abs() > 1e-10);
        assert!(has_nonzero, "Stack gradient input should have non-zero values");
    }
}
