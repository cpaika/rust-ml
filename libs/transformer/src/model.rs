//! Complete transformer language model

use crate::block::TransformerStack;
use crate::embedding::EmbeddingLayer;
use crate::layer_norm::LayerNorm;
use crate::{cross_entropy_loss, softmax, TransformerConfig};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Complete decoder-only transformer for language modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformer {
    /// Model configuration
    pub config: TransformerConfig,
    /// Token and position embeddings
    pub embedding: EmbeddingLayer,
    /// Stack of transformer blocks
    pub blocks: TransformerStack,
    /// Final layer normalization
    pub final_norm: LayerNorm,
    /// Output projection weights [d_model, vocab_size] (if not tied)
    pub lm_head: Option<Vec<f32>>,

    // Cached values for forward pass
    #[serde(skip)]
    cached_embedded: Vec<f32>, // After embedding + position, before blocks
    #[serde(skip)]
    cached_pre_final_norm: Vec<f32>, // After blocks, before final LayerNorm
    #[serde(skip)]
    cached_hidden: Vec<f32>, // After final LayerNorm (final hidden states)
    #[serde(skip)]
    cached_logits: Vec<f32>,
    #[serde(skip)]
    cached_token_ids: Vec<u32>,

    // Gradients for backward pass
    #[serde(skip)]
    embedding_grads: Vec<f32>,
    #[serde(skip)]
    lm_head_grads: Option<Vec<f32>>,
}

impl Transformer {
    /// Create a new transformer language model
    pub fn new(config: TransformerConfig) -> Result<Self, String> {
        config.validate()?;

        let embedding = EmbeddingLayer::from_config(&config);
        let blocks = TransformerStack::new(&config);
        let final_norm = LayerNorm::from_config(&config);

        // If not tying weights, create separate output projection
        let lm_head = if !config.tie_weights {
            let size = config.d_model * config.vocab_size;
            let weights: Vec<f32> = (0..size)
                .map(|_| crate::normal_init(0.02))
                .collect();
            Some(weights)
        } else {
            None
        };

        // Initialize gradient storage
        let embedding_grads = vec![0.0f32; config.vocab_size * config.d_model];
        let lm_head_grads = if !config.tie_weights {
            Some(vec![0.0f32; config.d_model * config.vocab_size])
        } else {
            None
        };

        Ok(Self {
            config,
            embedding,
            blocks,
            final_norm,
            lm_head,
            cached_embedded: Vec::new(),
            cached_pre_final_norm: Vec::new(),
            cached_hidden: Vec::new(),
            cached_logits: Vec::new(),
            cached_token_ids: Vec::new(),
            embedding_grads,
            lm_head_grads,
        })
    }

    /// Forward pass
    ///
    /// Input: token IDs [seq_len]
    /// Output: logits [seq_len, vocab_size] as flat vector
    pub fn forward(&mut self, token_ids: &[u32]) -> Vec<f32> {
        let seq_len = token_ids.len();

        // Cache token IDs for backward pass
        self.cached_token_ids = token_ids.to_vec();

        // Embedding lookup
        let embedded = self.embedding.forward(token_ids);
        // Cache embedded states (for gradient computation back to embeddings)
        self.cached_embedded = embedded.clone();

        // Pass through transformer blocks
        let block_output = self.blocks.forward(&embedded, seq_len);
        // Cache pre-final-norm states (for backprop through final_norm)
        self.cached_pre_final_norm = block_output.clone();

        // Final layer norm
        let hidden = self.final_norm.forward(&block_output, seq_len);
        // Cache hidden states (after final norm, used for lm_head gradient)
        self.cached_hidden = hidden.clone();

        // Output projection (logits)
        let logits = self.compute_logits(&hidden, seq_len);
        self.cached_logits = logits.clone();

        logits
    }

    /// Compute logits from hidden states
    fn compute_logits(&self, hidden: &[f32], seq_len: usize) -> Vec<f32> {
        let vocab_size = self.config.vocab_size;
        let d_model = self.config.d_model;
        let mut logits = vec![0.0f32; seq_len * vocab_size];

        // Use tied weights or separate lm_head
        let weights = if self.config.tie_weights {
            &self.embedding.token_embedding.weights
        } else {
            self.lm_head.as_ref().unwrap()
        };

        // logits[pos, vocab] = hidden[pos] dot embedding[vocab]
        for pos in 0..seq_len {
            for vocab in 0..vocab_size {
                let mut sum = 0.0;
                for d in 0..d_model {
                    sum += hidden[pos * d_model + d] * weights[vocab * d_model + d];
                }
                logits[pos * vocab_size + vocab] = sum;
            }
        }

        logits
    }

    /// Compute loss for next-token prediction
    ///
    /// Input: token IDs [seq_len]
    /// Target: next token IDs [seq_len] (shifted by 1)
    /// Returns: average cross-entropy loss
    pub fn compute_loss(&mut self, token_ids: &[u32], targets: &[u32]) -> f32 {
        assert_eq!(token_ids.len(), targets.len());

        let logits = self.forward(token_ids);
        let seq_len = token_ids.len();
        let vocab_size = self.config.vocab_size;

        let mut total_loss = 0.0;
        for pos in 0..seq_len {
            let start = pos * vocab_size;
            let end = start + vocab_size;
            let pos_logits = &logits[start..end];
            total_loss += cross_entropy_loss(pos_logits, targets[pos] as usize);
        }

        total_loss / seq_len as f32
    }

    /// Get next token probabilities for the last position
    pub fn get_next_token_probs(&self) -> Vec<f32> {
        if self.cached_logits.is_empty() {
            return vec![];
        }

        let vocab_size = self.config.vocab_size;
        let last_pos_start = self.cached_logits.len() - vocab_size;
        let mut probs = self.cached_logits[last_pos_start..].to_vec();
        softmax(&mut probs);
        probs
    }

    /// Sample next token from logits with temperature
    pub fn sample_next_token(&self, temperature: f32) -> Option<u32> {
        if self.cached_logits.is_empty() {
            return None;
        }

        let vocab_size = self.config.vocab_size;
        let last_pos_start = self.cached_logits.len() - vocab_size;
        let mut logits = self.cached_logits[last_pos_start..].to_vec();

        // Apply temperature (with safety for very low temperatures)
        let temp = temperature.max(1e-6);
        for l in logits.iter_mut() {
            *l /= temp;
        }

        // Softmax
        softmax(&mut logits);

        // Ensure probabilities sum to 1.0 (handle numerical precision issues)
        let sum: f32 = logits.iter().sum();
        if sum > 0.0 && (sum - 1.0).abs() > 1e-6 {
            for p in logits.iter_mut() {
                *p /= sum;
            }
        }

        // Handle degenerate case where all probs are 0 or nearly 0
        if sum < 1e-10 {
            // Fall back to greedy decoding
            return self.greedy_next_token();
        }

        // Sample from distribution
        let mut rng = rand::rng();
        let r: f32 = rng.random();
        let mut cumsum = 0.0;

        for (idx, &prob) in logits.iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                return Some(idx as u32);
            }
        }

        // Fallback: return token with highest probability (numerical safety)
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
    }

    /// Greedy decoding: get the most likely next token
    pub fn greedy_next_token(&self) -> Option<u32> {
        if self.cached_logits.is_empty() {
            return None;
        }

        let vocab_size = self.config.vocab_size;
        let last_pos_start = self.cached_logits.len() - vocab_size;
        let logits = &self.cached_logits[last_pos_start..];

        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
    }

    /// Get top-k next tokens with their probabilities
    pub fn top_k_next_tokens(&self, k: usize) -> Vec<(u32, f32)> {
        if self.cached_logits.is_empty() {
            return vec![];
        }

        let vocab_size = self.config.vocab_size;
        let last_pos_start = self.cached_logits.len() - vocab_size;
        let mut probs = self.cached_logits[last_pos_start..].to_vec();
        softmax(&mut probs);

        let mut indexed: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);
        indexed
    }

    /// Generate tokens autoregressively
    pub fn generate(&mut self, prompt: &[u32], max_tokens: usize, temperature: f32) -> Vec<u32> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_tokens {
            // Check sequence length limit
            if tokens.len() >= self.config.max_seq_len {
                break;
            }

            // Forward pass
            self.forward(&tokens);

            // Sample next token
            let next_token = if temperature > 0.0 {
                self.sample_next_token(temperature).unwrap_or(0)
            } else {
                self.greedy_next_token().unwrap_or(0)
            };

            // Check for EOS (assuming token 3 is EOS based on tokenizer)
            if next_token == 3 {
                break;
            }

            tokens.push(next_token);
        }

        tokens
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let embedding_params = self.embedding.num_parameters();
        let block_params = self.blocks.num_parameters();
        let norm_params = self.final_norm.num_parameters();
        let lm_head_params = if let Some(ref head) = self.lm_head {
            head.len()
        } else {
            0
        };

        embedding_params + block_params + norm_params + lm_head_params
    }

    /// Get attention weights from a specific layer for visualization
    pub fn get_layer_attention(&self, layer: usize) -> Option<&[f32]> {
        self.blocks.get_layer_attention(layer)
    }

    /// Get the hidden states from the last forward pass
    pub fn get_hidden_states(&self) -> &[f32] {
        &self.cached_hidden
    }

    /// Get the logits from the last forward pass
    pub fn get_logits(&self) -> &[f32] {
        &self.cached_logits
    }

    /// Full backward pass with gradient propagation through all layers
    ///
    /// This computes and applies gradients through the entire model:
    /// 1. Output projection (lm_head or tied embeddings)
    /// 2. Final layer normalization
    /// 3. All transformer blocks (attention + feed-forward)
    /// 4. Input embeddings (token + position)
    ///
    /// The gradient for cross-entropy loss with softmax is: softmax(logits) - one_hot(target)
    ///
    /// # Arguments
    /// * `targets` - Target token IDs for each position
    /// * `learning_rate` - Learning rate for weight updates
    pub fn backward(&mut self, targets: &[u32], learning_rate: f32) {
        if self.cached_hidden.is_empty() || self.cached_logits.is_empty() {
            return;
        }

        let seq_len = targets.len();
        let vocab_size = self.config.vocab_size;
        let d_model = self.config.d_model;

        // Clear previous gradients
        self.embedding_grads.iter_mut().for_each(|g| *g = 0.0);
        if let Some(ref mut grads) = self.lm_head_grads {
            grads.iter_mut().for_each(|g| *g = 0.0);
        }

        // Step 1: Compute gradient of loss w.r.t. logits and hidden states
        // grad_logits = softmax(logits) - one_hot(target)
        // grad_hidden = grad_logits @ W^T (where W is the output projection)
        let mut grad_hidden = vec![0.0f32; seq_len * d_model];

        // Get output projection weights
        let weights = if self.config.tie_weights {
            &self.embedding.token_embedding.weights
        } else {
            self.lm_head.as_ref().unwrap()
        };

        for pos in 0..seq_len {
            let logit_start = pos * vocab_size;
            let logit_end = logit_start + vocab_size;
            let pos_logits = &self.cached_logits[logit_start..logit_end];

            // Compute softmax
            let mut grad_logits = pos_logits.to_vec();
            softmax(&mut grad_logits);

            // Compute grad_logits = probs - one_hot(target)
            let target_idx = targets[pos] as usize;
            grad_logits[target_idx] -= 1.0;

            // Normalize by sequence length
            for g in grad_logits.iter_mut() {
                *g /= seq_len as f32;
            }

            let hidden_start = pos * d_model;

            // Accumulate gradients for output projection weights
            // grad_W[v, d] += hidden[d] * grad_logits[v]
            let grad_storage = if self.config.tie_weights {
                &mut self.embedding_grads
            } else {
                self.lm_head_grads.as_mut().unwrap()
            };

            for v in 0..vocab_size {
                let g_logit = grad_logits[v];
                for d in 0..d_model {
                    let hidden_val = self.cached_hidden[hidden_start + d];
                    grad_storage[v * d_model + d] += hidden_val * g_logit;
                }
            }

            // Compute gradient w.r.t. hidden states
            // grad_hidden[d] = sum_v(W[v, d] * grad_logits[v])
            for d in 0..d_model {
                let mut sum = 0.0;
                for v in 0..vocab_size {
                    sum += weights[v * d_model + d] * grad_logits[v];
                }
                grad_hidden[hidden_start + d] = sum;
            }
        }

        // Step 2: Apply output projection weight updates
        self.update_output_projection(learning_rate);

        // Step 3: Backward through final layer norm
        let grad_pre_norm = self.final_norm.backward(&grad_hidden, seq_len, learning_rate);

        // Step 4: Backward through transformer blocks
        let grad_embedded = self.blocks.backward(&grad_pre_norm, seq_len, learning_rate);

        // Step 5: Update embeddings with gradient from transformer blocks
        self.update_embeddings(&grad_embedded, seq_len, learning_rate);
    }

    /// Update output projection weights (lm_head or tied embeddings)
    fn update_output_projection(&mut self, learning_rate: f32) {
        let vocab_size = self.config.vocab_size;
        let d_model = self.config.d_model;

        if self.config.tie_weights {
            // Update token embeddings (which are tied to output)
            let weights = self.embedding.token_embedding.weights_mut();
            for i in 0..vocab_size * d_model {
                weights[i] -= learning_rate * self.embedding_grads[i];
            }
        } else {
            // Update separate lm_head
            if let (Some(weights), Some(grads)) = (&mut self.lm_head, &self.lm_head_grads) {
                for i in 0..weights.len() {
                    weights[i] -= learning_rate * grads[i];
                }
            }
        }
    }

    /// Update input embeddings with gradients from backpropagation
    fn update_embeddings(&mut self, grad_embedded: &[f32], seq_len: usize, learning_rate: f32) {
        let d_model = self.config.d_model;
        // Scale factor used in forward pass - gradients need to be scaled by this
        // because forward pass multiplies embeddings by scale, so dL/dw = dL/dy * scale
        let scale = self.embedding.scale;

        // Update token embeddings for input tokens
        for pos in 0..seq_len {
            if pos >= self.cached_token_ids.len() {
                break;
            }
            let token_id = self.cached_token_ids[pos] as usize;
            let grad_start = pos * d_model;

            // Accumulate gradient to the token embedding (scaled)
            let weights = self.embedding.token_embedding.weights_mut();
            for d in 0..d_model {
                let grad = grad_embedded[grad_start + d] * scale;
                weights[token_id * d_model + d] -= learning_rate * grad;
            }
        }

        // Update position embeddings (also scaled by √d_model now)
        let pos_weights = self.embedding.position_embedding.weights_mut();
        for pos in 0..seq_len.min(self.config.max_seq_len) {
            let grad_start = pos * d_model;
            for d in 0..d_model {
                let grad = grad_embedded[grad_start + d] * scale;
                pos_weights[pos * d_model + d] -= learning_rate * grad;
            }
        }
    }

    /// Legacy backward pass (without learning rate) - computes gradients only
    ///
    /// Note: This is deprecated. Use backward(targets, learning_rate) for full training.
    pub fn backward_gradients_only(&mut self, targets: &[u32]) {
        // Call backward with 0 learning rate to just compute gradients
        // Note: This won't update transformer block weights
        self.backward(targets, 0.0);
    }

    /// Update weights using pre-computed gradients (legacy compatibility)
    ///
    /// Note: For full training, use backward(targets, learning_rate) instead.
    pub fn update_weights(&mut self, learning_rate: f32) {
        // This only updates output projection - blocks are updated in backward()
        self.update_output_projection(learning_rate);
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.embedding_grads.iter_mut().for_each(|g| *g = 0.0);
        if let Some(ref mut grads) = self.lm_head_grads {
            grads.iter_mut().for_each(|g| *g = 0.0);
        }
    }

    //=========================================================================
    // Gradient Access Methods (for external optimizers like Adam)
    //=========================================================================

    /// Get embedding gradients for external optimizer
    pub fn get_embedding_gradients(&self) -> Vec<f32> {
        self.embedding_grads.clone()
    }

    /// Get mutable reference to embeddings for external optimizer
    pub fn get_embeddings_mut(&mut self) -> &mut [f32] {
        &mut self.embedding.token_embedding.weights
    }

    /// Get reference to embeddings
    pub fn get_embeddings(&self) -> &[f32] {
        &self.embedding.token_embedding.weights
    }

    /// Clear embedding gradients
    pub fn clear_embedding_gradients(&mut self) {
        self.embedding_grads.fill(0.0);
    }

    /// Get lm_head gradients for external optimizer (if not using tied weights)
    pub fn get_lm_head_gradients(&self) -> Option<Vec<f32>> {
        self.lm_head_grads.clone()
    }

    /// Get mutable reference to lm_head weights (if not using tied weights)
    pub fn get_lm_head_mut(&mut self) -> Option<&mut [f32]> {
        self.lm_head.as_mut().map(|v| v.as_mut_slice())
    }

    /// Clear lm_head gradients
    pub fn clear_lm_head_gradients(&mut self) {
        if let Some(ref mut grads) = self.lm_head_grads {
            grads.fill(0.0);
        }
    }

    /// Get model configuration
    pub fn get_config(&self) -> &TransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig::tiny(100)
    }

    #[test]
    fn test_transformer_creation() {
        let config = tiny_config();
        let model = Transformer::new(config.clone());
        assert!(model.is_ok());

        let model = model.unwrap();
        assert_eq!(model.config.vocab_size, 100);
    }

    #[test]
    fn test_transformer_forward_shape() {
        let config = tiny_config();
        let mut model = Transformer::new(config.clone()).unwrap();

        let tokens = vec![1, 2, 3, 4];
        let logits = model.forward(&tokens);

        assert_eq!(logits.len(), tokens.len() * config.vocab_size);
    }

    #[test]
    fn test_transformer_compute_loss() {
        let config = tiny_config();
        let mut model = Transformer::new(config).unwrap();

        let tokens = vec![1, 2, 3, 4];
        let targets = vec![2, 3, 4, 5];

        let loss = model.compute_loss(&tokens, &targets);

        // Loss should be positive
        assert!(loss > 0.0);
        // Loss should be finite
        assert!(loss.is_finite());
    }

    #[test]
    fn test_transformer_get_next_token_probs() {
        let config = tiny_config();
        let mut model = Transformer::new(config.clone()).unwrap();

        let tokens = vec![1, 2, 3];
        model.forward(&tokens);

        let probs = model.get_next_token_probs();
        assert_eq!(probs.len(), config.vocab_size);

        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_transformer_sample_next_token() {
        let config = tiny_config();
        let mut model = Transformer::new(config.clone()).unwrap();

        let tokens = vec![1, 2, 3];
        model.forward(&tokens);

        let next = model.sample_next_token(1.0);
        assert!(next.is_some());

        let token = next.unwrap();
        assert!(token < config.vocab_size as u32);
    }

    #[test]
    fn test_transformer_greedy_next_token() {
        let config = tiny_config();
        let mut model = Transformer::new(config.clone()).unwrap();

        let tokens = vec![1, 2, 3];
        model.forward(&tokens);

        let next = model.greedy_next_token();
        assert!(next.is_some());

        let token = next.unwrap();
        assert!(token < config.vocab_size as u32);
    }

    #[test]
    fn test_transformer_top_k() {
        let config = tiny_config();
        let mut model = Transformer::new(config).unwrap();

        let tokens = vec![1, 2, 3];
        model.forward(&tokens);

        let top_k = model.top_k_next_tokens(5);
        assert_eq!(top_k.len(), 5);

        // Should be sorted by probability (descending)
        for i in 0..4 {
            assert!(top_k[i].1 >= top_k[i + 1].1);
        }
    }

    #[test]
    fn test_transformer_generate() {
        let config = tiny_config();
        let mut model = Transformer::new(config).unwrap();

        let prompt = vec![1, 2, 3];
        let generated = model.generate(&prompt, 5, 1.0);

        // Should have at least the prompt
        assert!(generated.len() >= prompt.len());
        // Should have generated some tokens
        assert!(generated.len() <= prompt.len() + 5);
        // First tokens should match prompt
        assert_eq!(&generated[..prompt.len()], &prompt);
    }

    #[test]
    fn test_transformer_num_parameters() {
        let config = tiny_config();
        let model = Transformer::new(config).unwrap();

        let params = model.num_parameters();
        assert!(params > 0);
        println!("Tiny model parameters: {}", params);
    }

    #[test]
    fn test_transformer_get_layer_attention() {
        let config = tiny_config();
        let mut model = Transformer::new(config).unwrap();

        let tokens = vec![1, 2, 3, 4];
        model.forward(&tokens);

        let attn = model.get_layer_attention(0);
        assert!(attn.is_some());
        assert!(!attn.unwrap().is_empty());
    }

    #[test]
    fn test_transformer_tied_weights() {
        let mut config = tiny_config();
        config.tie_weights = true;
        let model = Transformer::new(config).unwrap();

        assert!(model.lm_head.is_none());
    }

    #[test]
    fn test_transformer_untied_weights() {
        let mut config = tiny_config();
        config.tie_weights = false;
        let model = Transformer::new(config.clone()).unwrap();

        assert!(model.lm_head.is_some());
        let lm_head = model.lm_head.as_ref().unwrap();
        assert_eq!(lm_head.len(), config.d_model * config.vocab_size);
    }

    #[test]
    fn test_transformer_get_hidden_states() {
        let config = tiny_config();
        let mut model = Transformer::new(config.clone()).unwrap();

        let tokens = vec![1, 2, 3];
        model.forward(&tokens);

        let hidden = model.get_hidden_states();
        assert_eq!(hidden.len(), tokens.len() * config.d_model);
    }

    #[test]
    fn test_transformer_deterministic() {
        let config = tiny_config();
        let mut model1 = Transformer::new(config.clone()).unwrap();
        let mut model2 = model1.clone();

        let tokens = vec![1, 2, 3];

        let out1 = model1.forward(&tokens);
        let out2 = model2.forward(&tokens);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_transformer_different_lengths() {
        let config = tiny_config();
        let mut model = Transformer::new(config.clone()).unwrap();

        // Short sequence
        let short = vec![1, 2];
        let short_out = model.forward(&short);
        assert_eq!(short_out.len(), 2 * config.vocab_size);

        // Longer sequence
        let long = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let long_out = model.forward(&long);
        assert_eq!(long_out.len(), 8 * config.vocab_size);
    }

    #[test]
    fn test_transformer_temperature_affects_sampling() {
        let config = tiny_config();
        let mut model = Transformer::new(config).unwrap();

        let tokens = vec![1, 2, 3];
        model.forward(&tokens);

        // Low temperature should give more deterministic results
        let greedy = model.greedy_next_token().unwrap();

        // With temperature 0, should match greedy
        // (Note: our sample uses temperature 0 as greedy)
        let cold = model.sample_next_token(0.001).unwrap();

        // They should often be the same token with very low temperature
        // (not guaranteed due to randomness, but very likely)
        println!("Greedy: {}, Cold sample: {}", greedy, cold);
    }

    #[test]
    fn test_transformer_max_sequence_length() {
        let mut config = tiny_config();
        config.max_seq_len = 16;
        let mut model = Transformer::new(config).unwrap();

        let prompt = vec![1, 2, 3];
        let generated = model.generate(&prompt, 100, 1.0);

        // Should not exceed max sequence length
        assert!(generated.len() <= 16);
    }

    // ==================== FULL BACKWARD PASS TESTS ====================

    #[test]
    fn test_model_backward_updates_all_weights() {
        let config = tiny_config();
        let mut model = Transformer::new(config).unwrap();

        // Store initial weights from all components
        let initial_token_emb = model.embedding.token_embedding.weights.clone();
        let initial_pos_emb = model.embedding.position_embedding.weights.clone();
        let initial_block_w_q = model.blocks.blocks[0].attention.w_q.clone();
        let initial_block_ff_w1 = model.blocks.blocks[0].ff.w1.clone();
        let initial_block_ln1_gamma = model.blocks.blocks[0].ln1.gamma.clone();
        let initial_final_norm_gamma = model.final_norm.gamma.clone();

        // Forward pass
        let tokens = vec![1u32, 2, 3, 4];
        let targets = vec![2u32, 3, 4, 5];
        model.forward(&tokens);

        // Backward pass with non-trivial learning rate
        model.backward(&targets, 0.1);

        // Check that token embeddings changed
        let token_changed = model.embedding.token_embedding.weights.iter()
            .zip(initial_token_emb.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(token_changed, "Token embeddings should be updated");

        // Check that position embeddings changed
        let pos_changed = model.embedding.position_embedding.weights.iter()
            .zip(initial_pos_emb.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(pos_changed, "Position embeddings should be updated");

        // Check that attention weights changed
        let w_q_changed = model.blocks.blocks[0].attention.w_q.iter()
            .zip(initial_block_w_q.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(w_q_changed, "Attention W_q should be updated");

        // Check that feed-forward weights changed
        let ff_changed = model.blocks.blocks[0].ff.w1.iter()
            .zip(initial_block_ff_w1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(ff_changed, "FeedForward W1 should be updated");

        // Check that layer norm gamma changed
        let ln1_changed = model.blocks.blocks[0].ln1.gamma.iter()
            .zip(initial_block_ln1_gamma.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(ln1_changed, "Block LayerNorm gamma should be updated");

        // Check that final norm gamma changed
        let final_norm_changed = model.final_norm.gamma.iter()
            .zip(initial_final_norm_gamma.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(final_norm_changed, "Final LayerNorm gamma should be updated");
    }

    #[test]
    fn test_model_training_loss_decreases() {
        let config = tiny_config();
        let mut model = Transformer::new(config).unwrap();

        // Simple repeating pattern to learn
        let pattern: Vec<u32> = (0..20).map(|i| (i % 5) as u32).collect();

        // Measure initial loss
        let mut initial_loss = 0.0;
        for i in 0..15 {
            let input: Vec<u32> = pattern[i..i + 4].to_vec();
            let target: Vec<u32> = pattern[i + 1..i + 5].to_vec();
            initial_loss += model.compute_loss(&input, &target);
        }
        initial_loss /= 15.0;

        // Train for several epochs
        for _ in 0..10 {
            for i in 0..15 {
                let input: Vec<u32> = pattern[i..i + 4].to_vec();
                let target: Vec<u32> = pattern[i + 1..i + 5].to_vec();
                model.forward(&input);
                model.backward(&target, 0.1);
            }
        }

        // Measure final loss
        let mut final_loss = 0.0;
        for i in 0..15 {
            let input: Vec<u32> = pattern[i..i + 4].to_vec();
            let target: Vec<u32> = pattern[i + 1..i + 5].to_vec();
            final_loss += model.compute_loss(&input, &target);
        }
        final_loss /= 15.0;

        println!("Initial loss: {:.4}, Final loss: {:.4}", initial_loss, final_loss);

        // Loss should decrease with training
        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={:.4}, final={:.4}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_model_backward_gradient_flow() {
        let config = tiny_config();
        let mut model = Transformer::new(config).unwrap();

        // Forward + backward with zero learning rate (just compute gradients)
        let tokens = vec![1u32, 2, 3];
        let targets = vec![2u32, 3, 4];
        model.forward(&tokens);
        model.backward(&targets, 0.0);

        // Embedding grads should have some non-zero values
        let has_nonzero_grad = model.embedding_grads.iter().any(|&g| g.abs() > 1e-10);
        assert!(has_nonzero_grad, "Embedding gradients should have non-zero values");
    }
}
