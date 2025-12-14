//! Transformer model configuration

use crate::Activation;
use serde::{Deserialize, Serialize};

/// Configuration for a transformer model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length (context window)
    pub max_seq_len: usize,
    /// Model dimension (embedding size)
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Feed-forward hidden dimension (typically 4 * d_model)
    pub d_ff: usize,
    /// Dropout probability (0.0 for inference)
    pub dropout: f32,
    /// Activation function for feed-forward layers
    pub activation: Activation,
    /// Whether to use pre-layer normalization (GPT-2 style) or post-layer norm
    pub pre_norm: bool,
    /// Epsilon for layer normalization
    pub layer_norm_eps: f32,
    /// Whether to tie embedding and output weights
    pub tie_weights: bool,
}

impl TransformerConfig {
    /// Create a new configuration with the given vocabulary size
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 256,
            d_model: 256,
            n_heads: 4,
            n_layers: 4,
            d_ff: 1024,
            dropout: 0.1,
            activation: Activation::GELU,
            pre_norm: true,
            layer_norm_eps: 1e-5,
            tie_weights: true,
        }
    }

    /// Create a tiny configuration for testing
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 64,
            d_model: 64,
            n_heads: 2,
            n_layers: 2,
            d_ff: 256,
            dropout: 0.0,
            activation: Activation::GELU,
            pre_norm: true,
            layer_norm_eps: 1e-5,
            tie_weights: true,
        }
    }

    /// Create a small configuration suitable for browser training
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 128,
            d_model: 128,
            n_heads: 4,
            n_layers: 4,
            d_ff: 512,
            dropout: 0.1,
            activation: Activation::GELU,
            pre_norm: true,
            layer_norm_eps: 1e-5,
            tie_weights: true,
        }
    }

    /// Create a medium configuration
    pub fn medium(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 256,
            d_model: 256,
            n_heads: 8,
            n_layers: 6,
            d_ff: 1024,
            dropout: 0.1,
            activation: Activation::GELU,
            pre_norm: true,
            layer_norm_eps: 1e-5,
            tie_weights: true,
        }
    }

    /// Get the dimension per attention head
    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Get total number of parameters (approximate)
    pub fn num_parameters(&self) -> usize {
        // Embedding: vocab_size * d_model
        let embedding_params = self.vocab_size * self.d_model;

        // Position embedding: max_seq_len * d_model
        let position_params = self.max_seq_len * self.d_model;

        // Per layer:
        // - Attention: Q, K, V projections (3 * d_model * d_model) + output (d_model * d_model)
        // - Feed-forward: (d_model * d_ff) + (d_ff * d_model)
        // - Layer norms: 2 * (2 * d_model) for pre-norm
        let attention_params = 4 * self.d_model * self.d_model;
        let ff_params = 2 * self.d_model * self.d_ff;
        let norm_params = 4 * self.d_model;
        let layer_params = attention_params + ff_params + norm_params;
        let all_layers = self.n_layers * layer_params;

        // Output: d_model * vocab_size (if not tied, else 0)
        let output_params = if self.tie_weights { 0 } else { self.d_model * self.vocab_size };

        // Final layer norm
        let final_norm = 2 * self.d_model;

        embedding_params + position_params + all_layers + output_params + final_norm
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }

        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".to_string());
        }

        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".to_string());
        }

        if self.n_layers == 0 {
            return Err("n_layers must be > 0".to_string());
        }

        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(format!("dropout must be in [0, 1), got {}", self.dropout));
        }

        Ok(())
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self::new(8192) // Default BPE vocabulary size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = TransformerConfig::new(10000);
        assert_eq!(config.vocab_size, 10000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_tiny() {
        let config = TransformerConfig::tiny(1000);
        assert_eq!(config.d_model, 64);
        assert_eq!(config.n_heads, 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_d_head() {
        let config = TransformerConfig {
            d_model: 256,
            n_heads: 8,
            ..Default::default()
        };
        assert_eq!(config.d_head(), 32);
    }

    #[test]
    fn test_num_parameters() {
        let config = TransformerConfig::tiny(1000);
        let params = config.num_parameters();

        // Should be a reasonable number
        assert!(params > 0);
        println!("Tiny config params: {}", params);
    }

    #[test]
    fn test_validate_d_model_divisibility() {
        let config = TransformerConfig {
            d_model: 100,
            n_heads: 3, // 100 not divisible by 3
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab() {
        let config = TransformerConfig {
            vocab_size: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_dropout() {
        let config = TransformerConfig {
            dropout: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = TransformerConfig::small(5000);
        let json = config.to_json().unwrap();
        let loaded = TransformerConfig::from_json(&json).unwrap();

        assert_eq!(config.vocab_size, loaded.vocab_size);
        assert_eq!(config.d_model, loaded.d_model);
        assert_eq!(config.n_heads, loaded.n_heads);
        assert_eq!(config.n_layers, loaded.n_layers);
    }

    #[test]
    fn test_default_config() {
        let config = TransformerConfig::default();
        assert_eq!(config.vocab_size, 8192);
        assert!(config.validate().is_ok());
    }
}
