//! Token and positional embeddings for the transformer

use crate::{normal_init, TransformerConfig};
use serde::{Deserialize, Serialize};

/// Token embedding layer
///
/// Maps token IDs to dense vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEmbedding {
    /// Embedding weights [vocab_size, d_model]
    pub weights: Vec<f32>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub d_model: usize,
}

impl TokenEmbedding {
    /// Create new token embeddings with random initialization
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        // Initialize with small random values
        let std = 0.02;
        let weights: Vec<f32> = (0..vocab_size * d_model)
            .map(|_| normal_init(std))
            .collect();

        Self {
            weights,
            vocab_size,
            d_model,
        }
    }

    /// Create from config
    pub fn from_config(config: &TransformerConfig) -> Self {
        Self::new(config.vocab_size, config.d_model)
    }

    /// Get embedding for a single token
    pub fn forward_single(&self, token_id: u32) -> Vec<f32> {
        let idx = token_id as usize;
        assert!(idx < self.vocab_size, "Token ID {} out of vocabulary range {}", idx, self.vocab_size);

        let start = idx * self.d_model;
        self.weights[start..start + self.d_model].to_vec()
    }

    /// Get embeddings for a sequence of tokens
    /// Returns [seq_len, d_model] as a flat vector
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let seq_len = token_ids.len();
        let mut output = vec![0.0f32; seq_len * self.d_model];

        for (pos, &token_id) in token_ids.iter().enumerate() {
            let idx = token_id as usize;
            assert!(idx < self.vocab_size, "Token ID {} out of vocabulary range", idx);

            let src_start = idx * self.d_model;
            let dst_start = pos * self.d_model;

            output[dst_start..dst_start + self.d_model]
                .copy_from_slice(&self.weights[src_start..src_start + self.d_model]);
        }

        output
    }

    /// Get a mutable reference to the weights (for training)
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Get the weight at a specific position
    #[inline]
    pub fn get_weight(&self, token_id: usize, dim: usize) -> f32 {
        self.weights[token_id * self.d_model + dim]
    }

    /// Set the weight at a specific position
    #[inline]
    pub fn set_weight(&mut self, token_id: usize, dim: usize, value: f32) {
        self.weights[token_id * self.d_model + dim] = value;
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.len()
    }
}

/// Positional encoding using learned embeddings
///
/// This is the GPT-style learned positional embeddings approach.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionalEmbedding {
    /// Position embedding weights [max_seq_len, d_model]
    pub weights: Vec<f32>,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Embedding dimension
    pub d_model: usize,
}

impl PositionalEmbedding {
    /// Create new positional embeddings with random initialization
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let std = 0.02;
        let weights: Vec<f32> = (0..max_seq_len * d_model)
            .map(|_| normal_init(std))
            .collect();

        Self {
            weights,
            max_seq_len,
            d_model,
        }
    }

    /// Create from config
    pub fn from_config(config: &TransformerConfig) -> Self {
        Self::new(config.max_seq_len, config.d_model)
    }

    /// Add positional embeddings to input embeddings in-place
    pub fn add_to(&self, embeddings: &mut [f32], seq_len: usize) {
        assert!(seq_len <= self.max_seq_len, "Sequence length {} exceeds maximum {}", seq_len, self.max_seq_len);
        assert_eq!(embeddings.len(), seq_len * self.d_model);

        for pos in 0..seq_len {
            let offset = pos * self.d_model;
            for dim in 0..self.d_model {
                embeddings[offset + dim] += self.weights[pos * self.d_model + dim];
            }
        }
    }

    /// Get positional embeddings for a given sequence length
    /// Returns [seq_len, d_model] as a flat vector
    pub fn forward(&self, seq_len: usize) -> Vec<f32> {
        assert!(seq_len <= self.max_seq_len, "Sequence length {} exceeds maximum {}", seq_len, self.max_seq_len);

        self.weights[..seq_len * self.d_model].to_vec()
    }

    /// Get a mutable reference to the weights (for training)
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.len()
    }
}

/// Sinusoidal positional encoding (fixed, not learned)
///
/// Uses sine and cosine functions of different frequencies.
/// This is the original transformer paper approach.
#[derive(Debug, Clone)]
pub struct SinusoidalPositionalEncoding {
    /// Precomputed encodings [max_seq_len, d_model]
    encodings: Vec<f32>,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Embedding dimension
    pub d_model: usize,
}

impl SinusoidalPositionalEncoding {
    /// Create sinusoidal positional encodings
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let mut encodings = vec![0.0f32; max_seq_len * d_model];

        for pos in 0..max_seq_len {
            for i in 0..d_model / 2 {
                let angle = pos as f32 / (10000.0f32).powf(2.0 * i as f32 / d_model as f32);
                encodings[pos * d_model + 2 * i] = angle.sin();
                encodings[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }

        Self {
            encodings,
            max_seq_len,
            d_model,
        }
    }

    /// Add positional encodings to input embeddings in-place
    pub fn add_to(&self, embeddings: &mut [f32], seq_len: usize) {
        assert!(seq_len <= self.max_seq_len);
        assert_eq!(embeddings.len(), seq_len * self.d_model);

        for i in 0..seq_len * self.d_model {
            embeddings[i] += self.encodings[i];
        }
    }

    /// Get encodings for a given sequence length
    pub fn forward(&self, seq_len: usize) -> Vec<f32> {
        assert!(seq_len <= self.max_seq_len);
        self.encodings[..seq_len * self.d_model].to_vec()
    }
}

/// Combined embedding layer (tokens + positions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayer {
    /// Token embeddings
    pub token_embedding: TokenEmbedding,
    /// Positional embeddings
    pub position_embedding: PositionalEmbedding,
    /// Scaling factor (sqrt(d_model))
    pub scale: f32,
}

impl EmbeddingLayer {
    /// Create new embedding layer
    pub fn new(vocab_size: usize, max_seq_len: usize, d_model: usize) -> Self {
        Self {
            token_embedding: TokenEmbedding::new(vocab_size, d_model),
            position_embedding: PositionalEmbedding::new(max_seq_len, d_model),
            scale: (d_model as f32).sqrt(),
        }
    }

    /// Create from config
    pub fn from_config(config: &TransformerConfig) -> Self {
        Self::new(config.vocab_size, config.max_seq_len, config.d_model)
    }

    /// Forward pass: get combined embeddings for token sequence
    /// Returns [seq_len, d_model] as a flat vector
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let seq_len = token_ids.len();

        // Get token embeddings and scale by √d_model
        let mut embeddings = self.token_embedding.forward(token_ids);
        for v in embeddings.iter_mut() {
            *v *= self.scale;
        }

        // Add positional embeddings (also scaled by √d_model for balance with token embeddings)
        // This is critical: without scaling, position embeddings are ~8x smaller than token
        // embeddings for d_model=64, causing the model to essentially ignore position info
        let pos_embeddings = self.position_embedding.forward(seq_len);
        for (i, v) in pos_embeddings.iter().enumerate() {
            embeddings[i] += v * self.scale;
        }

        embeddings
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.token_embedding.num_parameters() + self.position_embedding.num_parameters()
    }

    /// Get the embedding dimension
    pub fn d_model(&self) -> usize {
        self.token_embedding.d_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embedding_forward_single() {
        let vocab_size = 100;
        let d_model = 32;
        let embedding = TokenEmbedding::new(vocab_size, d_model);

        let result = embedding.forward_single(5);
        assert_eq!(result.len(), d_model);
    }

    #[test]
    fn test_token_embedding_forward_sequence() {
        let vocab_size = 100;
        let d_model = 32;
        let embedding = TokenEmbedding::new(vocab_size, d_model);

        let tokens = vec![1, 5, 10, 20];
        let result = embedding.forward(&tokens);

        assert_eq!(result.len(), tokens.len() * d_model);
    }

    #[test]
    fn test_token_embedding_consistency() {
        let embedding = TokenEmbedding::new(100, 32);

        let single = embedding.forward_single(5);
        let sequence = embedding.forward(&[5]);

        // Single and first element of sequence should match
        assert_eq!(single, sequence);
    }

    #[test]
    #[should_panic(expected = "Token ID")]
    fn test_token_embedding_out_of_range() {
        let embedding = TokenEmbedding::new(100, 32);
        embedding.forward_single(200); // Out of range
    }

    #[test]
    fn test_positional_embedding_add_to() {
        let max_seq_len = 64;
        let d_model = 32;
        let pos_emb = PositionalEmbedding::new(max_seq_len, d_model);

        let mut embeddings = vec![1.0f32; 4 * d_model]; // 4 positions
        let original = embeddings.clone();

        pos_emb.add_to(&mut embeddings, 4);

        // Values should have changed
        assert_ne!(embeddings, original);
    }

    #[test]
    fn test_positional_embedding_forward() {
        let pos_emb = PositionalEmbedding::new(64, 32);
        let result = pos_emb.forward(10);

        assert_eq!(result.len(), 10 * 32);
    }

    #[test]
    #[should_panic(expected = "Sequence length")]
    fn test_positional_embedding_too_long() {
        let pos_emb = PositionalEmbedding::new(64, 32);
        pos_emb.forward(100); // Exceeds max_seq_len
    }

    #[test]
    fn test_sinusoidal_encoding() {
        let encoding = SinusoidalPositionalEncoding::new(64, 32);

        // Position 0 should have specific pattern
        let pos0 = &encoding.encodings[0..32];
        // sin(0) = 0, cos(0) = 1
        assert!((pos0[0] - 0.0).abs() < 1e-5); // sin(0)
        assert!((pos0[1] - 1.0).abs() < 1e-5); // cos(0)
    }

    #[test]
    fn test_sinusoidal_different_positions() {
        let encoding = SinusoidalPositionalEncoding::new(64, 32);

        let pos0 = encoding.forward(1);
        let mut embeddings = vec![0.0f32; 32];
        embeddings.copy_from_slice(&pos0);

        // Different positions should have different encodings
        let pos1_encodings = &encoding.encodings[32..64];

        let mut all_same = true;
        for i in 0..32 {
            if (pos0[i] - pos1_encodings[i]).abs() > 1e-5 {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "Different positions should have different encodings");
    }

    #[test]
    fn test_embedding_layer_forward() {
        let layer = EmbeddingLayer::new(100, 64, 32);

        let tokens = vec![1, 2, 3, 4, 5];
        let result = layer.forward(&tokens);

        assert_eq!(result.len(), 5 * 32);
    }

    #[test]
    fn test_embedding_layer_from_config() {
        let config = TransformerConfig::tiny(1000);
        let layer = EmbeddingLayer::from_config(&config);

        assert_eq!(layer.token_embedding.vocab_size, 1000);
        assert_eq!(layer.token_embedding.d_model, config.d_model);
        assert_eq!(layer.position_embedding.max_seq_len, config.max_seq_len);
    }

    #[test]
    fn test_embedding_layer_num_parameters() {
        let layer = EmbeddingLayer::new(1000, 64, 32);
        let params = layer.num_parameters();

        // Token: 1000 * 32 = 32000
        // Position: 64 * 32 = 2048
        // Total: 34048
        assert_eq!(params, 1000 * 32 + 64 * 32);
    }

    #[test]
    fn test_embedding_scaling() {
        let layer = EmbeddingLayer::new(100, 64, 32);

        // Check that scale is sqrt(d_model)
        assert!((layer.scale - (32.0f32).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_get_set_weight() {
        let mut embedding = TokenEmbedding::new(100, 32);

        embedding.set_weight(5, 10, 42.0);
        assert_eq!(embedding.get_weight(5, 10), 42.0);
    }
}
