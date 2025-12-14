//! BPE (Byte Pair Encoding) Tokenizer
//!
//! A production-grade tokenizer for language model training.
//! Implements the BPE algorithm from scratch with byte-level fallback.

use serde::{Deserialize, Serialize};

mod bpe;
mod vocab;

pub use bpe::BpeTokenizer;
pub use vocab::Vocabulary;

/// Special tokens used by the tokenizer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialToken {
    /// Padding token
    Pad,
    /// Unknown token (byte fallback)
    Unk,
    /// Beginning of sequence
    Bos,
    /// End of sequence
    Eos,
}

impl SpecialToken {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpecialToken::Pad => "<PAD>",
            SpecialToken::Unk => "<UNK>",
            SpecialToken::Bos => "<BOS>",
            SpecialToken::Eos => "<EOS>",
        }
    }

    pub fn all() -> [SpecialToken; 4] {
        [
            SpecialToken::Pad,
            SpecialToken::Unk,
            SpecialToken::Bos,
            SpecialToken::Eos,
        ]
    }
}

/// Configuration for BPE training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeConfig {
    /// Target vocabulary size (including special tokens)
    pub vocab_size: usize,
    /// Minimum frequency for a pair to be merged
    pub min_frequency: u32,
    /// Whether to use byte-level BPE (recommended)
    pub byte_level: bool,
    /// Maximum token length in characters
    pub max_token_length: usize,
}

impl Default for BpeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            min_frequency: 2,
            byte_level: true,
            max_token_length: 100,
        }
    }
}

/// Error type for tokenizer operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// Vocabulary has not been trained
    NotTrained,
    /// Token not found in vocabulary
    TokenNotFound(String),
    /// Invalid token ID
    InvalidTokenId(u32),
    /// Serialization error
    SerializationError(String),
    /// IO error
    IoError(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::NotTrained => write!(f, "Tokenizer has not been trained"),
            TokenizerError::TokenNotFound(t) => write!(f, "Token not found: {}", t),
            TokenizerError::InvalidTokenId(id) => write!(f, "Invalid token ID: {}", id),
            TokenizerError::SerializationError(e) => write!(f, "Serialization error: {}", e),
            TokenizerError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for TokenizerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens() {
        assert_eq!(SpecialToken::Pad.as_str(), "<PAD>");
        assert_eq!(SpecialToken::Unk.as_str(), "<UNK>");
        assert_eq!(SpecialToken::Bos.as_str(), "<BOS>");
        assert_eq!(SpecialToken::Eos.as_str(), "<EOS>");
    }

    #[test]
    fn test_special_tokens_all() {
        let all = SpecialToken::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_bpe_config_default() {
        let config = BpeConfig::default();
        assert_eq!(config.vocab_size, 8192);
        assert_eq!(config.min_frequency, 2);
        assert!(config.byte_level);
        assert_eq!(config.max_token_length, 100);
    }

    #[test]
    fn test_tokenizer_error_display() {
        let err = TokenizerError::NotTrained;
        assert_eq!(format!("{}", err), "Tokenizer has not been trained");

        let err = TokenizerError::TokenNotFound("test".to_string());
        assert_eq!(format!("{}", err), "Token not found: test");

        let err = TokenizerError::InvalidTokenId(42);
        assert_eq!(format!("{}", err), "Invalid token ID: 42");
    }
}
