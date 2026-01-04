//! Vocabulary management for the tokenizer

use crate::{SpecialToken, TokenizerError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vocabulary mapping between tokens and IDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: Vec<String>,
    /// Special token IDs
    special_tokens: HashMap<SpecialToken, u32>,
}

impl Vocabulary {
    /// Create a new vocabulary with special tokens
    pub fn new() -> Self {
        let mut vocab = Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            special_tokens: HashMap::new(),
        };

        // Add special tokens first (IDs 0-3)
        for special in SpecialToken::all() {
            let id = vocab.id_to_token.len() as u32;
            let token = special.as_str().to_string();
            vocab.token_to_id.insert(token.clone(), id);
            vocab.id_to_token.push(token);
            vocab.special_tokens.insert(special, id);
        }

        vocab
    }

    /// Get the vocabulary size
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Check if vocabulary is empty (only special tokens)
    pub fn is_empty(&self) -> bool {
        self.id_to_token.len() <= SpecialToken::all().len()
    }

    /// Add a token to the vocabulary
    /// Returns the token ID (existing or new)
    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        id
    }

    /// Get token ID by token string
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token string by ID
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Get the ID for a special token
    pub fn special_token_id(&self, token: SpecialToken) -> u32 {
        self.special_tokens[&token]
    }

    /// Get the PAD token ID
    pub fn pad_id(&self) -> u32 {
        self.special_token_id(SpecialToken::Pad)
    }

    /// Get the UNK token ID
    pub fn unk_id(&self) -> u32 {
        self.special_token_id(SpecialToken::Unk)
    }

    /// Get the BOS token ID
    pub fn bos_id(&self) -> u32 {
        self.special_token_id(SpecialToken::Bos)
    }

    /// Get the EOS token ID
    pub fn eos_id(&self) -> u32 {
        self.special_token_id(SpecialToken::Eos)
    }

    /// Check if a token exists in the vocabulary
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Get all tokens (excluding special tokens)
    pub fn tokens(&self) -> impl Iterator<Item = &str> {
        self.id_to_token
            .iter()
            .skip(SpecialToken::all().len())
            .map(|s| s.as_str())
    }

    /// Serialize vocabulary to JSON
    pub fn to_json(&self) -> Result<String, TokenizerError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| TokenizerError::SerializationError(e.to_string()))
    }

    /// Deserialize vocabulary from JSON
    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        serde_json::from_str(json).map_err(|e| TokenizerError::SerializationError(e.to_string()))
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_new() {
        let vocab = Vocabulary::new();
        // Should have 4 special tokens
        assert_eq!(vocab.len(), 4);
        assert!(vocab.is_empty()); // Empty means only special tokens
    }

    #[test]
    fn test_special_token_ids() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.pad_id(), 0);
        assert_eq!(vocab.unk_id(), 1);
        assert_eq!(vocab.bos_id(), 2);
        assert_eq!(vocab.eos_id(), 3);
    }

    #[test]
    fn test_add_token() {
        let mut vocab = Vocabulary::new();
        let id1 = vocab.add_token("hello");
        let id2 = vocab.add_token("world");
        let id3 = vocab.add_token("hello"); // duplicate

        assert_eq!(id1, 4); // After 4 special tokens
        assert_eq!(id2, 5);
        assert_eq!(id3, 4); // Same as first "hello"
        assert_eq!(vocab.len(), 6);
    }

    #[test]
    fn test_get_id_and_token() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("test");

        assert_eq!(vocab.get_id("test"), Some(4));
        assert_eq!(vocab.get_id("nonexistent"), None);
        assert_eq!(vocab.get_token(4), Some("test"));
        assert_eq!(vocab.get_token(999), None);
    }

    #[test]
    fn test_contains() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("exists");

        assert!(vocab.contains("exists"));
        assert!(vocab.contains("<PAD>")); // Special token
        assert!(!vocab.contains("missing"));
    }

    #[test]
    fn test_tokens_iterator() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("a");
        vocab.add_token("b");
        vocab.add_token("c");

        let tokens: Vec<_> = vocab.tokens().collect();
        assert_eq!(tokens, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello");
        vocab.add_token("world");

        let json = vocab.to_json().unwrap();
        let loaded = Vocabulary::from_json(&json).unwrap();

        assert_eq!(loaded.len(), vocab.len());
        assert_eq!(loaded.get_id("hello"), vocab.get_id("hello"));
        assert_eq!(loaded.get_id("world"), vocab.get_id("world"));
    }

    #[test]
    fn test_special_tokens_preserved_in_serialization() {
        let vocab = Vocabulary::new();
        let json = vocab.to_json().unwrap();
        let loaded = Vocabulary::from_json(&json).unwrap();

        assert_eq!(loaded.pad_id(), 0);
        assert_eq!(loaded.unk_id(), 1);
        assert_eq!(loaded.bos_id(), 2);
        assert_eq!(loaded.eos_id(), 3);
    }
}
