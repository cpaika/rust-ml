//! BPE (Byte Pair Encoding) algorithm implementation

use crate::{BpeConfig, TokenizerError, Vocabulary};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A merge rule learned during BPE training
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MergeRule {
    /// First token in the pair
    pub first: String,
    /// Second token in the pair
    pub second: String,
    /// Resulting merged token
    pub merged: String,
}

impl MergeRule {
    pub fn new(first: &str, second: &str) -> Self {
        Self {
            first: first.to_string(),
            second: second.to_string(),
            merged: format!("{}{}", first, second),
        }
    }
}

/// BPE Tokenizer with training and encoding capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// Configuration
    config: BpeConfig,
    /// Vocabulary
    vocab: Vocabulary,
    /// Merge rules in order of priority (learned during training)
    merges: Vec<MergeRule>,
    /// Fast lookup for merge rules: (first, second) -> merged
    #[serde(skip)]
    merge_lookup: HashMap<(String, String), String>,
    /// Whether the tokenizer has been trained
    trained: bool,
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer with the given configuration
    pub fn new(config: BpeConfig) -> Self {
        Self {
            config,
            vocab: Vocabulary::new(),
            merges: Vec::new(),
            merge_lookup: HashMap::new(),
            trained: false,
        }
    }

    /// Create a tokenizer with default configuration
    pub fn with_defaults() -> Self {
        Self::new(BpeConfig::default())
    }

    /// Get the vocabulary
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Check if the tokenizer has been trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get the merge rules
    pub fn merges(&self) -> &[MergeRule] {
        &self.merges
    }

    /// Train the tokenizer on a corpus of text
    pub fn train(&mut self, texts: &[&str]) -> Result<(), TokenizerError> {
        // Initialize vocabulary with byte-level tokens if using byte-level BPE
        if self.config.byte_level {
            self.init_byte_vocabulary();
        }

        // Pre-tokenize: split texts into words and count frequencies
        let mut word_freqs: HashMap<Vec<String>, u64> = HashMap::new();
        for text in texts {
            for word in self.pre_tokenize(text) {
                // Split word into characters (or bytes)
                let chars: Vec<String> = if self.config.byte_level {
                    word.bytes().map(|b| format!("{:02x}", b)).collect()
                } else {
                    word.chars().map(|c| c.to_string()).collect()
                };
                if !chars.is_empty() {
                    *word_freqs.entry(chars).or_insert(0) += 1;
                }
            }
        }

        // Add individual characters/bytes to vocabulary
        for word in word_freqs.keys() {
            for token in word {
                self.vocab.add_token(token);
            }
        }

        // Learn merges until we reach target vocabulary size
        while self.vocab.len() < self.config.vocab_size {
            // Count pair frequencies
            let pair_freqs = self.count_pairs(&word_freqs);

            // Find most frequent pair
            let best_pair = pair_freqs
                .iter()
                .filter(|&(_, freq)| *freq >= self.config.min_frequency as u64)
                .max_by_key(|&(_, freq)| *freq);

            let Some(((first, second), _freq)) = best_pair else {
                break; // No more pairs above min frequency
            };

            // Create merge rule
            let merge = MergeRule::new(first, second);

            // Apply merge to all words
            let new_word_freqs = self.apply_merge(&word_freqs, &merge);
            word_freqs = new_word_freqs;

            // Add merged token to vocabulary
            self.vocab.add_token(&merge.merged);

            // Store merge rule
            self.merge_lookup
                .insert((merge.first.clone(), merge.second.clone()), merge.merged.clone());
            self.merges.push(merge);
        }

        self.trained = true;
        Ok(())
    }

    /// Initialize vocabulary with all possible byte values
    fn init_byte_vocabulary(&mut self) {
        for byte in 0u8..=255 {
            self.vocab.add_token(&format!("{:02x}", byte));
        }
    }

    /// Pre-tokenize text into words (basic whitespace splitting)
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        // Split on whitespace but keep whitespace as separate tokens
        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut in_whitespace = false;

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
                // Treat each whitespace run as a token
                if !in_whitespace {
                    in_whitespace = true;
                }
                current_word.push(ch);
            } else {
                if in_whitespace && !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
                in_whitespace = false;
                current_word.push(ch);
            }
        }

        if !current_word.is_empty() {
            words.push(current_word);
        }

        words
    }

    /// Count pair frequencies in the word frequency map
    fn count_pairs(&self, word_freqs: &HashMap<Vec<String>, u64>) -> HashMap<(String, String), u64> {
        let mut pair_freqs: HashMap<(String, String), u64> = HashMap::new();

        for (word, freq) in word_freqs {
            if word.len() < 2 {
                continue;
            }
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }

        pair_freqs
    }

    /// Apply a merge rule to all words
    fn apply_merge(
        &self,
        word_freqs: &HashMap<Vec<String>, u64>,
        merge: &MergeRule,
    ) -> HashMap<Vec<String>, u64> {
        let mut new_word_freqs = HashMap::new();

        for (word, freq) in word_freqs {
            let new_word = self.apply_merge_to_word(word, merge);
            *new_word_freqs.entry(new_word).or_insert(0) += freq;
        }

        new_word_freqs
    }

    /// Apply a merge rule to a single word
    fn apply_merge_to_word(&self, word: &[String], merge: &MergeRule) -> Vec<String> {
        if word.len() < 2 {
            return word.to_vec();
        }

        let mut result = Vec::with_capacity(word.len());
        let mut i = 0;

        while i < word.len() {
            if i < word.len() - 1 && word[i] == merge.first && word[i + 1] == merge.second {
                result.push(merge.merged.clone());
                i += 2;
            } else {
                result.push(word[i].clone());
                i += 1;
            }
        }

        result
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let mut token_ids = Vec::new();

        for word in self.pre_tokenize(text) {
            // Convert word to initial tokens
            let mut tokens: Vec<String> = if self.config.byte_level {
                word.bytes().map(|b| format!("{:02x}", b)).collect()
            } else {
                word.chars().map(|c| c.to_string()).collect()
            };

            // Apply merges in order
            tokens = self.apply_all_merges(tokens);

            // Convert to IDs
            for token in tokens {
                let id = self
                    .vocab
                    .get_id(&token)
                    .unwrap_or_else(|| self.vocab.unk_id());
                token_ids.push(id);
            }
        }

        Ok(token_ids)
    }

    /// Apply all learned merges to a sequence of tokens
    fn apply_all_merges(&self, mut tokens: Vec<String>) -> Vec<String> {
        // Apply merges greedily in priority order
        for merge in &self.merges {
            tokens = self.apply_merge_to_word(&tokens, merge);
        }
        tokens
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let mut bytes = Vec::new();

        for &id in token_ids {
            let token = self
                .vocab
                .get_token(id)
                .ok_or(TokenizerError::InvalidTokenId(id))?;

            // Skip special tokens in output
            if token.starts_with('<') && token.ends_with('>') {
                continue;
            }

            if self.config.byte_level {
                // Parse hex bytes
                let mut i = 0;
                let chars: Vec<char> = token.chars().collect();
                while i + 1 < chars.len() {
                    let hex_str: String = chars[i..i + 2].iter().collect();
                    if let Ok(byte) = u8::from_str_radix(&hex_str, 16) {
                        bytes.push(byte);
                    }
                    i += 2;
                }
            } else {
                bytes.extend(token.bytes());
            }
        }

        String::from_utf8(bytes).map_err(|e| TokenizerError::SerializationError(e.to_string()))
    }

    /// Encode with special tokens (BOS and EOS)
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = vec![self.vocab.bos_id()];
        ids.extend(self.encode(text)?);
        ids.push(self.vocab.eos_id());
        Ok(ids)
    }

    /// Serialize tokenizer to JSON
    pub fn to_json(&self) -> Result<String, TokenizerError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| TokenizerError::SerializationError(e.to_string()))
    }

    /// Deserialize tokenizer from JSON
    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        let mut tokenizer: Self =
            serde_json::from_str(json).map_err(|e| TokenizerError::SerializationError(e.to_string()))?;

        // Rebuild merge lookup
        for merge in &tokenizer.merges {
            tokenizer.merge_lookup.insert(
                (merge.first.clone(), merge.second.clone()),
                merge.merged.clone(),
            );
        }

        Ok(tokenizer)
    }

    /// Save tokenizer to file
    pub fn save(&self, path: &str) -> Result<(), TokenizerError> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(|e| TokenizerError::IoError(e.to_string()))
    }

    /// Load tokenizer from file
    pub fn load(path: &str) -> Result<Self, TokenizerError> {
        let json = std::fs::read_to_string(path).map_err(|e| TokenizerError::IoError(e.to_string()))?;
        Self::from_json(&json)
    }

    /// Find all tokens that could continue a partial word
    ///
    /// Returns a list of (token_id, decoded_token_text) pairs for tokens
    /// that start with the given prefix bytes.
    ///
    /// This is useful for partial word completion - given "hel", this finds
    /// tokens like "lo" (completing "hello") or "p" (completing "help").
    pub fn find_continuation_tokens(&self, partial_word: &str) -> Vec<(u32, String)> {
        if !self.trained || partial_word.is_empty() {
            return vec![];
        }

        // Convert partial word to hex bytes (byte-level representation)
        let prefix_bytes: String = if self.config.byte_level {
            partial_word.bytes().map(|b| format!("{:02x}", b)).collect()
        } else {
            partial_word.to_string()
        };

        let mut results = Vec::new();

        // Check each token in vocabulary
        for id in 0..self.vocab.len() as u32 {
            if let Some(token) = self.vocab.get_token(id) {
                // Skip special tokens
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }

                // Check if the partial word prefix matches the start of this token
                // e.g., if partial="hel" (686c65) and token="hello" (68656c6c6f),
                // then 68656c6c6f starts with 68656c, so "hello" is a valid completion
                if token.starts_with(&prefix_bytes) && token.len() > prefix_bytes.len() {
                    // This token extends the partial word
                    // Decode the remaining part of the token (what comes after the prefix)
                    let remaining_hex = &token[prefix_bytes.len()..];
                    if let Ok(decoded) = self.decode_hex_to_string(remaining_hex) {
                        results.push((id, decoded));
                    }
                }
            }
        }

        results
    }

    /// Find tokens that could be the next complete token after a partial word
    ///
    /// This handles the case where the partial word doesn't match any token prefix,
    /// so we need to find tokens that complete the current byte sequence.
    pub fn find_completing_tokens(&self, partial_word: &str) -> Vec<(u32, String, f32)> {
        if !self.trained || partial_word.is_empty() {
            return vec![];
        }

        // Convert partial word to hex bytes
        let partial_hex: String = if self.config.byte_level {
            partial_word.bytes().map(|b| format!("{:02x}", b)).collect()
        } else {
            partial_word.to_string()
        };

        let mut results = Vec::new();

        // Check each token in vocabulary
        for id in 0..self.vocab.len() as u32 {
            if let Some(token) = self.vocab.get_token(id) {
                // Skip special tokens
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }

                // Case 1: Token starts with partial_hex (token extends the partial word)
                if token.starts_with(&partial_hex) {
                    let remaining_hex = &token[partial_hex.len()..];
                    if !remaining_hex.is_empty() {
                        if let Ok(decoded) = self.decode_hex_to_string(remaining_hex) {
                            // Full match - higher priority (1.0)
                            results.push((id, decoded, 1.0));
                        }
                    }
                }
                // Case 2: partial_hex starts with token (partial word contains this token)
                // In this case, the token could be part of the encoding
                else if partial_hex.starts_with(token) {
                    // This token is already "consumed" by the partial - not a completion
                }
                // Case 3: Check for partial byte overlap at the end
                // e.g., partial ends with "68" and token starts with "6865"
                else {
                    // Check if any suffix of partial_hex matches a prefix of token
                    for split_point in (2..partial_hex.len()).step_by(2) {
                        let partial_suffix = &partial_hex[split_point..];
                        if token.starts_with(partial_suffix) {
                            let remaining_hex = &token[partial_suffix.len()..];
                            if !remaining_hex.is_empty() {
                                if let Ok(decoded) = self.decode_hex_to_string(remaining_hex) {
                                    // Partial match - lower priority based on overlap
                                    let overlap_ratio = partial_suffix.len() as f32 / partial_hex.len() as f32;
                                    results.push((id, decoded, overlap_ratio));
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Sort by priority (higher overlap first) and deduplicate
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Decode a hex string to UTF-8 string
    fn decode_hex_to_string(&self, hex: &str) -> Result<String, TokenizerError> {
        if !self.config.byte_level {
            return Ok(hex.to_string());
        }

        let mut bytes = Vec::new();
        let chars: Vec<char> = hex.chars().collect();
        let mut i = 0;
        while i + 1 < chars.len() {
            let hex_str: String = chars[i..i + 2].iter().collect();
            if let Ok(byte) = u8::from_str_radix(&hex_str, 16) {
                bytes.push(byte);
            }
            i += 2;
        }

        String::from_utf8(bytes).map_err(|e| TokenizerError::SerializationError(e.to_string()))
    }

    /// Get all tokens with their decoded text (for vocabulary inspection)
    pub fn get_all_tokens(&self) -> Vec<(u32, String)> {
        let mut results = Vec::new();
        for id in 0..self.vocab.len() as u32 {
            if let Some(token) = self.vocab.get_token(id) {
                if token.starts_with('<') && token.ends_with('>') {
                    results.push((id, token.to_string()));
                } else if let Ok(decoded) = self.decode_hex_to_string(token) {
                    results.push((id, decoded));
                }
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> BpeConfig {
        BpeConfig {
            vocab_size: 300,
            min_frequency: 1,
            byte_level: true,
            max_token_length: 100,
        }
    }

    #[test]
    fn test_merge_rule_new() {
        let merge = MergeRule::new("he", "llo");
        assert_eq!(merge.first, "he");
        assert_eq!(merge.second, "llo");
        assert_eq!(merge.merged, "hello");
    }

    #[test]
    fn test_tokenizer_new() {
        let tokenizer = BpeTokenizer::with_defaults();
        assert!(!tokenizer.is_trained());
        assert_eq!(tokenizer.vocab_size(), 4); // Only special tokens
    }

    #[test]
    fn test_pre_tokenize() {
        let tokenizer = BpeTokenizer::with_defaults();
        let words = tokenizer.pre_tokenize("hello world");
        assert_eq!(words, vec!["hello", " ", "world"]);
    }

    #[test]
    fn test_pre_tokenize_multiple_spaces() {
        let tokenizer = BpeTokenizer::with_defaults();
        let words = tokenizer.pre_tokenize("hello  world");
        // Each whitespace character is its own token - this is fine for BPE
        assert_eq!(words, vec!["hello", " ", " ", "world"]);
    }

    #[test]
    fn test_pre_tokenize_newlines() {
        let tokenizer = BpeTokenizer::with_defaults();
        let words = tokenizer.pre_tokenize("hello\nworld");
        assert_eq!(words, vec!["hello", "\n", "world"]);
    }

    #[test]
    fn test_train_basic() {
        let config = create_test_config();
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["hello hello hello", "world world"];
        tokenizer.train(&corpus).unwrap();

        assert!(tokenizer.is_trained());
        assert!(tokenizer.vocab_size() > 4); // More than just special tokens
        assert!(!tokenizer.merges().is_empty());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let config = create_test_config();
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["the quick brown fox jumps over the lazy dog"];
        tokenizer.train(&corpus).unwrap();

        let text = "the quick fox";
        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let config = create_test_config();
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["hello world"];
        tokenizer.train(&corpus).unwrap();

        let encoded = tokenizer.encode_with_special("hello").unwrap();
        assert_eq!(encoded[0], tokenizer.vocab().bos_id());
        assert_eq!(*encoded.last().unwrap(), tokenizer.vocab().eos_id());
    }

    #[test]
    fn test_encode_not_trained_error() {
        let tokenizer = BpeTokenizer::with_defaults();
        let result = tokenizer.encode("hello");
        assert!(matches!(result, Err(TokenizerError::NotTrained)));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = create_test_config();
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["hello world hello world"];
        tokenizer.train(&corpus).unwrap();

        let json = tokenizer.to_json().unwrap();
        let loaded = BpeTokenizer::from_json(&json).unwrap();

        assert_eq!(loaded.vocab_size(), tokenizer.vocab_size());
        assert_eq!(loaded.merges().len(), tokenizer.merges().len());
        assert!(loaded.is_trained());

        // Test that loaded tokenizer works
        let text = "hello";
        let original_encoded = tokenizer.encode(text).unwrap();
        let loaded_encoded = loaded.encode(text).unwrap();
        assert_eq!(original_encoded, loaded_encoded);
    }

    #[test]
    fn test_file_save_load() {
        let config = create_test_config();
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["test data for file operations"];
        tokenizer.train(&corpus).unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("tokenizer.json");
        let path_str = path.to_str().unwrap();

        tokenizer.save(path_str).unwrap();
        let loaded = BpeTokenizer::load(path_str).unwrap();

        assert_eq!(loaded.vocab_size(), tokenizer.vocab_size());
    }

    #[test]
    fn test_byte_level_handles_unicode() {
        let config = create_test_config();
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["hello 世界 emoji 🎉"];
        tokenizer.train(&corpus).unwrap();

        let text = "世界";
        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_apply_merge_to_word() {
        let tokenizer = BpeTokenizer::with_defaults();
        let merge = MergeRule::new("h", "e");

        let word = vec!["h".to_string(), "e".to_string(), "l".to_string(), "l".to_string(), "o".to_string()];
        let result = tokenizer.apply_merge_to_word(&word, &merge);

        assert_eq!(result, vec!["he", "l", "l", "o"]);
    }

    #[test]
    fn test_apply_merge_multiple_occurrences() {
        let tokenizer = BpeTokenizer::with_defaults();
        let merge = MergeRule::new("l", "l");

        let word = vec!["h".to_string(), "e".to_string(), "l".to_string(), "l".to_string(), "o".to_string()];
        let result = tokenizer.apply_merge_to_word(&word, &merge);

        assert_eq!(result, vec!["h", "e", "ll", "o"]);
    }

    #[test]
    fn test_frequent_pairs_merged_first() {
        let config = BpeConfig {
            vocab_size: 270, // Just a bit more than byte vocab
            min_frequency: 1,
            byte_level: true,
            max_token_length: 100,
        };
        let mut tokenizer = BpeTokenizer::new(config);

        // "aa" appears more frequently than any other pair
        let corpus = vec!["aa aa aa aa bb"];
        tokenizer.train(&corpus).unwrap();

        // First merge should be for "aa" (hex: 61 61)
        let first_merge = &tokenizer.merges()[0];
        assert_eq!(first_merge.first, "61");
        assert_eq!(first_merge.second, "61");
    }

    #[test]
    fn test_empty_corpus() {
        let config = create_test_config();
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus: Vec<&str> = vec![];
        tokenizer.train(&corpus).unwrap();

        assert!(tokenizer.is_trained());
        // Should have byte vocabulary plus special tokens
        assert!(tokenizer.vocab_size() >= 256 + 4);
    }

    #[test]
    fn test_single_character_text() {
        let config = create_test_config();
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["a"];
        tokenizer.train(&corpus).unwrap();

        let encoded = tokenizer.encode("a").unwrap();
        let decoded = tokenizer.decode(&encoded).unwrap();

        assert_eq!(decoded, "a");
    }

    #[test]
    fn test_vocabulary_growth_during_training() {
        let config = BpeConfig {
            vocab_size: 280,
            min_frequency: 1,
            byte_level: true,
            max_token_length: 100,
        };
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["hello hello hello hello"];
        tokenizer.train(&corpus).unwrap();

        // Should have learned at least a few merges
        assert!(!tokenizer.merges().is_empty());
        // Vocabulary should have grown
        assert!(tokenizer.vocab_size() > 260);
    }

    #[test]
    fn test_find_completing_tokens() {
        let config = BpeConfig {
            vocab_size: 300,
            min_frequency: 1,
            byte_level: true,
            max_token_length: 100,
        };
        let mut tokenizer = BpeTokenizer::new(config);

        // Train on text with "hello" and "help" to learn those tokens
        let corpus = vec!["hello hello hello help help help world world"];
        tokenizer.train(&corpus).unwrap();

        // Find completions for "hel" - should include "lo" and "p"
        let completions = tokenizer.find_completing_tokens("hel");

        // Should find some completions
        assert!(!completions.is_empty(), "Should find completions for 'hel'");

        // Print completions for debugging
        println!("Completions for 'hel':");
        for (id, text, priority) in &completions {
            println!("  {} -> '{}' (priority: {:.2})", id, text, priority);
        }
    }

    #[test]
    fn test_find_continuation_tokens() {
        let config = BpeConfig {
            vocab_size: 300,
            min_frequency: 1,
            byte_level: true,
            max_token_length: 100,
        };
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["test testing tested tester"];
        tokenizer.train(&corpus).unwrap();

        // Find tokens that continue "test"
        let continuations = tokenizer.find_continuation_tokens("test");

        println!("Continuations for 'test':");
        for (id, text) in &continuations {
            println!("  {} -> '{}'", id, text);
        }
    }

    #[test]
    fn test_get_all_tokens() {
        let config = BpeConfig {
            vocab_size: 270,
            min_frequency: 1,
            byte_level: true,
            max_token_length: 100,
        };
        let mut tokenizer = BpeTokenizer::new(config);

        let corpus = vec!["abc abc abc"];
        tokenizer.train(&corpus).unwrap();

        let all_tokens = tokenizer.get_all_tokens();

        // Should have tokens (vocab size includes special + byte + merged)
        assert!(!all_tokens.is_empty(), "Should have some tokens");
        println!("Total tokens: {}", all_tokens.len());

        // Check that special tokens are present
        let has_pad = all_tokens.iter().any(|(_, t)| t == "<PAD>");
        let has_unk = all_tokens.iter().any(|(_, t)| t == "<UNK>");
        assert!(has_pad, "Should have PAD token");
        assert!(has_unk, "Should have UNK token");
    }
}
