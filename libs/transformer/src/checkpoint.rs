//! Model checkpointing and persistence

use crate::{Transformer, TransformerConfig};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Checkpoint containing model state and training metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Model configuration
    pub config: TransformerConfig,
    /// Model weights (serialized)
    pub model_state: ModelState,
    /// Training step
    pub step: u64,
    /// Current epoch
    pub epoch: u32,
    /// Training loss history (last N losses)
    pub loss_history: Vec<f32>,
    /// Best validation loss seen
    pub best_val_loss: Option<f32>,
    /// Tokenizer path (if saved separately)
    pub tokenizer_path: Option<String>,
}

/// Serializable model state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    /// Token embedding weights
    pub token_embedding: Vec<f32>,
    /// Position embedding weights
    pub position_embedding: Vec<f32>,
    /// Block parameters (for each block)
    pub blocks: Vec<BlockState>,
    /// Final layer norm gamma
    pub final_norm_gamma: Vec<f32>,
    /// Final layer norm beta
    pub final_norm_beta: Vec<f32>,
    /// LM head weights (if not tied)
    pub lm_head: Option<Vec<f32>>,
}

/// State for a single transformer block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockState {
    /// Attention weights
    pub attention: AttentionState,
    /// Feed-forward weights
    pub ff: FeedForwardState,
    /// Layer norm 1 (gamma, beta)
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    /// Layer norm 2 (gamma, beta)
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
}

/// Attention layer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionState {
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub b_q: Vec<f32>,
    pub b_k: Vec<f32>,
    pub b_v: Vec<f32>,
    pub b_o: Vec<f32>,
}

/// Feed-forward layer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardState {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl Checkpoint {
    /// Create a checkpoint from a model
    pub fn from_model(
        model: &Transformer,
        step: u64,
        epoch: u32,
        loss_history: Vec<f32>,
        best_val_loss: Option<f32>,
    ) -> Self {
        let model_state = ModelState::from_model(model);

        Self {
            config: model.config.clone(),
            model_state,
            step,
            epoch,
            loss_history,
            best_val_loss,
            tokenizer_path: None,
        }
    }

    /// Restore model from checkpoint
    pub fn restore_model(&self) -> Result<Transformer, String> {
        let mut model = Transformer::new(self.config.clone())?;
        self.model_state.apply_to_model(&mut model)?;
        Ok(model)
    }

    /// Save checkpoint to file
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string(self)
            .map_err(|e| format!("Serialization error: {}", e))?;
        fs::write(path, json)
            .map_err(|e| format!("IO error: {}", e))
    }

    /// Load checkpoint from file
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = fs::read_to_string(path)
            .map_err(|e| format!("IO error: {}", e))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("Deserialization error: {}", e))
    }

    /// Save checkpoint as binary (more efficient)
    pub fn save_binary(&self, path: &Path) -> Result<(), String> {
        let data = bincode_serialize(self)?;
        fs::write(path, data)
            .map_err(|e| format!("IO error: {}", e))
    }

    /// Load checkpoint from binary
    pub fn load_binary(path: &Path) -> Result<Self, String> {
        let data = fs::read(path)
            .map_err(|e| format!("IO error: {}", e))?;
        bincode_deserialize(&data)
    }
}

impl ModelState {
    /// Extract state from a model
    pub fn from_model(model: &Transformer) -> Self {
        let blocks: Vec<BlockState> = model.blocks.blocks.iter()
            .map(BlockState::from_block)
            .collect();

        Self {
            token_embedding: model.embedding.token_embedding.weights.clone(),
            position_embedding: model.embedding.position_embedding.weights.clone(),
            blocks,
            final_norm_gamma: model.final_norm.gamma.clone(),
            final_norm_beta: model.final_norm.beta.clone(),
            lm_head: model.lm_head.clone(),
        }
    }

    /// Apply state to a model
    pub fn apply_to_model(&self, model: &mut Transformer) -> Result<(), String> {
        // Check dimensions match
        if self.token_embedding.len() != model.embedding.token_embedding.weights.len() {
            return Err("Token embedding size mismatch".to_string());
        }

        // Apply token embeddings
        model.embedding.token_embedding.weights.copy_from_slice(&self.token_embedding);

        // Apply position embeddings
        model.embedding.position_embedding.weights.copy_from_slice(&self.position_embedding);

        // Apply block weights
        if self.blocks.len() != model.blocks.blocks.len() {
            return Err("Block count mismatch".to_string());
        }

        for (state, block) in self.blocks.iter().zip(model.blocks.blocks.iter_mut()) {
            state.apply_to_block(block)?;
        }

        // Apply final norm
        model.final_norm.gamma.copy_from_slice(&self.final_norm_gamma);
        model.final_norm.beta.copy_from_slice(&self.final_norm_beta);

        // Apply LM head
        if let (Some(state_head), Some(model_head)) = (&self.lm_head, &mut model.lm_head) {
            model_head.copy_from_slice(state_head);
        }

        Ok(())
    }
}

impl BlockState {
    /// Extract state from a transformer block
    fn from_block(block: &crate::block::TransformerBlock) -> Self {
        Self {
            attention: AttentionState {
                w_q: block.attention.w_q.clone(),
                w_k: block.attention.w_k.clone(),
                w_v: block.attention.w_v.clone(),
                w_o: block.attention.w_o.clone(),
                b_q: block.attention.b_q.clone(),
                b_k: block.attention.b_k.clone(),
                b_v: block.attention.b_v.clone(),
                b_o: block.attention.b_o.clone(),
            },
            ff: FeedForwardState {
                w1: block.ff.w1.clone(),
                b1: block.ff.b1.clone(),
                w2: block.ff.w2.clone(),
                b2: block.ff.b2.clone(),
            },
            ln1_gamma: block.ln1.gamma.clone(),
            ln1_beta: block.ln1.beta.clone(),
            ln2_gamma: block.ln2.gamma.clone(),
            ln2_beta: block.ln2.beta.clone(),
        }
    }

    /// Apply state to a transformer block
    fn apply_to_block(&self, block: &mut crate::block::TransformerBlock) -> Result<(), String> {
        // Attention weights
        block.attention.w_q.copy_from_slice(&self.attention.w_q);
        block.attention.w_k.copy_from_slice(&self.attention.w_k);
        block.attention.w_v.copy_from_slice(&self.attention.w_v);
        block.attention.w_o.copy_from_slice(&self.attention.w_o);
        block.attention.b_q.copy_from_slice(&self.attention.b_q);
        block.attention.b_k.copy_from_slice(&self.attention.b_k);
        block.attention.b_v.copy_from_slice(&self.attention.b_v);
        block.attention.b_o.copy_from_slice(&self.attention.b_o);

        // Feed-forward weights
        block.ff.w1.copy_from_slice(&self.ff.w1);
        block.ff.b1.copy_from_slice(&self.ff.b1);
        block.ff.w2.copy_from_slice(&self.ff.w2);
        block.ff.b2.copy_from_slice(&self.ff.b2);

        // Layer norms
        block.ln1.gamma.copy_from_slice(&self.ln1_gamma);
        block.ln1.beta.copy_from_slice(&self.ln1_beta);
        block.ln2.gamma.copy_from_slice(&self.ln2_gamma);
        block.ln2.beta.copy_from_slice(&self.ln2_beta);

        Ok(())
    }
}

/// Checkpoint manager for automatic saving
pub struct CheckpointManager {
    /// Directory for checkpoints
    checkpoint_dir: std::path::PathBuf,
    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,
    /// Save interval (steps)
    save_interval: u64,
    /// Last save step
    last_save_step: u64,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(checkpoint_dir: &Path, max_checkpoints: usize, save_interval: u64) -> Result<Self, String> {
        // Create directory if it doesn't exist
        fs::create_dir_all(checkpoint_dir)
            .map_err(|e| format!("Failed to create checkpoint directory: {}", e))?;

        Ok(Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            max_checkpoints,
            save_interval,
            last_save_step: 0,
        })
    }

    /// Check if we should save at this step
    pub fn should_save(&self, step: u64) -> bool {
        step > 0 && step - self.last_save_step >= self.save_interval
    }

    /// Save a checkpoint
    pub fn save(
        &mut self,
        model: &Transformer,
        step: u64,
        epoch: u32,
        loss_history: Vec<f32>,
        best_val_loss: Option<f32>,
    ) -> Result<std::path::PathBuf, String> {
        let checkpoint = Checkpoint::from_model(model, step, epoch, loss_history, best_val_loss);

        let filename = format!("checkpoint_step_{}.json", step);
        let path = self.checkpoint_dir.join(&filename);

        checkpoint.save(&path)?;
        self.last_save_step = step;

        // Clean up old checkpoints
        self.cleanup()?;

        Ok(path)
    }

    /// Get the latest checkpoint
    pub fn get_latest(&self) -> Result<Option<Checkpoint>, String> {
        let mut checkpoints: Vec<_> = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| format!("Failed to read checkpoint directory: {}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false))
            .collect();

        if checkpoints.is_empty() {
            return Ok(None);
        }

        // Sort by modification time (newest first)
        checkpoints.sort_by(|a, b| {
            b.metadata().and_then(|m| m.modified()).ok()
                .cmp(&a.metadata().and_then(|m| m.modified()).ok())
        });

        let latest_path = checkpoints[0].path();
        let checkpoint = Checkpoint::load(&latest_path)?;
        Ok(Some(checkpoint))
    }

    /// Clean up old checkpoints
    fn cleanup(&self) -> Result<(), String> {
        let mut checkpoints: Vec<_> = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| format!("Failed to read checkpoint directory: {}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false))
            .collect();

        if checkpoints.len() <= self.max_checkpoints {
            return Ok(());
        }

        // Sort by modification time (oldest first)
        checkpoints.sort_by(|a, b| {
            a.metadata().and_then(|m| m.modified()).ok()
                .cmp(&b.metadata().and_then(|m| m.modified()).ok())
        });

        // Remove oldest checkpoints
        let to_remove = checkpoints.len() - self.max_checkpoints;
        for entry in checkpoints.iter().take(to_remove) {
            fs::remove_file(entry.path())
                .map_err(|e| format!("Failed to remove old checkpoint: {}", e))?;
        }

        Ok(())
    }
}

// Simple bincode-like serialization using JSON + compression would be ideal,
// but for now we'll use JSON for simplicity
fn bincode_serialize<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value)
        .map_err(|e| format!("Serialization error: {}", e))
}

fn bincode_deserialize<T: for<'de> Deserialize<'de>>(data: &[u8]) -> Result<T, String> {
    serde_json::from_slice(data)
        .map_err(|e| format!("Deserialization error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig::tiny(100)
    }

    #[test]
    fn test_checkpoint_from_model() {
        let config = tiny_config();
        let model = Transformer::new(config).unwrap();

        let checkpoint = Checkpoint::from_model(&model, 100, 5, vec![1.0, 0.9, 0.8], Some(0.75));

        assert_eq!(checkpoint.step, 100);
        assert_eq!(checkpoint.epoch, 5);
        assert_eq!(checkpoint.loss_history, vec![1.0, 0.9, 0.8]);
        assert_eq!(checkpoint.best_val_loss, Some(0.75));
    }

    #[test]
    fn test_checkpoint_restore_model() {
        let config = tiny_config();
        let original = Transformer::new(config).unwrap();

        let checkpoint = Checkpoint::from_model(&original, 0, 0, vec![], None);
        let restored = checkpoint.restore_model().unwrap();

        // Check that restored model has same config
        assert_eq!(restored.config.vocab_size, original.config.vocab_size);
        assert_eq!(restored.config.d_model, original.config.d_model);

        // Check that weights match
        for (a, b) in original.embedding.token_embedding.weights.iter()
            .zip(restored.embedding.token_embedding.weights.iter())
        {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_checkpoint_save_load() {
        let config = tiny_config();
        let model = Transformer::new(config).unwrap();

        let checkpoint = Checkpoint::from_model(&model, 50, 2, vec![0.5], None);

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_checkpoint.json");

        checkpoint.save(&path).unwrap();
        let loaded = Checkpoint::load(&path).unwrap();

        assert_eq!(loaded.step, 50);
        assert_eq!(loaded.epoch, 2);
    }

    #[test]
    fn test_checkpoint_binary_roundtrip() {
        let config = tiny_config();
        let model = Transformer::new(config).unwrap();

        let checkpoint = Checkpoint::from_model(&model, 100, 5, vec![1.0], Some(0.5));

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_checkpoint.bin");

        checkpoint.save_binary(&path).unwrap();
        let loaded = Checkpoint::load_binary(&path).unwrap();

        assert_eq!(loaded.step, checkpoint.step);
        assert_eq!(loaded.best_val_loss, checkpoint.best_val_loss);
    }

    #[test]
    fn test_model_state_from_model() {
        let config = tiny_config();
        let model = Transformer::new(config.clone()).unwrap();

        let state = ModelState::from_model(&model);

        assert_eq!(state.token_embedding.len(), config.vocab_size * config.d_model);
        assert_eq!(state.blocks.len(), config.n_layers);
    }

    #[test]
    fn test_model_state_apply() {
        let config = tiny_config();
        let original = Transformer::new(config.clone()).unwrap();
        let mut target = Transformer::new(config).unwrap();

        let state = ModelState::from_model(&original);
        state.apply_to_model(&mut target).unwrap();

        // Weights should now match
        for (a, b) in original.embedding.token_embedding.weights.iter()
            .zip(target.embedding.token_embedding.weights.iter())
        {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_checkpoint_manager_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path(), 5, 100);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_checkpoint_manager_should_save() {
        let temp_dir = tempfile::tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path(), 5, 100).unwrap();

        assert!(!manager.should_save(0));
        assert!(!manager.should_save(50));
        assert!(manager.should_save(100));
        assert!(manager.should_save(200));
    }

    #[test]
    fn test_checkpoint_manager_save_and_load() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 5, 100).unwrap();

        let config = tiny_config();
        let model = Transformer::new(config).unwrap();

        let path = manager.save(&model, 100, 1, vec![1.0], None).unwrap();
        assert!(path.exists());

        let latest = manager.get_latest().unwrap();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().step, 100);
    }

    #[test]
    fn test_checkpoint_manager_cleanup() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path(), 2, 100).unwrap();

        let config = tiny_config();
        let model = Transformer::new(config).unwrap();

        // Save 3 checkpoints
        manager.save(&model, 100, 1, vec![], None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.save(&model, 200, 2, vec![], None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.save(&model, 300, 3, vec![], None).unwrap();

        // Should only have 2 checkpoints (max_checkpoints = 2)
        let count = fs::read_dir(temp_dir.path())
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension().map(|ext| ext == "json").unwrap_or(false))
            .count();

        assert!(count <= 2, "Should have cleaned up old checkpoints, got {}", count);
    }

    #[test]
    fn test_checkpoint_preserves_model_behavior() {
        let config = tiny_config();
        let mut original = Transformer::new(config).unwrap();

        let tokens = vec![1, 2, 3, 4];
        let original_output = original.forward(&tokens);

        // Create checkpoint and restore
        let checkpoint = Checkpoint::from_model(&original, 0, 0, vec![], None);
        let mut restored = checkpoint.restore_model().unwrap();

        let restored_output = restored.forward(&tokens);

        // Outputs should match
        for (a, b) in original_output.iter().zip(restored_output.iter()) {
            assert!((a - b).abs() < 1e-5, "Output mismatch: {} vs {}", a, b);
        }
    }
}
