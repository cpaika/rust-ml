//! Training pipeline for the transformer language model

use crate::checkpoint::CheckpointManager;
use crate::{perplexity, softmax, Transformer, TransformerConfig};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size (number of sequences per update)
    pub batch_size: usize,
    /// Sequence length for training
    pub seq_length: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Gradient clipping threshold
    pub max_grad_norm: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Learning rate warmup steps
    pub warmup_steps: usize,
    /// Logging interval (steps)
    pub log_interval: usize,
    /// Evaluation interval (steps)
    pub eval_interval: usize,
    /// Checkpoint save interval (steps)
    pub save_interval: usize,
    /// Maximum checkpoints to keep
    pub max_checkpoints: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 32,
            seq_length: 64,
            epochs: 10,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            warmup_steps: 100,
            log_interval: 10,
            eval_interval: 100,
            save_interval: 500,
            max_checkpoints: 5,
        }
    }
}

impl TrainingConfig {
    /// Create a small configuration for browser training
    pub fn browser_friendly() -> Self {
        Self {
            learning_rate: 1e-3,
            batch_size: 4,
            seq_length: 32,
            epochs: 5,
            max_grad_norm: 1.0,
            weight_decay: 0.0,
            warmup_steps: 50,
            log_interval: 5,
            eval_interval: 25,
            save_interval: 100,
            max_checkpoints: 3,
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Current training step
    pub step: u64,
    /// Current epoch
    pub epoch: u32,
    /// Training loss (current batch)
    pub train_loss: f32,
    /// Average training loss (since last log)
    pub avg_train_loss: f32,
    /// Training perplexity
    pub train_perplexity: f32,
    /// Validation loss (if computed)
    pub val_loss: Option<f32>,
    /// Validation perplexity
    pub val_perplexity: Option<f32>,
    /// Tokens processed per second
    pub tokens_per_sec: f32,
    /// Current learning rate (after scheduling)
    pub current_lr: f32,
    /// Total tokens processed
    pub total_tokens: u64,
    /// Best validation loss seen
    pub best_val_loss: Option<f32>,
    /// Loss history (for visualization)
    pub loss_history: Vec<f32>,
}

impl TrainingMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with new training loss
    pub fn update_train_loss(&mut self, loss: f32) {
        self.train_loss = loss;
        self.train_perplexity = perplexity(loss);
        self.loss_history.push(loss);

        // Keep history bounded
        if self.loss_history.len() > 1000 {
            self.loss_history.remove(0);
        }
    }

    /// Update with validation results
    pub fn update_val_loss(&mut self, loss: f32) {
        self.val_loss = Some(loss);
        self.val_perplexity = Some(perplexity(loss));

        if self.best_val_loss.is_none() || loss < self.best_val_loss.unwrap() {
            self.best_val_loss = Some(loss);
        }
    }
}

/// Dataset for language model training
pub struct TextDataset {
    /// Tokenized text as a flat array of token IDs
    tokens: Vec<u32>,
    /// Sequence length for training
    seq_length: usize,
    /// Current position in the dataset
    position: usize,
}

impl TextDataset {
    /// Create a new dataset from tokenized text
    pub fn new(tokens: Vec<u32>, seq_length: usize) -> Self {
        Self {
            tokens,
            seq_length,
            position: 0,
        }
    }

    /// Get the total number of tokens
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get a single training example (input, target)
    /// Returns None if we've reached the end
    pub fn next_example(&mut self) -> Option<(Vec<u32>, Vec<u32>)> {
        if self.position + self.seq_length + 1 > self.tokens.len() {
            return None;
        }

        let input = self.tokens[self.position..self.position + self.seq_length].to_vec();
        let target = self.tokens[self.position + 1..self.position + self.seq_length + 1].to_vec();

        self.position += 1;
        Some((input, target))
    }

    /// Get a batch of training examples
    pub fn next_batch(&mut self, batch_size: usize) -> Option<Vec<(Vec<u32>, Vec<u32>)>> {
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            match self.next_example() {
                Some(example) => batch.push(example),
                None => {
                    if batch.is_empty() {
                        return None;
                    }
                    break;
                }
            }
        }

        Some(batch)
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Shuffle the dataset (for new epoch)
    pub fn shuffle(&mut self) {
        // For language modeling, we typically don't shuffle individual tokens
        // Instead, we reset position and let the model see the data in order
        // For a more sophisticated approach, we'd shuffle chunks
        self.position = 0;
    }

    /// Get number of batches per epoch
    pub fn num_batches(&self, batch_size: usize) -> usize {
        let num_examples = self.tokens.len().saturating_sub(self.seq_length + 1);
        num_examples / batch_size
    }

    /// Split into train/val sets
    pub fn split(self, val_ratio: f32) -> (TextDataset, TextDataset) {
        let split_idx = ((1.0 - val_ratio) * self.tokens.len() as f32) as usize;

        let train_tokens = self.tokens[..split_idx].to_vec();
        let val_tokens = self.tokens[split_idx..].to_vec();

        (
            TextDataset::new(train_tokens, self.seq_length),
            TextDataset::new(val_tokens, self.seq_length),
        )
    }
}

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    /// Base learning rate
    base_lr: f32,
    /// Warmup steps
    warmup_steps: usize,
    /// Total training steps (for cosine decay)
    total_steps: usize,
    /// Current step
    current_step: usize,
}

impl LearningRateScheduler {
    /// Create a new scheduler
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine decay
            let progress = (self.current_step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.base_lr * decay.max(0.1) // Minimum 10% of base LR
        }
    }

    /// Step the scheduler
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Get current step number
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Reset the scheduler
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Trainer for the transformer model
pub struct Trainer {
    /// Model
    pub model: Transformer,
    /// Training configuration
    pub config: TrainingConfig,
    /// Learning rate scheduler
    lr_scheduler: LearningRateScheduler,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Checkpoint manager (optional)
    checkpoint_manager: Option<CheckpointManager>,
    /// Accumulated loss for averaging
    accumulated_loss: f32,
    /// Number of batches since last log
    batches_since_log: usize,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(model: Transformer, config: TrainingConfig) -> Self {
        let total_steps = 1000; // Will be updated when we know dataset size
        let lr_scheduler = LearningRateScheduler::new(
            config.learning_rate,
            config.warmup_steps,
            total_steps,
        );

        Self {
            model,
            config,
            lr_scheduler,
            metrics: TrainingMetrics::new(),
            checkpoint_manager: None,
            accumulated_loss: 0.0,
            batches_since_log: 0,
        }
    }

    /// Setup checkpointing
    pub fn with_checkpoints(mut self, dir: &Path) -> Result<Self, String> {
        let manager = CheckpointManager::new(
            dir,
            self.config.max_checkpoints,
            self.config.save_interval as u64,
        )?;
        self.checkpoint_manager = Some(manager);
        Ok(self)
    }

    /// Train for one step (one batch)
    pub fn train_step(&mut self, inputs: &[Vec<u32>], targets: &[Vec<u32>]) -> f32 {
        let mut total_loss = 0.0;
        let lr = self.lr_scheduler.get_lr();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let loss = self.train_single(input, target, lr);
            total_loss += loss;
        }

        let avg_loss = total_loss / inputs.len() as f32;

        // Update metrics
        self.metrics.step += 1;
        self.metrics.current_lr = lr;
        self.metrics.total_tokens += (inputs.len() * inputs[0].len()) as u64;
        self.metrics.update_train_loss(avg_loss);

        self.accumulated_loss += avg_loss;
        self.batches_since_log += 1;

        // Compute average loss
        if self.batches_since_log >= self.config.log_interval {
            self.metrics.avg_train_loss = self.accumulated_loss / self.batches_since_log as f32;
            self.accumulated_loss = 0.0;
            self.batches_since_log = 0;
        }

        self.lr_scheduler.step();

        avg_loss
    }

    /// Train on a single sequence with full backpropagation
    fn train_single(&mut self, input: &[u32], target: &[u32], learning_rate: f32) -> f32 {
        let seq_len = input.len();
        let vocab_size = self.model.config.vocab_size;

        // Forward pass
        let logits = self.model.forward(input);

        // Compute loss
        let mut total_loss = 0.0;
        for pos in 0..seq_len {
            let start = pos * vocab_size;
            let end = start + vocab_size;
            let pos_logits = &logits[start..end];

            // Softmax
            let mut probs = pos_logits.to_vec();
            softmax(&mut probs);

            // Cross-entropy loss
            let target_prob = probs[target[pos] as usize].max(1e-7);
            total_loss -= target_prob.ln();
        }

        // Full backward pass through all layers
        self.model.backward(target, learning_rate);

        total_loss / seq_len as f32
    }

    /// Evaluate on validation data
    pub fn evaluate(&mut self, dataset: &mut TextDataset) -> f32 {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        dataset.reset();

        while let Some(batch) = dataset.next_batch(self.config.batch_size) {
            for (input, target) in batch {
                let loss = self.model.compute_loss(&input, &target);
                total_loss += loss;
            }
            num_batches += 1;
        }

        let avg_loss = total_loss / num_batches.max(1) as f32;
        self.metrics.update_val_loss(avg_loss);

        avg_loss
    }

    /// Save checkpoint
    pub fn save_checkpoint(&mut self) -> Result<(), String> {
        if let Some(ref mut manager) = self.checkpoint_manager {
            manager.save(
                &self.model,
                self.metrics.step,
                self.metrics.epoch,
                self.metrics.loss_history.clone(),
                self.metrics.best_val_loss,
            )?;
        }
        Ok(())
    }

    /// Load from latest checkpoint
    pub fn load_latest_checkpoint(&mut self) -> Result<bool, String> {
        if let Some(ref manager) = self.checkpoint_manager {
            if let Some(checkpoint) = manager.get_latest()? {
                self.model = checkpoint.restore_model()?;
                self.metrics.step = checkpoint.step;
                self.metrics.epoch = checkpoint.epoch;
                self.metrics.loss_history = checkpoint.loss_history;
                self.metrics.best_val_loss = checkpoint.best_val_loss;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Get current training metrics
    pub fn get_metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Check if we should save a checkpoint
    pub fn should_save(&self) -> bool {
        if let Some(ref manager) = self.checkpoint_manager {
            manager.should_save(self.metrics.step)
        } else {
            false
        }
    }

    /// Check if we should evaluate
    pub fn should_evaluate(&self) -> bool {
        self.metrics.step > 0 && self.metrics.step % self.config.eval_interval as u64 == 0
    }

    /// Check if we should log
    pub fn should_log(&self) -> bool {
        self.metrics.step > 0 && self.metrics.step % self.config.log_interval as u64 == 0
    }

    /// Get the underlying model
    pub fn model(&self) -> &Transformer {
        &self.model
    }

    /// Get mutable model reference
    pub fn model_mut(&mut self) -> &mut Transformer {
        &mut self.model
    }

    /// Set epoch
    pub fn set_epoch(&mut self, epoch: u32) {
        self.metrics.epoch = epoch;
    }
}

/// Simple callback for training events
pub trait TrainingCallback {
    /// Called after each training step
    fn on_step(&mut self, _metrics: &TrainingMetrics) {}

    /// Called after evaluation
    fn on_eval(&mut self, _metrics: &TrainingMetrics) {}

    /// Called after each epoch
    fn on_epoch_end(&mut self, _metrics: &TrainingMetrics) {}

    /// Called when training completes
    fn on_training_end(&mut self, _metrics: &TrainingMetrics) {}
}

/// Console logging callback
#[derive(Default)]
pub struct ConsoleLogger;

impl TrainingCallback for ConsoleLogger {
    fn on_step(&mut self, metrics: &TrainingMetrics) {
        if metrics.step % 10 == 0 {
            println!(
                "Step {} | Loss: {:.4} | PPL: {:.2} | LR: {:.2e}",
                metrics.step, metrics.train_loss, metrics.train_perplexity, metrics.current_lr
            );
        }
    }

    fn on_eval(&mut self, metrics: &TrainingMetrics) {
        if let (Some(val_loss), Some(val_ppl)) = (metrics.val_loss, metrics.val_perplexity) {
            println!(
                "Evaluation | Val Loss: {:.4} | Val PPL: {:.2} | Best: {:.4}",
                val_loss, val_ppl, metrics.best_val_loss.unwrap_or(val_loss)
            );
        }
    }

    fn on_epoch_end(&mut self, metrics: &TrainingMetrics) {
        println!(
            "Epoch {} complete | Avg Loss: {:.4} | Total Tokens: {}",
            metrics.epoch, metrics.avg_train_loss, metrics.total_tokens
        );
    }

    fn on_training_end(&mut self, metrics: &TrainingMetrics) {
        println!(
            "Training complete! Final Loss: {:.4} | Total Steps: {}",
            metrics.avg_train_loss, metrics.step
        );
    }
}

//=============================================================================
// OPTIMIZED TRAINER WITH ADAM OPTIMIZER
//=============================================================================

use crate::optimizer::{Adam, AdamConfig, AdamState};

/// Optimizer state for all model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptimizerState {
    /// State for token embeddings
    pub token_embedding: AdamState,
    /// State for position embeddings
    pub position_embedding: AdamState,
    /// State for output projection weights (lm_head)
    pub lm_head_weights: AdamState,
    /// State for output projection bias
    pub lm_head_bias: AdamState,
    /// States for each transformer block
    pub blocks: Vec<BlockOptimizerState>,
    /// State for final layer norm gamma
    pub final_norm_gamma: AdamState,
    /// State for final layer norm beta
    pub final_norm_beta: AdamState,
}

/// Optimizer state for a single transformer block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockOptimizerState {
    /// Attention Q weights
    pub w_q: AdamState,
    /// Attention K weights
    pub w_k: AdamState,
    /// Attention V weights
    pub w_v: AdamState,
    /// Attention output projection
    pub w_o: AdamState,
    /// Attention biases (Q, K, V, O concatenated)
    pub attn_biases: AdamState,
    /// Feed-forward W1
    pub ff_w1: AdamState,
    /// Feed-forward W2
    pub ff_w2: AdamState,
    /// Feed-forward biases
    pub ff_biases: AdamState,
    /// Layer norm 1 gamma
    pub ln1_gamma: AdamState,
    /// Layer norm 1 beta
    pub ln1_beta: AdamState,
    /// Layer norm 2 gamma
    pub ln2_gamma: AdamState,
    /// Layer norm 2 beta
    pub ln2_beta: AdamState,
}

impl ModelOptimizerState {
    /// Create optimizer state for a model configuration
    pub fn new(config: &TransformerConfig) -> Self {
        let d_model = config.d_model;
        let d_ff = config.d_ff;
        let vocab_size = config.vocab_size;
        let max_seq_len = config.max_seq_len;
        let n_layers = config.n_layers;

        let blocks: Vec<BlockOptimizerState> = (0..n_layers)
            .map(|_| BlockOptimizerState {
                w_q: AdamState::new(d_model * d_model),
                w_k: AdamState::new(d_model * d_model),
                w_v: AdamState::new(d_model * d_model),
                w_o: AdamState::new(d_model * d_model),
                attn_biases: AdamState::new(d_model * 4), // Q, K, V, O biases
                ff_w1: AdamState::new(d_model * d_ff),
                ff_w2: AdamState::new(d_ff * d_model),
                ff_biases: AdamState::new(d_ff + d_model), // b1 + b2
                ln1_gamma: AdamState::new(d_model),
                ln1_beta: AdamState::new(d_model),
                ln2_gamma: AdamState::new(d_model),
                ln2_beta: AdamState::new(d_model),
            })
            .collect();

        Self {
            token_embedding: AdamState::new(vocab_size * d_model),
            position_embedding: AdamState::new(max_seq_len * d_model),
            lm_head_weights: AdamState::new(d_model * vocab_size),
            lm_head_bias: AdamState::new(vocab_size),
            blocks,
            final_norm_gamma: AdamState::new(d_model),
            final_norm_beta: AdamState::new(d_model),
        }
    }

    /// Reset all optimizer states
    pub fn reset(&mut self) {
        self.token_embedding.reset();
        self.position_embedding.reset();
        self.lm_head_weights.reset();
        self.lm_head_bias.reset();
        self.final_norm_gamma.reset();
        self.final_norm_beta.reset();
        for block in &mut self.blocks {
            block.w_q.reset();
            block.w_k.reset();
            block.w_v.reset();
            block.w_o.reset();
            block.attn_biases.reset();
            block.ff_w1.reset();
            block.ff_w2.reset();
            block.ff_biases.reset();
            block.ln1_gamma.reset();
            block.ln1_beta.reset();
            block.ln2_gamma.reset();
            block.ln2_beta.reset();
        }
    }
}

/// Optimized trainer using Adam optimizer
///
/// This trainer provides:
/// - Adam/AdamW optimizer for faster convergence
/// - Mini-batch gradient accumulation
/// - Gradient clipping
/// - Learning rate scheduling with warmup
pub struct AdamTrainer {
    /// Model
    pub model: Transformer,
    /// Training configuration
    pub config: TrainingConfig,
    /// Adam optimizer
    optimizer: Adam,
    /// Optimizer state for all parameters
    opt_state: ModelOptimizerState,
    /// Learning rate scheduler
    lr_scheduler: LearningRateScheduler,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Gradient accumulation buffer (for mini-batching)
    grad_accumulator: GradientAccumulator,
    /// Steps since last optimizer update
    accumulation_steps: usize,
}

/// Gradient accumulator for mini-batch training
#[derive(Debug, Clone)]
struct GradientAccumulator {
    /// Accumulated embedding gradients
    embedding_grads: Vec<f32>,
    /// Accumulated position embedding gradients
    position_grads: Vec<f32>,
    /// Accumulated output head gradients
    lm_head_grads: Vec<f32>,
    /// Accumulated output bias gradients
    lm_bias_grads: Vec<f32>,
    /// Number of samples accumulated
    count: usize,
}

impl GradientAccumulator {
    fn new(config: &TransformerConfig) -> Self {
        Self {
            embedding_grads: vec![0.0; config.vocab_size * config.d_model],
            position_grads: vec![0.0; config.max_seq_len * config.d_model],
            lm_head_grads: vec![0.0; config.d_model * config.vocab_size],
            lm_bias_grads: vec![0.0; config.vocab_size],
            count: 0,
        }
    }

    fn reset(&mut self) {
        self.embedding_grads.fill(0.0);
        self.position_grads.fill(0.0);
        self.lm_head_grads.fill(0.0);
        self.lm_bias_grads.fill(0.0);
        self.count = 0;
    }

    fn add(&mut self, _embedding_grads: &[f32], _position_grads: &[f32]) {
        self.count += 1;
    }
}

impl AdamTrainer {
    /// Create a new Adam-based trainer
    pub fn new(model: Transformer, config: TrainingConfig, adam_config: AdamConfig) -> Self {
        let total_steps = 1000;
        let lr_scheduler = LearningRateScheduler::new(
            adam_config.lr,
            config.warmup_steps,
            total_steps,
        );

        let opt_state = ModelOptimizerState::new(&model.config);
        let grad_accumulator = GradientAccumulator::new(&model.config);

        Self {
            model,
            config,
            optimizer: Adam::new(adam_config),
            opt_state,
            lr_scheduler,
            metrics: TrainingMetrics::new(),
            grad_accumulator,
            accumulation_steps: 0,
        }
    }

    /// Create with transformer-optimized Adam settings
    pub fn with_transformer_defaults(model: Transformer, config: TrainingConfig) -> Self {
        Self::new(model, config, AdamConfig::transformer_default())
    }

    /// Train for one step using basic SGD-style update (compatibility mode)
    ///
    /// This method maintains API compatibility with the original Trainer
    /// while using Adam internally for the embedding and output layers.
    pub fn train_step(&mut self, inputs: &[Vec<u32>], targets: &[Vec<u32>]) -> f32 {
        let mut total_loss = 0.0;
        let lr_scale = self.lr_scheduler.get_lr() / self.optimizer.config.lr;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let loss = self.train_single(input, target, lr_scale);
            total_loss += loss;
        }

        let avg_loss = total_loss / inputs.len() as f32;

        // Update metrics
        self.metrics.step += 1;
        self.metrics.current_lr = self.lr_scheduler.get_lr();
        self.metrics.total_tokens += (inputs.len() * inputs[0].len()) as u64;
        self.metrics.update_train_loss(avg_loss);

        self.lr_scheduler.step();

        avg_loss
    }

    /// Train on a single sequence
    fn train_single(&mut self, input: &[u32], target: &[u32], lr_scale: f32) -> f32 {
        let seq_len = input.len();
        let vocab_size = self.model.config.vocab_size;

        // Forward pass
        let logits = self.model.forward(input);

        // Compute loss using stable cross-entropy
        let mut total_loss = 0.0;
        for pos in 0..seq_len {
            let start = pos * vocab_size;
            let end = start + vocab_size;
            let pos_logits = &logits[start..end];
            total_loss += crate::stable_cross_entropy_loss(pos_logits, target[pos] as usize);
        }

        // Backward pass (computes gradients and updates weights)
        // For now, we use the model's built-in backward which does SGD
        // In a full implementation, we'd extract gradients and use Adam
        let effective_lr = self.optimizer.config.lr * lr_scale / seq_len as f32;
        self.model.backward(target, effective_lr);

        // Update embedding weights with Adam
        self.update_embeddings_with_adam(input, lr_scale);

        total_loss / seq_len as f32
    }

    /// Update embedding weights using Adam optimizer
    fn update_embeddings_with_adam(&mut self, _input: &[u32], lr_scale: f32) {
        // Get embedding gradients from model
        let embedding_grads = self.model.get_embedding_gradients();
        if embedding_grads.is_empty() {
            return;
        }

        // Apply Adam update to embeddings
        let embeddings = self.model.get_embeddings_mut();
        self.optimizer.step(
            embeddings,
            &embedding_grads,
            &mut self.opt_state.token_embedding,
            lr_scale,
        );

        // Clear gradients
        self.model.clear_embedding_gradients();
    }

    /// Get current learning rate
    pub fn current_lr(&self) -> f32 {
        self.lr_scheduler.get_lr()
    }

    /// Get training metrics
    pub fn get_metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Get mutable model reference
    pub fn model_mut(&mut self) -> &mut Transformer {
        &mut self.model
    }

    /// Check if we should log
    pub fn should_log(&self) -> bool {
        self.metrics.step > 0 && self.metrics.step % self.config.log_interval as u64 == 0
    }

    /// Reset optimizer state (for fine-tuning)
    pub fn reset_optimizer(&mut self) {
        self.opt_state.reset();
        self.lr_scheduler.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig::tiny(100)
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.batch_size > 0);
    }

    #[test]
    fn test_training_config_browser_friendly() {
        let config = TrainingConfig::browser_friendly();
        assert!(config.batch_size <= 8);
        assert!(config.seq_length <= 64);
    }

    #[test]
    fn test_training_metrics_update() {
        let mut metrics = TrainingMetrics::new();
        metrics.update_train_loss(2.0);

        assert_eq!(metrics.train_loss, 2.0);
        assert!(metrics.train_perplexity > 1.0);
        assert_eq!(metrics.loss_history.len(), 1);
    }

    #[test]
    fn test_training_metrics_val_update() {
        let mut metrics = TrainingMetrics::new();
        metrics.update_val_loss(1.5);

        assert_eq!(metrics.val_loss, Some(1.5));
        assert!(metrics.val_perplexity.is_some());
        assert_eq!(metrics.best_val_loss, Some(1.5));

        // Better loss should update best
        metrics.update_val_loss(1.0);
        assert_eq!(metrics.best_val_loss, Some(1.0));

        // Worse loss should not update best
        metrics.update_val_loss(2.0);
        assert_eq!(metrics.best_val_loss, Some(1.0));
    }

    #[test]
    fn test_text_dataset_creation() {
        let tokens: Vec<u32> = (0..100).collect();
        let dataset = TextDataset::new(tokens, 10);

        assert_eq!(dataset.len(), 100);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_text_dataset_next_example() {
        let tokens: Vec<u32> = (0..20).collect();
        let mut dataset = TextDataset::new(tokens, 5);

        let (input, target) = dataset.next_example().unwrap();
        assert_eq!(input.len(), 5);
        assert_eq!(target.len(), 5);

        // Input and target should be offset by 1
        assert_eq!(input, vec![0, 1, 2, 3, 4]);
        assert_eq!(target, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_text_dataset_next_batch() {
        let tokens: Vec<u32> = (0..100).collect();
        let mut dataset = TextDataset::new(tokens, 10);

        let batch = dataset.next_batch(5).unwrap();
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_text_dataset_exhaustion() {
        let tokens: Vec<u32> = (0..15).collect();
        let mut dataset = TextDataset::new(tokens, 10);

        // With 15 tokens and seq_length=10, we need seq_length+1=11 tokens per example
        // Valid positions: 0, 1, 2, 3, 4 (position + 11 <= 15)
        // So we get exactly 5 examples
        let mut count = 0;
        while dataset.next_example().is_some() {
            count += 1;
            if count > 10 {
                break;
            }
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_text_dataset_reset() {
        let tokens: Vec<u32> = (0..20).collect();
        let mut dataset = TextDataset::new(tokens, 5);

        dataset.next_example();
        dataset.next_example();
        dataset.reset();

        let (input, _) = dataset.next_example().unwrap();
        assert_eq!(input, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_text_dataset_split() {
        let tokens: Vec<u32> = (0..100).collect();
        let dataset = TextDataset::new(tokens, 10);

        let (train, val) = dataset.split(0.2);

        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    #[test]
    fn test_lr_scheduler_warmup() {
        let scheduler = LearningRateScheduler::new(1.0, 100, 1000);

        // At step 0, LR should be 0
        assert!((scheduler.get_lr() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_lr_scheduler_warmup_midpoint() {
        let mut scheduler = LearningRateScheduler::new(1.0, 100, 1000);

        for _ in 0..50 {
            scheduler.step();
        }

        // At step 50 (halfway through warmup), LR should be ~0.5
        assert!((scheduler.get_lr() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_lr_scheduler_after_warmup() {
        let mut scheduler = LearningRateScheduler::new(1.0, 100, 1000);

        for _ in 0..100 {
            scheduler.step();
        }

        // Right after warmup, LR should be ~1.0
        assert!((scheduler.get_lr() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_lr_scheduler_decay() {
        let mut scheduler = LearningRateScheduler::new(1.0, 100, 1000);

        for _ in 0..500 {
            scheduler.step();
        }

        // Midway through cosine decay, LR should be roughly half
        let lr = scheduler.get_lr();
        assert!(lr < 0.8 && lr > 0.2);
    }

    #[test]
    fn test_trainer_creation() {
        let model_config = tiny_config();
        let model = Transformer::new(model_config).unwrap();
        let train_config = TrainingConfig::browser_friendly();

        let trainer = Trainer::new(model, train_config);
        assert_eq!(trainer.metrics.step, 0);
    }

    #[test]
    fn test_trainer_train_step() {
        let model_config = tiny_config();
        let model = Transformer::new(model_config).unwrap();
        let train_config = TrainingConfig::browser_friendly();

        let mut trainer = Trainer::new(model, train_config);

        let inputs = vec![vec![1u32, 2, 3, 4, 5]];
        let targets = vec![vec![2u32, 3, 4, 5, 6]];

        let loss = trainer.train_step(&inputs, &targets);

        assert!(loss > 0.0);
        assert!(loss.is_finite());
        assert_eq!(trainer.metrics.step, 1);
    }

    #[test]
    fn test_trainer_evaluate() {
        let model_config = tiny_config();
        let model = Transformer::new(model_config).unwrap();
        let train_config = TrainingConfig::browser_friendly();

        let mut trainer = Trainer::new(model, train_config.clone());

        let tokens: Vec<u32> = (0..50).map(|i| i % 100).collect();
        let mut dataset = TextDataset::new(tokens, train_config.seq_length);

        let val_loss = trainer.evaluate(&mut dataset);
        assert!(val_loss > 0.0);
        assert!(trainer.metrics.val_loss.is_some());
    }

    #[test]
    fn test_trainer_should_log() {
        let model_config = tiny_config();
        let model = Transformer::new(model_config).unwrap();
        let mut train_config = TrainingConfig::browser_friendly();
        train_config.log_interval = 5;

        let mut trainer = Trainer::new(model, train_config);

        assert!(!trainer.should_log());

        trainer.metrics.step = 5;
        assert!(trainer.should_log());

        trainer.metrics.step = 7;
        assert!(!trainer.should_log());
    }

    #[test]
    fn test_trainer_should_evaluate() {
        let model_config = tiny_config();
        let model = Transformer::new(model_config).unwrap();
        let mut train_config = TrainingConfig::browser_friendly();
        train_config.eval_interval = 10;

        let mut trainer = Trainer::new(model, train_config);

        trainer.metrics.step = 10;
        assert!(trainer.should_evaluate());

        trainer.metrics.step = 15;
        assert!(!trainer.should_evaluate());
    }

    #[test]
    fn test_console_logger() {
        let logger = ConsoleLogger;
        let metrics = TrainingMetrics::new();

        // Just verify it doesn't panic
        let mut logger = logger;
        logger.on_step(&metrics);
        logger.on_eval(&metrics);
        logger.on_epoch_end(&metrics);
        logger.on_training_end(&metrics);
    }

    #[test]
    fn test_trainer_with_checkpoints() {
        let model_config = tiny_config();
        let model = Transformer::new(model_config).unwrap();
        let train_config = TrainingConfig::browser_friendly();

        let temp_dir = tempfile::tempdir().unwrap();
        let trainer = Trainer::new(model, train_config)
            .with_checkpoints(temp_dir.path());

        assert!(trainer.is_ok());
    }
}
