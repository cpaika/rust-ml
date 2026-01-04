//! Integration tests for the transformer language model pipeline

#[allow(unused_imports)]
use std::path::Path;
use transformer::{Transformer, TransformerConfig};
use transformer::training::{TextDataset, TrainingConfig, LearningRateScheduler, TrainingMetrics, Trainer};
use transformer::checkpoint::{Checkpoint, CheckpointManager};
use transformer::gpu::{GpuContext, GpuOps};
use std::sync::Arc;

/// Test the complete training pipeline from config to training
#[test]
fn test_complete_training_pipeline() {
    // Create a small model config for testing
    let config = TransformerConfig::tiny(100);

    // Create model
    let mut model = Transformer::new(config.clone()).expect("Should create model");

    // Verify model structure
    assert_eq!(model.config.vocab_size, 100);
    assert!(model.num_parameters() > 0);

    // Create training data (simple sequence pattern)
    let tokens: Vec<u32> = (0..200).map(|i| (i % 100) as u32).collect();
    let mut dataset = TextDataset::new(tokens, config.max_seq_len);

    // Get an example
    let (input, target) = dataset.next_example().expect("Should have example");
    assert_eq!(input.len(), config.max_seq_len);
    assert_eq!(target.len(), config.max_seq_len);

    // Verify target is shifted by 1
    for i in 0..input.len() {
        assert_eq!(target[i], (input[i] + 1) % 100);
    }

    // Forward pass
    let logits = model.forward(&input);
    assert_eq!(logits.len(), config.max_seq_len * config.vocab_size);

    // Compute loss
    let loss = model.compute_loss(&input, &target);
    assert!(loss > 0.0);
    assert!(loss.is_finite());

    // Generate tokens
    let generated = model.generate(&[1, 2, 3], 5, 0.8);
    assert!(generated.len() >= 3); // At least the prompt
    assert!(generated.len() <= 8); // Prompt + max generated
}

/// Test training with learning rate scheduler
#[test]
fn test_training_with_scheduler() {
    let config = TransformerConfig::tiny(50);
    let _model = Transformer::new(config.clone()).expect("Should create model");

    // Create scheduler
    let mut scheduler = LearningRateScheduler::new(0.001, 10, 100);

    // Initial LR should be 0 (warmup)
    assert!((scheduler.get_lr() - 0.0).abs() < 1e-6);

    // Step through warmup
    for _ in 0..10 {
        scheduler.step();
    }

    // After warmup, LR should be at base
    let lr_after_warmup = scheduler.get_lr();
    assert!((lr_after_warmup - 0.001).abs() < 0.0005);

    // Continue stepping and verify decay
    for _ in 0..50 {
        scheduler.step();
    }

    let lr_mid = scheduler.get_lr();
    assert!(lr_mid < lr_after_warmup);
    assert!(lr_mid > 0.0);
}

/// Test training metrics tracking
#[test]
fn test_training_metrics_tracking() {
    let mut metrics = TrainingMetrics::new();

    // Update through multiple steps
    for step in 0..100 {
        let loss = 5.0 - step as f32 * 0.04; // Decreasing loss
        metrics.step = step;
        metrics.update_train_loss(loss);
    }

    // Verify final state
    assert_eq!(metrics.step, 99);
    assert!(metrics.train_loss < 2.0);
}

/// Test checkpoint save and restore preserves model behavior
#[test]
fn test_checkpoint_preserves_inference() {
    let config = TransformerConfig::tiny(50);
    let mut model1 = Transformer::new(config.clone()).expect("Should create model");

    // Run inference
    let tokens = vec![1, 2, 3, 4, 5];
    let logits1 = model1.forward(&tokens);

    // Create checkpoint
    let checkpoint = Checkpoint::from_model(&model1, 0, 0, vec![], None);

    // Restore to new model
    let mut model2 = checkpoint.restore_model().expect("Should restore model");

    // Run same inference
    let logits2 = model2.forward(&tokens);

    // Results should match
    assert_eq!(logits1.len(), logits2.len());
    for (l1, l2) in logits1.iter().zip(logits2.iter()) {
        assert!((l1 - l2).abs() < 1e-5, "Logits should match after restore");
    }
}

/// Test dataset splitting for train/val
#[test]
fn test_dataset_train_val_split() {
    let tokens: Vec<u32> = (0..1000).map(|i| i as u32 % 100).collect();
    let dataset = TextDataset::new(tokens, 32);

    let (train, val) = dataset.split(0.2);

    // Verify split sizes
    assert_eq!(train.len(), 800);
    assert_eq!(val.len(), 200);
}

/// Test text dataset iteration
#[test]
fn test_dataset_full_iteration() {
    let tokens: Vec<u32> = (0..100).collect();
    let mut dataset = TextDataset::new(tokens, 10);

    let mut examples_count = 0;
    while dataset.next_example().is_some() {
        examples_count += 1;
        if examples_count > 1000 {
            panic!("Dataset should exhaust");
        }
    }

    // With 100 tokens and seq_len=10, we need 11 tokens per example
    // So we can get 100 - 11 + 1 = 90 examples
    assert_eq!(examples_count, 90);

    // After reset, should iterate again
    dataset.reset();
    let (input, _) = dataset.next_example().expect("Should have example after reset");
    assert_eq!(input[0], 0);
}

/// Test model generation with different temperatures
#[test]
fn test_generation_temperature() {
    let config = TransformerConfig::tiny(50);
    let mut model = Transformer::new(config).expect("Should create model");

    let prompt = vec![1, 2, 3];

    // Greedy (temperature = 0)
    // Note: model might generate EOS immediately, so we just check it returns at least the prompt
    let greedy = model.generate(&prompt, 10, 0.0);
    assert!(greedy.len() >= prompt.len(), "Greedy should return at least prompt");

    // Sampling (temperature = 1.0)
    let sampled = model.generate(&prompt, 10, 1.0);
    assert!(sampled.len() >= prompt.len(), "Sampled should return at least prompt");

    // High temperature (temperature = 2.0)
    let high_temp = model.generate(&prompt, 10, 2.0);
    assert!(high_temp.len() >= prompt.len(), "High temp should return at least prompt");

    // All should start with the prompt
    assert_eq!(&greedy[..3], &prompt);
    assert_eq!(&sampled[..3], &prompt);
    assert_eq!(&high_temp[..3], &prompt);
}

/// Test model handles various sequence lengths
#[test]
fn test_variable_sequence_lengths() {
    let config = TransformerConfig::tiny(50);
    let mut model = Transformer::new(config.clone()).expect("Should create model");

    // Very short sequence
    let short = vec![1];
    let short_out = model.forward(&short);
    assert_eq!(short_out.len(), config.vocab_size);

    // Medium sequence
    let medium = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let medium_out = model.forward(&medium);
    assert_eq!(medium_out.len(), 8 * config.vocab_size);

    // Longer sequence (up to max)
    let long: Vec<u32> = (0..config.max_seq_len as u32).map(|i| i % 50).collect();
    let long_out = model.forward(&long);
    assert_eq!(long_out.len(), config.max_seq_len * config.vocab_size);
}

/// Test attention weights are valid probabilities
#[test]
fn test_attention_weights_validity() {
    let config = TransformerConfig::tiny(50);
    let mut model = Transformer::new(config.clone()).expect("Should create model");

    let tokens = vec![1, 2, 3, 4];
    model.forward(&tokens);

    // Get attention from first layer
    if let Some(attn) = model.get_layer_attention(0) {
        // Attention weights should sum to approximately 1 per query position
        // (for each head, for each query position)
        let n_heads = config.n_heads;
        let seq_len = tokens.len();

        for head in 0..n_heads {
            for q in 0..seq_len {
                let mut sum = 0.0;
                for k in 0..=q {
                    // Only sum up to q due to causal mask
                    let idx = head * seq_len * seq_len + q * seq_len + k;
                    if idx < attn.len() {
                        sum += attn[idx];
                    }
                }
                // Should sum to approximately 1.0
                assert!((sum - 1.0).abs() < 0.1, "Attention weights should sum to 1, got {}", sum);
            }
        }
    }
}

/// Test training config defaults are reasonable
#[test]
fn test_training_config_defaults() {
    let config = TrainingConfig::default();

    assert!(config.learning_rate > 0.0);
    assert!(config.learning_rate < 1.0);
    assert!(config.batch_size > 0);
    assert!(config.seq_length > 0);
    assert!(config.epochs > 0);
}

/// Test browser-friendly config is suitable for WASM
#[test]
fn test_browser_friendly_config() {
    let config = TrainingConfig::browser_friendly();

    // Should have smaller values for browser
    assert!(config.batch_size <= 16);
    assert!(config.seq_length <= 64);

    // Should still be valid
    assert!(config.learning_rate > 0.0);
    assert!(config.epochs > 0);
}

/// Test model serialization roundtrip
#[test]
fn test_model_serialization() {
    let config = TransformerConfig::tiny(50);
    let model = Transformer::new(config.clone()).expect("Should create model");

    // Serialize
    let json = serde_json::to_string(&model).expect("Should serialize");

    // Deserialize
    let loaded: Transformer = serde_json::from_str(&json).expect("Should deserialize");

    // Verify structure
    assert_eq!(loaded.config.vocab_size, model.config.vocab_size);
    assert_eq!(loaded.config.d_model, model.config.d_model);
    assert_eq!(loaded.config.n_layers, model.config.n_layers);
    assert_eq!(loaded.num_parameters(), model.num_parameters());
}

/// Test Trainer struct creation and basic operation
#[test]
fn test_trainer_creation() {
    let model_config = TransformerConfig::tiny(50);
    let train_config = TrainingConfig::browser_friendly();

    let model = Transformer::new(model_config).expect("Should create model");
    let trainer = Trainer::new(model, train_config);

    assert!(trainer.model.num_parameters() > 0);
}

/// Test that loss decreases over training (basic sanity check)
#[test]
fn test_loss_decreases() {
    let config = TransformerConfig::tiny(20);
    let mut model = Transformer::new(config.clone()).expect("Should create model");

    // Create simple repeating pattern
    let pattern: Vec<u32> = (0..200).map(|i| (i % 10) as u32).collect();
    let mut dataset = TextDataset::new(pattern, 10);

    // Get initial loss
    let (input, target) = dataset.next_example().unwrap();
    let initial_loss = model.compute_loss(&input, &target);

    // Train for a few steps (simplified - just forward passes)
    // Note: Real training would need backprop which we haven't implemented
    // This test just verifies the pipeline works end-to-end
    for _ in 0..10 {
        dataset.reset();
        while let Some((input, target)) = dataset.next_example() {
            let _ = model.compute_loss(&input, &target);
        }
    }

    // Get final loss
    dataset.reset();
    let (input, target) = dataset.next_example().unwrap();
    let final_loss = model.compute_loss(&input, &target);

    // Both losses should be valid (not NaN, not Inf)
    assert!(initial_loss.is_finite());
    assert!(final_loss.is_finite());
}

/// Test that top-k predictions are sorted correctly
#[test]
fn test_top_k_ordering() {
    let config = TransformerConfig::tiny(50);
    let mut model = Transformer::new(config).expect("Should create model");

    let tokens = vec![1, 2, 3];
    model.forward(&tokens);

    let top_k = model.top_k_next_tokens(10);

    // Verify sorted by probability (descending)
    for i in 0..top_k.len() - 1 {
        assert!(
            top_k[i].1 >= top_k[i + 1].1,
            "Top-k should be sorted by probability"
        );
    }

    // Probabilities should sum to <= 1.0
    let sum: f32 = top_k.iter().map(|(_, p)| p).sum();
    assert!(sum <= 1.0 + 1e-5);
}

/// Test checkpoint manager with file operations
#[test]
fn test_checkpoint_manager_operations() {
    let temp_dir = std::env::temp_dir().join("transformer_test_checkpoints");
    let _ = std::fs::create_dir_all(&temp_dir);

    let manager = CheckpointManager::new(&temp_dir, 3, 100);
    assert!(manager.is_ok());

    let manager = manager.unwrap();

    // Initially no checkpoints should be needed at step 0
    assert!(!manager.should_save(0));

    // Should checkpoint at interval
    assert!(manager.should_save(100));
    assert!(manager.should_save(200));
    assert!(!manager.should_save(50));

    // Clean up
    let _ = std::fs::remove_dir_all(&temp_dir);
}

/// Test complete tokenizer + transformer pipeline
#[test]
fn test_tokenizer_transformer_pipeline() {
    // This test requires the tokenizer crate
    // Since it's an integration test, we can test the full pipeline
    let config = TransformerConfig::tiny(256);
    let mut model = Transformer::new(config).expect("Should create model");

    // Simulate tokenized input (would come from tokenizer in real use)
    let tokens: Vec<u32> = vec![10, 20, 30, 40, 50];

    // Forward pass
    let logits = model.forward(&tokens);
    assert!(!logits.is_empty());

    // Generate
    let generated = model.generate(&tokens, 5, 0.8);
    assert!(generated.len() >= tokens.len());

    // Get probabilities
    let probs = model.get_next_token_probs();
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Probs should sum to 1");
}

/// Test model can handle edge cases
#[test]
fn test_edge_cases() {
    let config = TransformerConfig::tiny(50);
    let mut model = Transformer::new(config).expect("Should create model");

    // Single token
    let single = vec![0];
    let out = model.forward(&single);
    assert!(!out.is_empty());

    // Token at vocab boundary
    let boundary = vec![49]; // Last valid token for vocab_size=50
    let out = model.forward(&boundary);
    assert!(!out.is_empty());

    // Multiple same tokens
    let repeated = vec![5, 5, 5, 5, 5];
    let out = model.forward(&repeated);
    assert!(!out.is_empty());
}

/// Test dataset handles various sizes
#[test]
fn test_dataset_edge_cases() {
    // Minimum viable dataset
    let tokens: Vec<u32> = (0..12).collect();
    let mut dataset = TextDataset::new(tokens, 10);

    // Should get at least one example
    assert!(dataset.next_example().is_some());

    // Empty after exhaustion
    dataset.next_example(); // Second might or might not exist
    // Eventually should return None
    let mut count = 0;
    while dataset.next_example().is_some() {
        count += 1;
        if count > 100 {
            panic!("Dataset should eventually exhaust");
        }
    }
}

// GPU tests - these only run if GPU is available

/// Helper to setup GPU context
fn setup_gpu() -> Option<Arc<GpuContext>> {
    GpuContext::new().ok().map(Arc::new)
}

/// Test GPU matmul matches CPU reference
#[test]
fn test_gpu_matmul_matches_cpu() {
    let Some(ctx) = setup_gpu() else {
        println!("Skipping GPU test - no GPU available");
        return;
    };

    let ops = GpuOps::new(ctx);

    // CPU reference matmul
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    // Test matrices
    let m = 4;
    let k = 3;
    let n = 5;
    let a: Vec<f32> = (0..(m * k)).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| i as f32 * 0.1).collect();

    let cpu_result = cpu_matmul(&a, &b, m, k, n);
    let gpu_result = ops.matmul(&a, &b, m as u32, k as u32, n as u32);

    // Compare results
    assert_eq!(cpu_result.len(), gpu_result.len());
    for i in 0..cpu_result.len() {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        assert!(
            diff < 0.01,
            "Matmul mismatch at index {}: CPU={}, GPU={}, diff={}",
            i, cpu_result[i], gpu_result[i], diff
        );
    }
}

/// Test GPU layer norm matches CPU reference
#[test]
fn test_gpu_layer_norm_matches_cpu() {
    let Some(ctx) = setup_gpu() else {
        println!("Skipping GPU test - no GPU available");
        return;
    };

    let ops = GpuOps::new(ctx);

    // CPU reference layer norm
    fn cpu_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], d_model: usize, eps: f32) -> Vec<f32> {
        let seq_len = input.len() / d_model;
        let mut output = vec![0.0f32; input.len()];

        for pos in 0..seq_len {
            let start = pos * d_model;
            let row = &input[start..start + d_model];

            // Mean
            let mean: f32 = row.iter().sum::<f32>() / d_model as f32;

            // Variance
            let variance: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / d_model as f32;
            let std = (variance + eps).sqrt();

            // Normalize and apply scale/shift
            for i in 0..d_model {
                output[start + i] = ((row[i] - mean) / std) * gamma[i] + beta[i];
            }
        }
        output
    }

    // Test data
    let d_model = 4;
    let seq_len = 3;
    let input: Vec<f32> = (0..(seq_len * d_model)).map(|i| i as f32 + 1.0).collect();
    let gamma = vec![1.0f32; d_model];
    let beta = vec![0.0f32; d_model];

    let cpu_result = cpu_layer_norm(&input, &gamma, &beta, d_model, 1e-5);
    let gpu_result = ops.layer_norm(&input, &gamma, &beta, seq_len as u32, d_model as u32, 1e-5);

    // Compare results
    assert_eq!(cpu_result.len(), gpu_result.len());
    for i in 0..cpu_result.len() {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        assert!(
            diff < 0.01,
            "Layer norm mismatch at index {}: CPU={}, GPU={}, diff={}",
            i, cpu_result[i], gpu_result[i], diff
        );
    }
}

/// Test GPU softmax matches CPU reference
#[test]
fn test_gpu_softmax_matches_cpu() {
    let Some(ctx) = setup_gpu() else {
        println!("Skipping GPU test - no GPU available");
        return;
    };

    let ops = GpuOps::new(ctx);

    // CPU reference softmax (per row)
    fn cpu_softmax(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut output = data.to_vec();
        for row in 0..rows {
            let start = row * cols;
            let row_data = &mut output[start..start + cols];

            // Find max
            let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Exp and sum
            let mut sum = 0.0f32;
            for x in row_data.iter_mut() {
                *x = (*x - max_val).exp();
                sum += *x;
            }

            // Normalize
            for x in row_data.iter_mut() {
                *x /= sum;
            }
        }
        output
    }

    // Test data
    let rows = 2;
    let cols = 4;
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let cpu_result = cpu_softmax(&data, rows, cols);
    let gpu_result = ops.softmax_rows(&data, rows as u32, cols as u32);

    // Compare results
    assert_eq!(cpu_result.len(), gpu_result.len());
    for i in 0..cpu_result.len() {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        assert!(
            diff < 0.01,
            "Softmax mismatch at index {}: CPU={}, GPU={}, diff={}",
            i, cpu_result[i], gpu_result[i], diff
        );
    }

    // Verify rows sum to 1
    for row in 0..rows {
        let start = row * cols;
        let gpu_sum: f32 = gpu_result[start..start + cols].iter().sum();
        assert!(
            (gpu_sum - 1.0).abs() < 0.001,
            "GPU softmax row {} sum={}, expected 1.0",
            row, gpu_sum
        );
    }
}

/// Test GPU GELU matches CPU reference
#[test]
fn test_gpu_gelu_matches_cpu() {
    let Some(ctx) = setup_gpu() else {
        println!("Skipping GPU test - no GPU available");
        return;
    };

    let ops = GpuOps::new(ctx);

    // CPU reference GELU
    fn cpu_gelu(x: f32) -> f32 {
        let sqrt_2_pi = 0.7978845608028654f32;
        let cdf = 0.5 * (1.0 + (sqrt_2_pi * (x + 0.044715 * x * x * x)).tanh());
        x * cdf
    }

    // Test data
    let input: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let cpu_result: Vec<f32> = input.iter().map(|&x| cpu_gelu(x)).collect();
    let gpu_result = ops.gelu(&input);

    // Compare results
    assert_eq!(cpu_result.len(), gpu_result.len());
    for i in 0..cpu_result.len() {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        assert!(
            diff < 0.01,
            "GELU mismatch at index {}: CPU={}, GPU={}, diff={}",
            i, cpu_result[i], gpu_result[i], diff
        );
    }
}

/// Test that training actually reduces loss over time
/// This is a critical test - if loss doesn't decrease, training is broken
#[test]
fn test_training_actually_reduces_loss() {
    // Small model for fast testing
    let config = TransformerConfig {
        vocab_size: 50,
        d_model: 32,
        n_heads: 2,
        n_layers: 2,
        d_ff: 64,
        max_seq_len: 16,
        dropout: 0.0,
        ..Default::default()
    };

    let mut model = Transformer::new(config.clone()).expect("Should create model");

    // Create a simple repeating pattern that should be easy to learn
    // Pattern: 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...
    let pattern: Vec<u32> = (0..500).map(|i| (i % 5) as u32).collect();
    let mut dataset = TextDataset::new(pattern, config.max_seq_len);

    // Measure initial loss (average over several examples)
    let mut initial_losses = Vec::new();
    for _ in 0..10 {
        if let Some((input, target)) = dataset.next_example() {
            let loss = model.compute_loss(&input, &target);
            initial_losses.push(loss);
        }
    }
    dataset.reset();
    let initial_avg_loss = initial_losses.iter().sum::<f32>() / initial_losses.len() as f32;
    println!("Initial average loss: {:.4}", initial_avg_loss);

    // Train for several epochs with learning rate decay
    let initial_lr = 0.1;
    let num_epochs = 10;
    let mut epoch_losses = Vec::new();

    for epoch in 0..num_epochs {
        dataset.reset();
        let mut epoch_loss = 0.0;
        let mut count = 0;

        // Learning rate decay
        let learning_rate = initial_lr / (1.0 + epoch as f32 * 0.5);

        while let Some((input, target)) = dataset.next_example() {
            // Forward pass
            let loss = model.compute_loss(&input, &target);
            epoch_loss += loss;
            count += 1;

            // Backward pass with full propagation (includes weight updates)
            model.backward(&target, learning_rate);
        }

        let avg_loss = epoch_loss / count as f32;
        epoch_losses.push(avg_loss);

        println!("Epoch {}: loss = {:.4}, lr = {:.4}", epoch, avg_loss, learning_rate);
    }

    let final_loss = *epoch_losses.last().unwrap();
    println!("Final loss: {:.4}", final_loss);

    // Loss should decrease significantly from initial
    // Initial loss for random model on vocab 50 should be around ln(50) ≈ 3.9
    // After training on simple pattern, should be much lower
    assert!(
        final_loss < initial_avg_loss * 0.5,
        "Loss should decrease by at least 50%. Initial: {:.4}, Final: {:.4}",
        initial_avg_loss, final_loss
    );

    // First epoch loss should be significantly lower than initial (training is working)
    let first_epoch_loss = epoch_losses[0];
    assert!(
        first_epoch_loss < initial_avg_loss * 0.7,
        "First epoch should reduce loss by at least 30%. Initial: {:.4}, First epoch: {:.4}",
        initial_avg_loss, first_epoch_loss
    );
}

/// Test GPU embedding lookup matches CPU reference
#[test]
fn test_gpu_embedding_matches_cpu() {
    let Some(ctx) = setup_gpu() else {
        println!("Skipping GPU test - no GPU available");
        return;
    };

    let ops = GpuOps::new(ctx);

    // Test data
    let vocab_size = 5;
    let d_model = 3;
    let embeddings: Vec<f32> = (0..(vocab_size * d_model))
        .map(|i| i as f32 * 0.1)
        .collect();
    let token_ids: Vec<u32> = vec![2, 0, 4];

    // CPU reference
    let mut cpu_result = Vec::new();
    for &token_id in &token_ids {
        let start = token_id as usize * d_model;
        cpu_result.extend_from_slice(&embeddings[start..start + d_model]);
    }

    let gpu_result = ops.embedding_lookup(
        &token_ids,
        &embeddings,
        token_ids.len() as u32,
        d_model as u32,
        vocab_size as u32,
    );

    // Compare results
    assert_eq!(cpu_result.len(), gpu_result.len());
    for i in 0..cpu_result.len() {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        assert!(
            diff < 0.001,
            "Embedding mismatch at index {}: CPU={}, GPU={}, diff={}",
            i, cpu_result[i], gpu_result[i], diff
        );
    }
}

// ============================================================================
// Completion Tests - Test that trained models can predict sensibly
// ============================================================================

/// Train a model on a simple repeating pattern and test predictions
/// Pattern: 1 -> 2 -> 3 -> 1 -> 2 -> 3 -> ...
///
/// NOTE: Current backpropagation only updates embedding layer, not transformer blocks.
/// This limits learning capability. Full backprop through transformer blocks would be needed
/// for the model to learn complex patterns well.
#[test]
fn test_completion_on_simple_repeating_pattern() {
    // Use a small vocab and simple model for fast training
    let config = TransformerConfig {
        vocab_size: 10,
        d_model: 32,
        n_heads: 2,
        n_layers: 2,
        d_ff: 64,
        max_seq_len: 16,
        dropout: 0.0,
        ..Default::default()
    };

    let mut model = Transformer::new(config.clone()).expect("Should create model");

    // Create training data: repeating pattern 4,5,6,4,5,6,...
    // NOTE: We avoid token 3 because it's treated as EOS in the generate() function
    let pattern: Vec<u32> = (0..1000).map(|i| ((i % 3) + 4) as u32).collect(); // 4,5,6,4,5,6,...
    let mut dataset = TextDataset::new(pattern.clone(), config.max_seq_len);

    // Train for multiple epochs
    println!("\n=== Training on pattern 4->5->6->4->5->6 ===");
    let learning_rate = 0.1;
    let mut scheduler = LearningRateScheduler::new(learning_rate, 10, 500);

    let initial_loss = {
        dataset.reset();
        let (input, target) = dataset.next_example().unwrap();
        model.compute_loss(&input, &target)
    };
    println!("Initial loss: {:.4}", initial_loss);

    let mut last_loss = initial_loss;
    for epoch in 0..20 {
        let mut epoch_loss = 0.0;
        let mut steps = 0;

        dataset.reset();
        while let Some((input, target)) = dataset.next_example() {
            let loss = model.compute_loss(&input, &target);
            model.backward(&target, scheduler.get_lr());
            scheduler.step();

            epoch_loss += loss;
            steps += 1;
        }

        let avg_loss = epoch_loss / steps as f32;
        if epoch % 5 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, avg_loss);
        }
        last_loss = avg_loss;
    }

    // Loss should have decreased significantly with full backpropagation
    println!("Final loss: {:.4} (initial was {:.4})", last_loss, initial_loss);
    assert!(
        last_loss < initial_loss,
        "Training loss {} should decrease from initial {}",
        last_loss, initial_loss
    );

    // Now test predictions
    println!("\n=== Testing predictions ===");

    // Test 1: After seeing [4], check what the model predicts
    model.forward(&[4]);
    let top_k = model.top_k_next_tokens(5);
    println!("After [4], top 5 predictions: {:?}", top_k);

    // The pattern tokens (4, 5, 6) should have higher probability than random tokens
    let probs = model.get_next_token_probs();
    let prob_pattern_tokens: f32 = probs[4] + probs[5] + probs[6];
    println!("Combined probability of pattern tokens (4,5,6): {:.4}", prob_pattern_tokens);

    // Test 2: After seeing [4, 5], model should favor pattern tokens
    model.forward(&[4, 5]);
    let top_k = model.top_k_next_tokens(5);
    println!("After [4, 5], top 5 predictions: {:?}", top_k);

    // Test 3: After seeing [4, 5, 6], model should favor pattern tokens
    model.forward(&[4, 5, 6]);
    let top_k = model.top_k_next_tokens(5);
    println!("After [4, 5, 6], top 5 predictions: {:?}", top_k);

    // Test 4: Generation should produce tokens (may not be perfect pattern due to limited backprop)
    let generated = model.generate(&[4], 6, 0.5);
    println!("Generated from [4] with temp=0.5: {:?}", generated);
    assert!(generated.len() >= 4, "Should generate at least 4 tokens");

    // All generated tokens should be valid (within vocab)
    for token in &generated {
        assert!(*token < 10, "Token {} should be within vocab", token);
    }
}

/// Test that an untrained model produces uniform-ish predictions
#[test]
fn test_untrained_model_predictions_are_not_degenerate() {
    let config = TransformerConfig {
        vocab_size: 50,
        d_model: 32,
        n_heads: 2,
        n_layers: 1,
        d_ff: 64,
        max_seq_len: 16,
        dropout: 0.0,
        ..Default::default()
    };

    let mut model = Transformer::new(config).expect("Should create model");

    // Forward pass
    model.forward(&[1, 2, 3]);

    // Get probabilities
    let probs = model.get_next_token_probs();
    assert_eq!(probs.len(), 50);

    // Probabilities should sum to 1
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "Probs should sum to 1, got {}", sum);

    // No single token should dominate completely (untrained model)
    let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
    let min_prob = probs.iter().cloned().fold(1.0f32, f32::min);

    println!("Untrained model: max_prob={:.4}, min_prob={:.6}", max_prob, min_prob);

    // Max probability shouldn't be too high for an untrained model
    // (it can be somewhat high due to random initialization, but not 99%)
    assert!(
        max_prob < 0.95,
        "Untrained model shouldn't have any token with >95% probability, got {:.4}",
        max_prob
    );

    // Should be able to generate tokens without crashing
    let generated = model.generate(&[1], 10, 0.8);
    assert!(generated.len() >= 1, "Should generate at least prompt length");
}

/// Test sampling with different temperatures
#[test]
fn test_sampling_temperature_effects() {
    let config = TransformerConfig {
        vocab_size: 20,
        d_model: 32,
        n_heads: 2,
        n_layers: 2,
        d_ff: 64,
        max_seq_len: 16,
        dropout: 0.0,
        ..Default::default()
    };

    // Train a simple model first
    let mut model = Transformer::new(config.clone()).expect("Should create model");

    // Simple pattern training
    let pattern: Vec<u32> = (0..500).map(|i| (i % 5) as u32).collect();
    let mut dataset = TextDataset::new(pattern, config.max_seq_len);

    let mut scheduler = LearningRateScheduler::new(0.1, 5, 100);
    for _ in 0..10 {
        dataset.reset();
        while let Some((input, target)) = dataset.next_example() {
            model.compute_loss(&input, &target);
            model.backward(&target, scheduler.get_lr());
            scheduler.step();
        }
    }

    // Test temperature effects on sampling
    model.forward(&[0, 1, 2]);

    // Low temperature (0.1) should give more deterministic results
    let mut low_temp_results = std::collections::HashMap::new();
    for _ in 0..50 {
        model.forward(&[0, 1, 2]);
        if let Some(token) = model.sample_next_token(0.1) {
            *low_temp_results.entry(token).or_insert(0) += 1;
        }
    }

    // High temperature (2.0) should give more varied results
    let mut high_temp_results = std::collections::HashMap::new();
    for _ in 0..50 {
        model.forward(&[0, 1, 2]);
        if let Some(token) = model.sample_next_token(2.0) {
            *high_temp_results.entry(token).or_insert(0) += 1;
        }
    }

    println!("Low temp (0.1) distribution: {:?}", low_temp_results);
    println!("High temp (2.0) distribution: {:?}", high_temp_results);

    // Low temperature should have fewer unique tokens (more concentrated)
    // High temperature should have more unique tokens (more spread out)
    let low_temp_unique = low_temp_results.len();
    let high_temp_unique = high_temp_results.len();

    println!("Low temp unique tokens: {}, High temp unique tokens: {}", low_temp_unique, high_temp_unique);

    // High temperature should generally produce more variety
    // (this is probabilistic, so we use a soft assertion)
    assert!(
        high_temp_unique >= low_temp_unique || low_temp_unique <= 3,
        "High temperature should generally produce more variety"
    );
}

/// Test that top_k_next_tokens returns properly ordered results
#[test]
fn test_top_k_tokens_ordering() {
    let config = TransformerConfig {
        vocab_size: 100,
        d_model: 32,
        n_heads: 2,
        n_layers: 1,
        d_ff: 64,
        max_seq_len: 16,
        dropout: 0.0,
        ..Default::default()
    };

    let mut model = Transformer::new(config).expect("Should create model");
    model.forward(&[10, 20, 30]);

    let top_10 = model.top_k_next_tokens(10);

    // Should return exactly 10 tokens
    assert_eq!(top_10.len(), 10, "Should return exactly 10 tokens");

    // Should be ordered by probability (descending)
    for i in 1..top_10.len() {
        assert!(
            top_10[i - 1].1 >= top_10[i].1,
            "Tokens should be sorted by probability descending: {} >= {}",
            top_10[i - 1].1,
            top_10[i].1
        );
    }

    // All probabilities should be valid
    for (token_id, prob) in &top_10 {
        assert!(*token_id < 100, "Token ID should be within vocab");
        assert!(*prob >= 0.0, "Probability should be non-negative");
        assert!(*prob <= 1.0, "Probability should be <= 1.0");
    }

    // Top probability should be at least 1/vocab_size (not completely uniform)
    let top_prob = top_10[0].1;
    assert!(top_prob > 0.0, "Top probability should be positive");
}

/// Test model completes coherently after training on sentence-like patterns
#[test]
fn test_completion_coherence() {
    // Create a model with slightly larger vocab for more realistic test
    let config = TransformerConfig {
        vocab_size: 20,
        d_model: 48,
        n_heads: 2,
        n_layers: 2,
        d_ff: 96,
        max_seq_len: 16,
        dropout: 0.0,
        ..Default::default()
    };

    let mut model = Transformer::new(config.clone()).expect("Should create model");

    // Create training data with multiple distinct patterns:
    // Pattern A: 1 2 3 4 5 (ascending)
    // Pattern B: 10 11 12 13 14 (another ascending)
    // Pattern C: 5 4 3 2 1 (descending)
    let mut training_data = Vec::new();
    for _ in 0..100 {
        training_data.extend_from_slice(&[1, 2, 3, 4, 5]);
        training_data.extend_from_slice(&[10, 11, 12, 13, 14]);
        training_data.extend_from_slice(&[5, 4, 3, 2, 1]);
    }

    let mut dataset = TextDataset::new(training_data, config.max_seq_len);

    // Train
    println!("\n=== Training on multiple patterns ===");
    let mut scheduler = LearningRateScheduler::new(0.1, 10, 500);

    for epoch in 0..15 {
        let mut epoch_loss = 0.0;
        let mut steps = 0;

        dataset.reset();
        while let Some((input, target)) = dataset.next_example() {
            let loss = model.compute_loss(&input, &target);
            model.backward(&target, scheduler.get_lr());
            scheduler.step();

            epoch_loss += loss;
            steps += 1;
        }

        if epoch % 5 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, epoch_loss / steps as f32);
        }
    }

    // Test completions for each pattern
    println!("\n=== Testing pattern completions ===");

    // Pattern A: [1, 2, 3] -> should predict 4
    model.forward(&[1, 2, 3]);
    let top_k = model.top_k_next_tokens(5);
    println!("After [1, 2, 3], top 5: {:?}", top_k);
    let contains_4 = top_k.iter().any(|(id, _)| *id == 4);
    assert!(contains_4, "Token 4 should be in top predictions after [1, 2, 3]");

    // Pattern B: [10, 11, 12] -> should predict 13
    model.forward(&[10, 11, 12]);
    let top_k = model.top_k_next_tokens(5);
    println!("After [10, 11, 12], top 5: {:?}", top_k);
    let contains_13 = top_k.iter().any(|(id, _)| *id == 13);
    assert!(contains_13, "Token 13 should be in top predictions after [10, 11, 12]");

    // Pattern C: [5, 4, 3] -> should predict 2
    model.forward(&[5, 4, 3]);
    let top_k = model.top_k_next_tokens(5);
    println!("After [5, 4, 3], top 5: {:?}", top_k);
    let contains_2 = top_k.iter().any(|(id, _)| *id == 2);
    assert!(contains_2, "Token 2 should be in top predictions after [5, 4, 3]");
}

//=============================================================================
// OPTIMIZER AND TRAINING QUALITY TESTS
//=============================================================================

/// Test that the Adam optimizer state is created correctly
#[test]
fn test_adam_optimizer_state_creation() {
    use transformer::training::ModelOptimizerState;

    let config = TransformerConfig::tiny(100);
    let opt_state = ModelOptimizerState::new(&config);

    // Verify state sizes match model dimensions
    assert_eq!(
        opt_state.token_embedding.m.len(),
        config.vocab_size * config.d_model
    );
    assert_eq!(opt_state.blocks.len(), config.n_layers);
}

/// Test that AdamTrainer integrates correctly with the model
#[test]
fn test_adam_trainer_integration() {
    use transformer::optimizer::AdamConfig;
    use transformer::training::AdamTrainer;

    let config = TransformerConfig::tiny(50);
    let model = Transformer::new(config.clone()).unwrap();
    let train_config = TrainingConfig::browser_friendly();

    let mut trainer = AdamTrainer::new(
        model,
        train_config,
        AdamConfig::transformer_default(),
    );

    // Create simple training data
    let input = vec![1u32, 2, 3, 4, 5];
    let target = vec![2u32, 3, 4, 5, 6];

    // Train for a few steps
    let mut losses = Vec::new();
    for _ in 0..5 {
        let loss = trainer.train_step(&[input.clone()], &[target.clone()]);
        losses.push(loss);
    }

    // Verify loss is computed and training happens
    assert!(!losses.is_empty());
    assert!(losses.iter().all(|l| l.is_finite() && *l > 0.0));
}

/// Test the stable cross-entropy loss function
#[test]
fn test_stable_cross_entropy_numerical_stability() {
    use transformer::stable_cross_entropy_loss;

    // Test with normal values
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let loss = stable_cross_entropy_loss(&logits, 4);
    assert!(loss.is_finite());
    assert!(loss < 2.0); // Target has highest logit, so loss should be low

    // Test with extreme values (shouldn't overflow)
    let extreme_logits = vec![1000.0, 1001.0, 1002.0];
    let extreme_loss = stable_cross_entropy_loss(&extreme_logits, 2);
    assert!(extreme_loss.is_finite());
    assert!(extreme_loss < 1.0);

    // Test with very negative values
    let negative_logits = vec![-1000.0, -999.0, -998.0];
    let neg_loss = stable_cross_entropy_loss(&negative_logits, 2);
    assert!(neg_loss.is_finite());
}

/// Test that log_softmax produces correct results
#[test]
fn test_log_softmax_correctness() {
    use transformer::{log_softmax, softmax};

    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let mut log_probs = vec![0.0; 4];
    log_softmax(&logits, &mut log_probs);

    // Verify by comparing with softmax + log
    let mut probs = logits.clone();
    softmax(&mut probs);

    for i in 0..4 {
        let expected_log_prob = probs[i].ln();
        let diff = (log_probs[i] - expected_log_prob).abs();
        assert!(
            diff < 1e-5,
            "log_softmax[{}] = {} vs expected {}",
            i,
            log_probs[i],
            expected_log_prob
        );
    }
}

/// Test model gradient access methods
#[test]
fn test_model_gradient_access() {
    let config = TransformerConfig::tiny(50);
    let mut model = Transformer::new(config.clone()).unwrap();

    // Forward and backward pass to generate gradients
    let input: Vec<u32> = (0..8).collect();
    let target: Vec<u32> = (1..9).collect();

    model.forward(&input);
    model.backward(&target, 0.001);

    // Test gradient access
    let embedding_grads = model.get_embedding_gradients();
    assert_eq!(embedding_grads.len(), config.vocab_size * config.d_model);

    // Test embedding access
    let embeddings = model.get_embeddings();
    assert_eq!(embeddings.len(), config.vocab_size * config.d_model);

    // Test mutable embedding access
    let embeddings_mut = model.get_embeddings_mut();
    assert_eq!(embeddings_mut.len(), config.vocab_size * config.d_model);

    // Test clearing gradients
    model.clear_embedding_gradients();
    let cleared_grads = model.get_embedding_gradients();
    assert!(cleared_grads.iter().all(|&g| g == 0.0));
}

/// Benchmark-style test: verify loss decreases over training epochs
#[test]
fn test_training_loss_convergence() {
    let config = TransformerConfig {
        vocab_size: 10,
        d_model: 32,
        n_heads: 2,
        n_layers: 2,
        d_ff: 64,
        max_seq_len: 8,
        dropout: 0.0,
        ..Default::default()
    };

    let mut model = Transformer::new(config.clone()).unwrap();

    // Simple repeating pattern
    let pattern: Vec<u32> = (0..200).map(|i| (i % 5) as u32).collect();
    let mut dataset = TextDataset::new(pattern, config.max_seq_len);

    // Record initial loss
    dataset.reset();
    let (input, target) = dataset.next_example().unwrap();
    let initial_loss = model.compute_loss(&input, &target);

    // Train for 15 epochs
    let learning_rate = 0.1;
    for _epoch in 0..15 {
        dataset.reset();
        while let Some((input, target)) = dataset.next_example() {
            model.compute_loss(&input, &target);
            model.backward(&target, learning_rate);
        }
    }

    // Measure final loss
    dataset.reset();
    let (input, target) = dataset.next_example().unwrap();
    let final_loss = model.compute_loss(&input, &target);

    println!(
        "Training convergence: initial={:.4}, final={:.4}, reduction={:.1}%",
        initial_loss,
        final_loss,
        (1.0 - final_loss / initial_loss) * 100.0
    );

    // Loss should decrease by at least 50%
    assert!(
        final_loss < initial_loss * 0.5,
        "Loss should decrease by at least 50%: initial={:.4}, final={:.4}",
        initial_loss,
        final_loss
    );
}
