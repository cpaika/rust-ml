//! Completion Quality Evaluation Tests
//!
//! This module provides rigorous evaluation of model completion quality.
//! Run with: cargo test -p transformer --test completion_eval -- --nocapture
//!
//! Metrics measured:
//! - Next-token prediction accuracy on trained patterns
//! - Average probability assigned to correct tokens
//! - Pattern completion accuracy (full sequence prediction)
//! - Top-k inclusion rate (is correct token in top-k predictions?)

use transformer::training::{TextDataset, LearningRateScheduler};
use transformer::{Transformer, TransformerConfig};

/// Evaluation results for completion quality
#[derive(Debug, Clone)]
pub struct CompletionEvalResults {
    /// Accuracy of predicting the exact next token
    pub next_token_accuracy: f32,
    /// Average probability assigned to the correct next token
    pub avg_correct_prob: f32,
    /// Rate at which correct token appears in top-3 predictions
    pub top3_inclusion_rate: f32,
    /// Rate at which correct token appears in top-5 predictions
    pub top5_inclusion_rate: f32,
    /// Pattern completion accuracy (predicting full sequence)
    pub pattern_completion_accuracy: f32,
    /// Number of test samples
    pub num_samples: usize,
}

impl CompletionEvalResults {
    pub fn print_report(&self, label: &str) {
        println!("\n============================================");
        println!("  Completion Quality Report: {}", label);
        println!("============================================");
        println!("  Next Token Accuracy:      {:.1}%", self.next_token_accuracy * 100.0);
        println!("  Avg Correct Token Prob:   {:.3}", self.avg_correct_prob);
        println!("  Top-3 Inclusion Rate:     {:.1}%", self.top3_inclusion_rate * 100.0);
        println!("  Top-5 Inclusion Rate:     {:.1}%", self.top5_inclusion_rate * 100.0);
        println!("  Pattern Completion Acc:   {:.1}%", self.pattern_completion_accuracy * 100.0);
        println!("  Samples Evaluated:        {}", self.num_samples);
        println!("============================================\n");
    }
}

/// Evaluate completion quality on a trained model
pub fn evaluate_completion_quality(
    model: &mut Transformer,
    test_patterns: &[(Vec<u32>, u32)], // (context, expected_next_token)
) -> CompletionEvalResults {
    let mut correct_predictions = 0;
    let mut total_correct_prob = 0.0f32;
    let mut top3_correct = 0;
    let mut top5_correct = 0;

    for (context, expected_token) in test_patterns {
        // Forward pass
        model.forward(context);

        // Get predictions
        let top_k = model.top_k_next_tokens(5);
        let probs = model.get_next_token_probs();

        // Check if prediction is correct
        if !top_k.is_empty() && top_k[0].0 == *expected_token {
            correct_predictions += 1;
        }

        // Get probability of correct token
        let correct_prob = probs.get(*expected_token as usize).copied().unwrap_or(0.0);
        total_correct_prob += correct_prob;

        // Check top-k inclusion
        if top_k.iter().take(3).any(|(t, _)| *t == *expected_token) {
            top3_correct += 1;
        }
        if top_k.iter().any(|(t, _)| *t == *expected_token) {
            top5_correct += 1;
        }
    }

    let n = test_patterns.len() as f32;

    CompletionEvalResults {
        next_token_accuracy: correct_predictions as f32 / n,
        avg_correct_prob: total_correct_prob / n,
        top3_inclusion_rate: top3_correct as f32 / n,
        top5_inclusion_rate: top5_correct as f32 / n,
        pattern_completion_accuracy: 0.0, // Will be computed separately
        num_samples: test_patterns.len(),
    }
}

/// Evaluate full pattern completion (generating multiple tokens)
pub fn evaluate_pattern_completion(
    model: &mut Transformer,
    prompt: &[u32],
    expected_sequence: &[u32],
    temperature: f32,
) -> f32 {
    let generated = model.generate(prompt, expected_sequence.len(), temperature);

    // Check how many tokens match
    let mut matches = 0;
    for (i, &expected) in expected_sequence.iter().enumerate() {
        let gen_idx = prompt.len() + i;
        if gen_idx < generated.len() && generated[gen_idx] == expected {
            matches += 1;
        }
    }

    matches as f32 / expected_sequence.len() as f32
}

/// Train a model and return it for evaluation
fn train_model_on_pattern(
    config: TransformerConfig,
    pattern: &[u32],
    epochs: usize,
    learning_rate: f32,
) -> Transformer {
    let mut model = Transformer::new(config.clone()).expect("Model creation failed");
    let mut dataset = TextDataset::new(pattern.to_vec(), config.max_seq_len);
    let mut scheduler = LearningRateScheduler::new(learning_rate, 10, epochs * 100);

    for _epoch in 0..epochs {
        dataset.reset();
        while let Some((input, target)) = dataset.next_example() {
            model.compute_loss(&input, &target);
            model.backward(&target, scheduler.get_lr());
            scheduler.step();
        }
    }

    model
}

/// Create test patterns for evaluation
/// Returns (context, expected_next_token) pairs
fn create_test_patterns_for_sequence(base_pattern: &[u32], context_len: usize) -> Vec<(Vec<u32>, u32)> {
    let mut test_cases = Vec::new();
    let pattern_len = base_pattern.len();

    // Create test cases where we give context and expect the next token in the pattern
    for start in 0..(pattern_len * 3 - context_len - 1) {
        let context: Vec<u32> = (0..context_len)
            .map(|i| base_pattern[(start + i) % pattern_len])
            .collect();
        let expected = base_pattern[(start + context_len) % pattern_len];
        test_cases.push((context, expected));
    }

    test_cases
}

//=============================================================================
// EVALUATION TESTS
//=============================================================================

/// Main evaluation test - measures completion quality before and after training
#[test]
fn test_completion_quality_evaluation() {
    println!("\n=== COMPLETION QUALITY EVALUATION ===\n");

    // Create a simple model configuration
    let config = TransformerConfig {
        vocab_size: 10,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        max_seq_len: 32,
        dropout: 0.0,
        ..Default::default()
    };

    // Simple repeating pattern: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...
    let base_pattern: Vec<u32> = vec![1, 2, 3, 4, 5];
    let training_data: Vec<u32> = (0..500).map(|i| base_pattern[i % 5]).collect();

    // Create test patterns
    let test_patterns = create_test_patterns_for_sequence(&base_pattern, 4);
    println!("Created {} test patterns", test_patterns.len());

    // Evaluate UNTRAINED model
    println!("\n--- Evaluating UNTRAINED model ---");
    let mut untrained_model = Transformer::new(config.clone()).unwrap();
    let untrained_results = evaluate_completion_quality(&mut untrained_model, &test_patterns);
    untrained_results.print_report("UNTRAINED");

    // Train the model
    println!("\n--- Training model for 20 epochs ---");
    let mut trained_model = train_model_on_pattern(config.clone(), &training_data, 20, 0.05);

    // Evaluate TRAINED model
    println!("\n--- Evaluating TRAINED model ---");
    let trained_results = evaluate_completion_quality(&mut trained_model, &test_patterns);
    trained_results.print_report("TRAINED (20 epochs)");

    // Evaluate pattern completion
    println!("\n--- Pattern Completion Test ---");
    let prompt = vec![1, 2, 3];
    let expected_continuation = vec![4, 5, 1, 2, 3];
    let completion_acc = evaluate_pattern_completion(
        &mut trained_model,
        &prompt,
        &expected_continuation,
        0.1, // Low temperature for more deterministic
    );
    println!("Pattern completion accuracy: {:.1}%", completion_acc * 100.0);

    // Print generated sequence
    let generated = trained_model.generate(&prompt, 10, 0.1);
    println!("Prompt: {:?}", prompt);
    println!("Generated: {:?}", generated);
    println!("Expected pattern would be: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]");

    // Assertions - trained model should be significantly better
    assert!(
        trained_results.next_token_accuracy > untrained_results.next_token_accuracy,
        "Trained model should have better accuracy"
    );
    assert!(
        trained_results.avg_correct_prob > untrained_results.avg_correct_prob,
        "Trained model should assign higher probability to correct tokens"
    );

    // Print improvement summary
    println!("\n=== IMPROVEMENT SUMMARY ===");
    println!(
        "Accuracy improvement: {:.1}% -> {:.1}% (+{:.1}%)",
        untrained_results.next_token_accuracy * 100.0,
        trained_results.next_token_accuracy * 100.0,
        (trained_results.next_token_accuracy - untrained_results.next_token_accuracy) * 100.0
    );
    println!(
        "Probability improvement: {:.3} -> {:.3} (+{:.3})",
        untrained_results.avg_correct_prob,
        trained_results.avg_correct_prob,
        trained_results.avg_correct_prob - untrained_results.avg_correct_prob
    );
}

/// Test to establish baseline metrics for comparison
#[test]
fn test_baseline_completion_metrics() {
    println!("\n=== BASELINE COMPLETION METRICS ===\n");

    let config = TransformerConfig {
        vocab_size: 10,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        max_seq_len: 32,
        dropout: 0.0,
        ..Default::default()
    };

    // Train on simple ascending pattern
    let pattern: Vec<u32> = (0..500).map(|i| (i % 5) as u32 + 1).collect();
    let mut model = train_model_on_pattern(config.clone(), &pattern, 30, 0.05);

    // Test predictions at different context lengths
    println!("Testing predictions at different context lengths:\n");

    for context_len in [2, 3, 4, 5, 6] {
        let test_patterns = create_test_patterns_for_sequence(&[1, 2, 3, 4, 5], context_len);
        let results = evaluate_completion_quality(&mut model, &test_patterns);
        println!(
            "Context length {}: Accuracy={:.1}%, AvgProb={:.3}, Top5={:.1}%",
            context_len,
            results.next_token_accuracy * 100.0,
            results.avg_correct_prob,
            results.top5_inclusion_rate * 100.0
        );
    }

    // Test actual generation
    println!("\n--- Generation Tests ---");
    for temp in [0.1, 0.5, 1.0] {
        let generated = model.generate(&[1, 2, 3], 8, temp);
        println!("Temp {:.1}: {:?}", temp, generated);
    }
}

/// Diagnostic test to check what the model is actually learning
#[test]
fn test_model_learns_position_info() {
    println!("\n=== POSITION LEARNING DIAGNOSTIC ===\n");

    let config = TransformerConfig {
        vocab_size: 10,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        max_seq_len: 16,
        dropout: 0.0,
        ..Default::default()
    };

    // Create a pattern where the next token depends on POSITION in sequence
    // Pattern: position 0 always followed by 1, position 1 by 2, etc.
    let pattern: Vec<u32> = (0..200).map(|i| (i % 5) as u32 + 1).collect();

    println!("Training pattern: 1,2,3,4,5,1,2,3,4,5,...");
    let mut model = train_model_on_pattern(config, &pattern, 30, 0.05);

    // Test: Given context [1,2,3], predict 4
    // Given context [2,3,4], predict 5
    // Given context [3,4,5], predict 1
    let test_cases = vec![
        (vec![1, 2, 3], 4u32, "After 1,2,3 expect 4"),
        (vec![2, 3, 4], 5u32, "After 2,3,4 expect 5"),
        (vec![3, 4, 5], 1u32, "After 3,4,5 expect 1"),
        (vec![4, 5, 1], 2u32, "After 4,5,1 expect 2"),
        (vec![5, 1, 2], 3u32, "After 5,1,2 expect 3"),
    ];

    println!("\nPrediction test results:");
    for (context, expected, description) in &test_cases {
        model.forward(context);
        let top_k = model.top_k_next_tokens(5);
        let predicted = top_k.get(0).map(|(t, _)| *t).unwrap_or(0);
        let correct = predicted == *expected;
        let prob = model.get_next_token_probs().get(*expected as usize).copied().unwrap_or(0.0);

        println!(
            "  {}: Predicted={}, Expected={}, Prob={:.3} {}",
            description,
            predicted,
            expected,
            prob,
            if correct { "✓" } else { "✗" }
        );
    }

    // Check embedding magnitudes to diagnose position vs token imbalance
    println!("\n--- Embedding Magnitude Analysis ---");
    let d_model = 64;
    let embeddings = model.get_embeddings();

    // Sample some token embeddings
    let token_1_emb = &embeddings[1 * d_model..(1 + 1) * d_model];
    let token_1_norm: f32 = token_1_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Token 1 embedding L2 norm: {:.4}", token_1_norm);
}

/// Test completion quality with different training durations
#[test]
fn test_completion_vs_training_epochs() {
    println!("\n=== COMPLETION QUALITY vs TRAINING EPOCHS ===\n");

    let config = TransformerConfig {
        vocab_size: 10,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        max_seq_len: 32,
        dropout: 0.0,
        ..Default::default()
    };

    let pattern: Vec<u32> = (0..500).map(|i| (i % 5) as u32 + 1).collect();
    let test_patterns = create_test_patterns_for_sequence(&[1, 2, 3, 4, 5], 4);

    println!("Epochs | Accuracy | AvgProb | Top5Rate");
    println!("-------|----------|---------|----------");

    for epochs in [5, 10, 20, 30, 50] {
        let mut model = train_model_on_pattern(config.clone(), &pattern, epochs, 0.05);
        let results = evaluate_completion_quality(&mut model, &test_patterns);

        println!(
            "{:6} | {:7.1}% | {:7.3} | {:7.1}%",
            epochs,
            results.next_token_accuracy * 100.0,
            results.avg_correct_prob,
            results.top5_inclusion_rate * 100.0
        );
    }
}
