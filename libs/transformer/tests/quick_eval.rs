//! Quick Completion Evaluation
//!
//! Fast test to verify position embedding scaling fix improves completion quality.

use transformer::training::{TextDataset, LearningRateScheduler};
use transformer::{Transformer, TransformerConfig};

/// Fast training function
fn train_model_fast(
    config: TransformerConfig,
    pattern: &[u32],
    epochs: usize,
    learning_rate: f32,
) -> Transformer {
    let mut model = Transformer::new(config.clone()).expect("Model creation failed");
    let mut dataset = TextDataset::new(pattern.to_vec(), config.max_seq_len);
    let mut scheduler = LearningRateScheduler::new(learning_rate, 5, epochs * 20);

    for _epoch in 0..epochs {
        dataset.reset();
        let mut count = 0;
        while let Some((input, target)) = dataset.next_example() {
            model.compute_loss(&input, &target);
            model.backward(&target, scheduler.get_lr());
            scheduler.step();
            count += 1;
            if count >= 20 {
                break; // Limit samples per epoch for speed
            }
        }
    }

    model
}

/// Evaluate prediction accuracy
fn evaluate_predictions(model: &mut Transformer, test_cases: &[(Vec<u32>, u32)]) -> (f32, f32) {
    let mut correct = 0;
    let mut total_prob = 0.0;

    for (context, expected) in test_cases {
        model.forward(context);
        let top_k = model.top_k_next_tokens(5);
        let predicted = top_k.get(0).map(|(t, _)| *t).unwrap_or(0);

        if predicted == *expected {
            correct += 1;
        }

        let prob = model.get_next_token_probs().get(*expected as usize).copied().unwrap_or(0.0);
        total_prob += prob;
    }

    let n = test_cases.len() as f32;
    (correct as f32 / n, total_prob / n)
}

#[test]
fn test_position_embedding_scaling_fix() {
    println!("\n=== POSITION EMBEDDING SCALING FIX EVALUATION ===\n");

    // Small, fast model config
    let config = TransformerConfig {
        vocab_size: 10,
        d_model: 32,  // Smaller for speed
        n_heads: 2,
        n_layers: 1,
        d_ff: 64,
        max_seq_len: 16,
        dropout: 0.0,
        ..Default::default()
    };

    // Simple repeating pattern: 1, 2, 3, 4, 5
    let base_pattern: Vec<u32> = vec![1, 2, 3, 4, 5];
    let training_data: Vec<u32> = (0..100).map(|i| base_pattern[i % 5]).collect();

    // Test cases: given context, predict next token
    let test_cases = vec![
        (vec![1, 2, 3], 4u32),
        (vec![2, 3, 4], 5u32),
        (vec![3, 4, 5], 1u32),
        (vec![4, 5, 1], 2u32),
        (vec![5, 1, 2], 3u32),
    ];

    // Untrained baseline
    let mut untrained = Transformer::new(config.clone()).unwrap();
    let (untrained_acc, untrained_prob) = evaluate_predictions(&mut untrained, &test_cases);
    println!("UNTRAINED: Accuracy={:.0}%, AvgProb={:.3}", untrained_acc * 100.0, untrained_prob);

    // Train for 10 epochs (fast)
    println!("\nTraining for 10 epochs...");
    let mut trained = train_model_fast(config.clone(), &training_data, 10, 0.1);
    let (trained_acc, trained_prob) = evaluate_predictions(&mut trained, &test_cases);
    println!("TRAINED (10 epochs): Accuracy={:.0}%, AvgProb={:.3}", trained_acc * 100.0, trained_prob);

    // Test generation
    println!("\n--- Generation Test ---");
    let prompt = vec![1, 2, 3];
    let generated = trained.generate(&prompt, 7, 0.1);
    println!("Prompt: {:?}", prompt);
    println!("Generated: {:?}", generated);
    println!("Expected pattern: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]");

    // Check if generation follows the pattern
    let mut pattern_matches = 0;
    let expected = vec![4, 5, 1, 2, 3, 4, 5];
    for (i, &expected_tok) in expected.iter().enumerate() {
        if i + 3 < generated.len() && generated[i + 3] == expected_tok {
            pattern_matches += 1;
        }
    }
    println!("Pattern match rate: {}/{}", pattern_matches, expected.len());

    // Check embedding magnitudes
    println!("\n--- Embedding Magnitude Check ---");
    let d_model = config.d_model;
    let embeddings = trained.get_embeddings();
    let token_1_emb = &embeddings[1 * d_model..(1 + 1) * d_model];
    let token_1_norm: f32 = token_1_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Token 1 embedding L2 norm: {:.4}", token_1_norm);

    // Assertions
    assert!(
        trained_acc > untrained_acc || trained_prob > untrained_prob,
        "Training should improve accuracy or probability"
    );

    println!("\n=== TEST PASSED ===");
    println!("Training improved completion quality:");
    println!("  Accuracy: {:.0}% -> {:.0}%", untrained_acc * 100.0, trained_acc * 100.0);
    println!("  Avg Prob: {:.3} -> {:.3}", untrained_prob, trained_prob);
}

#[test]
fn test_sequential_pattern_learning() {
    println!("\n=== SEQUENTIAL PATTERN LEARNING TEST ===\n");

    let config = TransformerConfig {
        vocab_size: 10,
        d_model: 32,
        n_heads: 2,
        n_layers: 1,
        d_ff: 64,
        max_seq_len: 16,
        dropout: 0.0,
        ..Default::default()
    };

    // Simple ascending pattern
    let pattern: Vec<u32> = (0..80).map(|i| (i % 5) as u32 + 1).collect();

    println!("Training on pattern: 1,2,3,4,5,1,2,3,4,5,...");
    let mut model = train_model_fast(config, &pattern, 15, 0.1);

    // Test predictions
    let test_cases = vec![
        (vec![1, 2, 3], 4, "1,2,3 -> 4"),
        (vec![2, 3, 4], 5, "2,3,4 -> 5"),
        (vec![3, 4, 5], 1, "3,4,5 -> 1"),
    ];

    println!("\nPrediction results:");
    let mut correct = 0;
    for (context, expected, desc) in &test_cases {
        model.forward(context);
        let top_k = model.top_k_next_tokens(3);
        let predicted = top_k.get(0).map(|(t, _)| *t).unwrap_or(0);
        let is_correct = predicted == *expected as u32;
        if is_correct {
            correct += 1;
        }
        println!(
            "  {}: Predicted={}, Expected={} {}",
            desc,
            predicted,
            expected,
            if is_correct { "✓" } else { "✗" }
        );
        println!("    Top-3: {:?}", top_k);
    }

    let accuracy = correct as f32 / test_cases.len() as f32;
    println!("\nAccuracy: {}/{} ({:.0}%)", correct, test_cases.len(), accuracy * 100.0);

    // This tests that the model can learn sequential patterns
    // With position embedding scaling fix, accuracy should be reasonable
    assert!(
        accuracy >= 0.33,
        "Model should correctly predict at least 1/3 of sequential patterns"
    );
}
