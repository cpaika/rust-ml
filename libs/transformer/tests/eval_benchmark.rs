//! Comprehensive Evaluation Benchmark
//!
//! This test:
//! 1. Trains a baseline model
//! 2. Saves it for reproducibility
//! 3. Runs comprehensive evaluations
//! 4. Provides detailed metrics for before/after comparisons
//!
//! Run with: cargo test -p transformer --test eval_benchmark -- --nocapture

use std::path::Path;
use transformer::checkpoint::Checkpoint;
use transformer::eval::{Benchmarks, EvalMetrics, Evaluator};
use transformer::training::{LearningRateScheduler, TextDataset};
use transformer::{Transformer, TransformerConfig};

/// Model configuration for evaluation
fn eval_model_config() -> TransformerConfig {
    TransformerConfig {
        vocab_size: 16,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        max_seq_len: 32,
        dropout: 0.0,
        ..Default::default()
    }
}

/// Train a model on pattern data
fn train_model(
    config: TransformerConfig,
    pattern: &[u32],
    epochs: usize,
    learning_rate: f32,
) -> (Transformer, Vec<f32>) {
    let training_data: Vec<u32> = (0..500).map(|i| pattern[i % pattern.len()]).collect();

    let mut model = Transformer::new(config.clone()).expect("Model creation failed");
    let mut dataset = TextDataset::new(training_data, config.max_seq_len);
    let mut scheduler = LearningRateScheduler::new(learning_rate, 10, epochs * 100);

    let mut losses = Vec::new();

    for epoch in 0..epochs {
        dataset.reset();
        let mut epoch_loss = 0.0;
        let mut count = 0;

        while let Some((input, target)) = dataset.next_example() {
            let loss = model.compute_loss(&input, &target);
            model.backward(&target, scheduler.get_lr());
            scheduler.step();
            epoch_loss += loss;
            count += 1;
        }

        let avg_loss = epoch_loss / count as f32;
        losses.push(avg_loss);

        if epoch % 5 == 0 {
            println!("  Epoch {:3}: Loss = {:.4}", epoch, avg_loss);
        }
    }

    (model, losses)
}

/// Train a model on multiple patterns (improved training)
fn train_model_multi_pattern(
    config: TransformerConfig,
    patterns: &[&[u32]],
    epochs: usize,
    learning_rate: f32,
) -> (Transformer, Vec<f32>) {
    // Create mixed training data - interleave patterns
    let mut training_data: Vec<u32> = Vec::new();
    for round in 0..100 {
        for pattern in patterns {
            for &token in *pattern {
                training_data.push(token);
            }
            // Add pattern separator (helps model learn boundaries)
            if round % 5 == 0 {
                for &token in *pattern {
                    training_data.push(token);
                }
            }
        }
    }

    let mut model = Transformer::new(config.clone()).expect("Model creation failed");
    let mut dataset = TextDataset::new(training_data, config.max_seq_len);
    let mut scheduler = LearningRateScheduler::new(learning_rate, 10, epochs * 100);

    let mut losses = Vec::new();

    for epoch in 0..epochs {
        dataset.reset();
        let mut epoch_loss = 0.0;
        let mut count = 0;

        while let Some((input, target)) = dataset.next_example() {
            let loss = model.compute_loss(&input, &target);
            model.backward(&target, scheduler.get_lr());
            scheduler.step();
            epoch_loss += loss;
            count += 1;
        }

        let avg_loss = epoch_loss / count as f32;
        losses.push(avg_loss);

        if epoch % 5 == 0 {
            println!("  Epoch {:3}: Loss = {:.4}", epoch, avg_loss);
        }
    }

    (model, losses)
}

/// Run full evaluation suite on a model
fn run_full_evaluation(model: &mut Transformer, label: &str) -> EvalMetrics {
    println!("\n--- Running Evaluation: {} ---", label);

    let evaluator = Evaluator::new(model.get_config().max_seq_len);

    // Evaluate on multiple benchmarks
    let counting = Benchmarks::counting_pattern();
    let alternating = Benchmarks::alternating_pattern();

    let counting_metrics = evaluator.evaluate(model, &counting);
    let alternating_metrics = evaluator.evaluate(model, &alternating);

    // Average metrics across benchmarks
    let avg_metrics = EvalMetrics {
        accuracy: (counting_metrics.accuracy + alternating_metrics.accuracy) / 2.0,
        top3_rate: (counting_metrics.top3_rate + alternating_metrics.top3_rate) / 2.0,
        top5_rate: (counting_metrics.top5_rate + alternating_metrics.top5_rate) / 2.0,
        avg_correct_prob: (counting_metrics.avg_correct_prob + alternating_metrics.avg_correct_prob)
            / 2.0,
        perplexity: (counting_metrics.perplexity + alternating_metrics.perplexity) / 2.0,
        avg_loss: (counting_metrics.avg_loss + alternating_metrics.avg_loss) / 2.0,
        pattern_accuracy: (counting_metrics.pattern_accuracy + alternating_metrics.pattern_accuracy)
            / 2.0,
        num_samples: counting_metrics.num_samples + alternating_metrics.num_samples,
    };

    println!("\nCounting Pattern (1,2,3,4,5):");
    counting_metrics.print_report("Counting");

    println!("Alternating Pattern (1,2):");
    alternating_metrics.print_report("Alternating");

    avg_metrics.print_report(label);

    avg_metrics
}

#[test]
fn test_baseline_evaluation() {
    println!("\n======================================================================");
    println!("                    BASELINE MODEL EVALUATION");
    println!("======================================================================\n");

    let config = eval_model_config();
    let pattern: Vec<u32> = vec![1, 2, 3, 4, 5];

    // Evaluate untrained model
    println!("=== UNTRAINED MODEL ===");
    let mut untrained = Transformer::new(config.clone()).unwrap();
    let untrained_metrics = run_full_evaluation(&mut untrained, "Untrained");

    // Train model
    println!("\n=== TRAINING MODEL (20 epochs) ===");
    let (mut trained, losses) = train_model(config.clone(), &pattern, 20, 0.05);

    println!("\nTraining complete. Final loss: {:.4}", losses.last().unwrap_or(&0.0));

    // Evaluate trained model
    println!("\n=== TRAINED MODEL ===");
    let trained_metrics = run_full_evaluation(&mut trained, "Trained (20 epochs)");

    // Print comparison
    trained_metrics.compare(&untrained_metrics, "Trained vs Untrained");

    // Verify improvement
    assert!(
        trained_metrics.accuracy > untrained_metrics.accuracy,
        "Trained model should have better accuracy"
    );
    assert!(
        trained_metrics.avg_correct_prob > untrained_metrics.avg_correct_prob,
        "Trained model should assign higher probability to correct tokens"
    );
}

#[test]
fn test_training_duration_impact() {
    println!("\n======================================================================");
    println!("               TRAINING DURATION IMPACT EVALUATION");
    println!("======================================================================\n");

    let config = eval_model_config();
    let pattern: Vec<u32> = vec![1, 2, 3, 4, 5];

    println!("Epochs | Accuracy |  Top-3  |  Top-5  | Avg Prob | Perplexity | Pattern");
    println!("-------|----------|---------|---------|----------|------------|--------");

    let mut prev_metrics: Option<EvalMetrics> = None;

    for epochs in [5, 10, 20, 30, 50] {
        let (mut model, _losses) = train_model(config.clone(), &pattern, epochs, 0.05);

        let evaluator = Evaluator::new(config.max_seq_len);
        let dataset = Benchmarks::counting_pattern();
        let metrics = evaluator.evaluate(&mut model, &dataset);

        println!(
            "{:6} | {:7.1}% | {:6.1}% | {:6.1}% |   {:5.3}  |   {:6.2}   | {:5.1}%",
            epochs,
            metrics.accuracy * 100.0,
            metrics.top3_rate * 100.0,
            metrics.top5_rate * 100.0,
            metrics.avg_correct_prob,
            metrics.perplexity,
            metrics.pattern_accuracy * 100.0
        );

        if let Some(prev) = &prev_metrics {
            assert!(
                metrics.accuracy >= prev.accuracy * 0.9,
                "More training should not drastically reduce accuracy"
            );
        }
        prev_metrics = Some(metrics);
    }
}

#[test]
fn test_model_checkpoint_evaluation() {
    println!("\n======================================================================");
    println!("                  MODEL CHECKPOINT EVALUATION");
    println!("======================================================================\n");

    let config = eval_model_config();
    let pattern: Vec<u32> = vec![1, 2, 3, 4, 5];

    // Train a model
    println!("Training model...");
    let (model, losses) = train_model(config.clone(), &pattern, 15, 0.05);

    // Save checkpoint
    let checkpoint = Checkpoint::from_model(&model, 100, 15, losses.clone(), None);

    // Save to temp file
    let temp_dir = std::env::temp_dir();
    let checkpoint_path = temp_dir.join("eval_test_checkpoint.json");
    checkpoint.save(&checkpoint_path).expect("Checkpoint save failed");

    println!("Checkpoint saved to: {:?}", checkpoint_path);

    // Load and evaluate
    let loaded_checkpoint = Checkpoint::load(&checkpoint_path).expect("Checkpoint load failed");
    let mut loaded_model = loaded_checkpoint
        .restore_model()
        .expect("Model restoration failed");

    println!("\nEvaluating original model:");
    let original_metrics = run_full_evaluation(&mut model.clone(), "Original");

    println!("Evaluating loaded model:");
    let loaded_metrics = run_full_evaluation(&mut loaded_model, "Loaded");

    // Metrics should be identical
    assert!(
        (original_metrics.accuracy - loaded_metrics.accuracy).abs() < 0.001,
        "Loaded model should have same accuracy as original"
    );
    assert!(
        (original_metrics.perplexity - loaded_metrics.perplexity).abs() < 0.01,
        "Loaded model should have same perplexity as original"
    );

    // Clean up
    let _ = std::fs::remove_file(&checkpoint_path);

    println!("Checkpoint test passed - loaded model matches original!");
}

#[test]
fn test_generation_quality() {
    println!("\n======================================================================");
    println!("                   GENERATION QUALITY EVALUATION");
    println!("======================================================================\n");

    let config = eval_model_config();
    let pattern: Vec<u32> = vec![1, 2, 3, 4, 5];

    // Train a model
    let (mut model, _) = train_model(config.clone(), &pattern, 25, 0.05);

    // Test generation from various prompts
    let test_prompts = vec![
        (vec![1, 2, 3], "1,2,3 -> ?"),
        (vec![2, 3, 4], "2,3,4 -> ?"),
        (vec![3, 4, 5], "3,4,5 -> ?"),
        (vec![4, 5, 1], "4,5,1 -> ?"),
        (vec![5, 1, 2], "5,1,2 -> ?"),
    ];

    println!("Generation test (expected pattern: 1,2,3,4,5,1,2,3,4,5...):\n");
    println!("Prompt      | Generated                | Expected Next");
    println!("------------|--------------------------|---------------");

    let mut correct = 0;

    for (prompt, label) in &test_prompts {
        let generated = model.generate(prompt, 5, 0.0); // Greedy decoding
        let expected_next = pattern[(prompt.len()) % pattern.len()];

        let gen_str: String = generated
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let is_correct = generated.get(prompt.len()) == Some(&expected_next);
        if is_correct {
            correct += 1;
        }

        println!(
            "{:11} | {:24} | {} {}",
            label,
            gen_str,
            expected_next,
            if is_correct { "Y" } else { "N" }
        );
    }

    let accuracy = correct as f32 / test_prompts.len() as f32;
    println!("\nGeneration accuracy: {}/{} ({:.0}%)", correct, test_prompts.len(), accuracy * 100.0);

    assert!(
        accuracy >= 0.6,
        "Model should correctly predict at least 60% of next tokens"
    );
}

#[test]
fn test_learning_rate_impact() {
    println!("\n======================================================================");
    println!("                LEARNING RATE IMPACT EVALUATION");
    println!("======================================================================\n");

    let config = eval_model_config();
    let pattern: Vec<u32> = vec![1, 2, 3, 4, 5];

    println!("   LR   | Final Loss | Accuracy |  Top-3  | Perplexity");
    println!("--------|------------|----------|---------|------------");

    for lr in [0.01, 0.03, 0.05, 0.1, 0.2] {
        let (mut model, losses) = train_model(config.clone(), &pattern, 15, lr);

        let evaluator = Evaluator::new(config.max_seq_len);
        let dataset = Benchmarks::counting_pattern();
        let metrics = evaluator.evaluate(&mut model, &dataset);

        println!(
            "  {:.2}  |   {:6.4}   | {:6.1}%  | {:6.1}% |   {:6.2}",
            lr,
            losses.last().unwrap_or(&0.0),
            metrics.accuracy * 100.0,
            metrics.top3_rate * 100.0,
            metrics.perplexity
        );
    }
}

/// Main benchmark test that runs everything
#[test]
fn test_comprehensive_benchmark() {
    println!("\n");
    println!("========================================================================");
    println!("              COMPREHENSIVE MODEL EVALUATION BENCHMARK");
    println!("========================================================================");
    println!("\nThis benchmark evaluates model performance across multiple dimensions:");
    println!("- Next token prediction accuracy");
    println!("- Top-k inclusion rates");
    println!("- Perplexity on pattern data");
    println!("- Generation quality");
    println!("\n");

    let config = eval_model_config();
    println!("Model Configuration:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  d_model: {}", config.d_model);
    println!("  n_heads: {}", config.n_heads);
    println!("  n_layers: {}", config.n_layers);
    println!("  d_ff: {}", config.d_ff);
    println!("  max_seq_len: {}", config.max_seq_len);

    // Run all benchmarks
    println!("\n--- BENCHMARK RESULTS ---\n");

    // 1. Untrained baseline
    let mut untrained = Transformer::new(config.clone()).unwrap();
    let untrained_metrics = run_full_evaluation(&mut untrained, "Untrained Baseline");

    // 2. Train on single pattern (baseline approach)
    println!("\n========== BASELINE: Single Pattern Training ==========");
    let counting_pattern: Vec<u32> = vec![1, 2, 3, 4, 5];
    let (mut single_trained, single_losses) = train_model(config.clone(), &counting_pattern, 30, 0.05);
    let single_trained_metrics = run_full_evaluation(&mut single_trained, "Single Pattern (counting only)");

    // 3. Train on multiple patterns (improved approach)
    println!("\n========== IMPROVED: Multi-Pattern Training ==========");
    let alternating_pattern: Vec<u32> = vec![1, 2];
    let patterns: Vec<&[u32]> = vec![&counting_pattern, &alternating_pattern];
    let (mut multi_trained, multi_losses) = train_model_multi_pattern(config.clone(), &patterns, 40, 0.05);
    let multi_trained_metrics = run_full_evaluation(&mut multi_trained, "Multi Pattern (counting + alternating)");

    // 4. Summary
    println!("\n========================================================================");
    println!("                            FINAL SUMMARY");
    println!("========================================================================");
    println!(
        "\nSingle-pattern training: Loss ~{:.2} -> {:.4}",
        (config.vocab_size as f32).ln(),
        single_losses.last().unwrap_or(&0.0)
    );
    println!(
        "Multi-pattern training:  Loss ~{:.2} -> {:.4}",
        (config.vocab_size as f32).ln(),
        multi_losses.last().unwrap_or(&0.0)
    );

    println!("\n--- IMPROVEMENT COMPARISON ---");
    multi_trained_metrics.compare(&single_trained_metrics, "Multi-Pattern vs Single-Pattern");

    // Assertions for CI
    let evaluator = Evaluator::new(config.max_seq_len);
    let counting_dataset = Benchmarks::counting_pattern();
    let multi_counting = evaluator.evaluate(&mut multi_trained, &counting_dataset);

    assert!(multi_trained_metrics.accuracy > 0.5, "Should achieve >50% accuracy");
    assert!(
        multi_counting.perplexity < untrained_metrics.perplexity,
        "In-distribution perplexity should decrease"
    );

    // Multi-pattern should be better than single-pattern on overall accuracy
    // (because single-pattern gets ~50% on alternating which it wasn't trained on)
    println!("\n--- VERIFICATION ---");
    println!("Single-pattern overall accuracy: {:.1}%", single_trained_metrics.accuracy * 100.0);
    println!("Multi-pattern overall accuracy:  {:.1}%", multi_trained_metrics.accuracy * 100.0);

    // Multi-pattern training should improve overall accuracy
    assert!(
        multi_trained_metrics.accuracy >= single_trained_metrics.accuracy,
        "Multi-pattern training should achieve at least equal accuracy: multi={:.2}%, single={:.2}%",
        multi_trained_metrics.accuracy * 100.0,
        single_trained_metrics.accuracy * 100.0
    );

    println!("\nAll benchmarks passed!");
}
