//! Training Quality Benchmarks
//!
//! These benchmarks measure model quality metrics after training:
//! - Loss reduction after N epochs
//! - Convergence rate (epochs to reach target loss)
//! - Pattern learning accuracy
//!
//! Run with: cargo bench -p transformer --bench training_quality
//!
//! This is critical for verifying that optimizations don't degrade training quality.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use transformer::training::{TextDataset, TrainingConfig};
use transformer::{softmax, Transformer, TransformerConfig};

/// Training result for analysis
#[derive(Debug, Clone)]
struct TrainingResult {
    initial_loss: f32,
    final_loss: f32,
    loss_history: Vec<f32>,
    epochs_completed: usize,
    total_steps: usize,
}

impl TrainingResult {
    fn loss_reduction_ratio(&self) -> f32 {
        if self.initial_loss > 0.0 {
            self.final_loss / self.initial_loss
        } else {
            1.0
        }
    }

    fn converged(&self, target_ratio: f32) -> bool {
        self.loss_reduction_ratio() < target_ratio
    }
}

/// Train a model and return the results
fn train_model(
    config: TransformerConfig,
    training_config: TrainingConfig,
    tokens: Vec<u32>,
    epochs: usize,
) -> TrainingResult {
    let mut model = Transformer::new(config.clone()).unwrap();
    let mut dataset = TextDataset::new(tokens, training_config.seq_length);

    let vocab_size = config.vocab_size;
    let learning_rate = training_config.learning_rate;

    let mut loss_history = Vec::new();
    let mut total_steps = 0;

    // Compute initial loss
    dataset.reset();
    let mut initial_losses = Vec::new();
    for _ in 0..5 {
        if let Some((input, target)) = dataset.next_example() {
            let logits = model.forward(&input);
            let mut loss = 0.0f32;
            for (pos, &t) in target.iter().enumerate() {
                let start = pos * vocab_size;
                let end = start + vocab_size;
                let mut probs = logits[start..end].to_vec();
                softmax(&mut probs);
                loss -= probs[t as usize].max(1e-7).ln();
            }
            initial_losses.push(loss / target.len() as f32);
        }
    }
    let initial_loss = initial_losses.iter().sum::<f32>() / initial_losses.len().max(1) as f32;

    // Training loop
    for _epoch in 0..epochs {
        dataset.reset();

        while let Some((input, target)) = dataset.next_example() {
            // Forward pass
            let logits = model.forward(&input);

            // Compute loss
            let mut loss = 0.0f32;
            for (pos, &t) in target.iter().enumerate() {
                let start = pos * vocab_size;
                let end = start + vocab_size;
                let mut probs = logits[start..end].to_vec();
                softmax(&mut probs);
                loss -= probs[t as usize].max(1e-7).ln();
            }
            loss /= target.len() as f32;

            // Backward pass
            model.backward(&target, learning_rate);

            loss_history.push(loss);
            total_steps += 1;
        }
    }

    // Compute final loss
    dataset.reset();
    let mut final_losses = Vec::new();
    for _ in 0..5 {
        if let Some((input, target)) = dataset.next_example() {
            let logits = model.forward(&input);
            let mut loss = 0.0f32;
            for (pos, &t) in target.iter().enumerate() {
                let start = pos * vocab_size;
                let end = start + vocab_size;
                let mut probs = logits[start..end].to_vec();
                softmax(&mut probs);
                loss -= probs[t as usize].max(1e-7).ln();
            }
            final_losses.push(loss / target.len() as f32);
        }
    }
    let final_loss = final_losses.iter().sum::<f32>() / final_losses.len().max(1) as f32;

    TrainingResult {
        initial_loss,
        final_loss,
        loss_history,
        epochs_completed: epochs,
        total_steps,
    }
}

/// Benchmark: Loss reduction after N epochs on repeating pattern
fn bench_pattern_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_learning");
    group.sample_size(10); // Minimum allowed by Criterion

    // Simple repeating pattern: 1,2,3,4,5,1,2,3,4,5,...
    let pattern: Vec<u32> = (0..200).map(|i| (i % 5) + 1).collect();

    let model_config = TransformerConfig::tiny(10); // Small vocab
    let mut train_config = TrainingConfig::browser_friendly();
    train_config.seq_length = 8;
    train_config.learning_rate = 0.01;

    for epochs in [5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("epochs", epochs),
            epochs,
            |b, &epochs| {
                b.iter(|| {
                    let result = train_model(
                        model_config.clone(),
                        train_config.clone(),
                        pattern.clone(),
                        epochs,
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Convergence speed (how fast does loss drop by 50%?)
fn bench_convergence_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence_speed");
    group.sample_size(10);

    // Longer repeating pattern
    let pattern: Vec<u32> = (0..500).map(|i| (i % 10)).collect();

    let model_config = TransformerConfig::tiny(20);
    let mut train_config = TrainingConfig::browser_friendly();
    train_config.seq_length = 16;
    train_config.learning_rate = 0.005;

    group.bench_function("time_to_50_percent_reduction", |b| {
        b.iter(|| {
            let result = train_model(
                model_config.clone(),
                train_config.clone(),
                pattern.clone(),
                20, // Max epochs
            );

            // Find epoch where we first achieve 50% loss reduction
            let target = result.initial_loss * 0.5;
            let epochs_to_target = result
                .loss_history
                .iter()
                .position(|&l| l < target)
                .unwrap_or(result.loss_history.len());

            black_box((result.final_loss, epochs_to_target))
        });
    });

    group.finish();
}

/// Benchmark: Training throughput (tokens per second)
fn bench_training_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_throughput");
    group.sample_size(10);

    let pattern: Vec<u32> = (0..1000).map(|i| (i % 20) as u32).collect();

    // Test different model sizes
    for (name, config) in [
        ("tiny", TransformerConfig::tiny(30)),
        ("small", TransformerConfig::small(30)),
    ] {
        let mut train_config = TrainingConfig::browser_friendly();
        train_config.seq_length = 16;
        train_config.learning_rate = 0.001;

        group.bench_function(BenchmarkId::new("model", name), |b| {
            b.iter(|| {
                let result = train_model(
                    config.clone(),
                    train_config.clone(),
                    pattern.clone(),
                    3, // Just 3 epochs for throughput measurement
                );
                black_box(result.total_steps)
            });
        });
    }

    group.finish();
}

/// Quality verification test (run as part of benchmarks to ensure quality)
fn bench_quality_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_verification");
    group.sample_size(10); // Minimum allowed by Criterion

    group.bench_function("verify_learning", |b| {
        let pattern: Vec<u32> = (0..300).map(|i| (i % 5) + 1).collect();
        let model_config = TransformerConfig::tiny(10);
        let mut train_config = TrainingConfig::browser_friendly();
        train_config.seq_length = 8;
        train_config.learning_rate = 0.01;

        b.iter(|| {
            let result = train_model(
                model_config.clone(),
                train_config.clone(),
                pattern.clone(),
                15,
            );

            // Verify loss decreased significantly
            let reduction = result.loss_reduction_ratio();
            assert!(
                reduction < 0.5,
                "Loss should decrease by at least 50%, got {:.1}% reduction (initial: {:.4}, final: {:.4})",
                (1.0 - reduction) * 100.0,
                result.initial_loss,
                result.final_loss
            );

            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pattern_learning,
    bench_convergence_speed,
    bench_training_throughput,
    bench_quality_verification,
);

criterion_main!(benches);
