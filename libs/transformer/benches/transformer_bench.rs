//! Micro-benchmarks for transformer components
//!
//! Run with: cargo bench -p transformer
//!
//! These benchmarks measure the raw performance of individual components:
//! - Forward pass throughput
//! - Backward pass throughput
//! - Softmax performance
//! - Attention computation
//! - Feed-forward layers

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use transformer::{softmax, Transformer, TransformerConfig};

/// Benchmark softmax function with various input sizes
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for size in [64, 256, 1024, 4096, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut logits: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            b.iter(|| {
                softmax(black_box(&mut logits));
            });
        });
    }

    group.finish();
}

/// Benchmark forward pass with different model sizes
fn bench_forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass");
    group.sample_size(50); // Fewer samples for slower benchmarks

    // Tiny model (fast)
    let tiny_config = TransformerConfig::tiny(100);
    let tiny_model = Transformer::new(tiny_config).unwrap();

    for seq_len in [8, 16, 32, 64].iter() {
        let input: Vec<u32> = (0..*seq_len as u32).map(|i| i % 100).collect();
        let tokens_per_pass = *seq_len as u64;

        group.throughput(Throughput::Elements(tokens_per_pass));
        group.bench_with_input(
            BenchmarkId::new("tiny", seq_len),
            &input,
            |b, input| {
                let mut model = tiny_model.clone();
                b.iter(|| {
                    model.forward(black_box(input))
                });
            },
        );
    }

    // Small model
    let small_config = TransformerConfig::small(100);
    let small_model = Transformer::new(small_config).unwrap();

    for seq_len in [8, 16, 32].iter() {
        let input: Vec<u32> = (0..*seq_len as u32).map(|i| i % 100).collect();
        let tokens_per_pass = *seq_len as u64;

        group.throughput(Throughput::Elements(tokens_per_pass));
        group.bench_with_input(
            BenchmarkId::new("small", seq_len),
            &input,
            |b, input| {
                let mut model = small_model.clone();
                b.iter(|| {
                    model.forward(black_box(input))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark backward pass (forward + backward)
fn bench_backward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_pass");
    group.sample_size(30); // Fewer samples for slower benchmarks

    let tiny_config = TransformerConfig::tiny(100);
    let learning_rate = 0.001;

    for seq_len in [8, 16, 32].iter() {
        let input: Vec<u32> = (0..*seq_len as u32).map(|i| i % 100).collect();
        let target: Vec<u32> = (1..=*seq_len as u32).map(|i| i % 100).collect();
        let tokens_per_pass = *seq_len as u64;

        group.throughput(Throughput::Elements(tokens_per_pass));
        group.bench_with_input(
            BenchmarkId::new("tiny", seq_len),
            &(&input, &target),
            |b, (input, target)| {
                let mut model = Transformer::new(tiny_config.clone()).unwrap();
                b.iter(|| {
                    model.forward(black_box(*input));
                    model.backward(black_box(*target), learning_rate);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full training step (forward + loss + backward)
fn bench_training_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_step");
    group.sample_size(20);

    let tiny_config = TransformerConfig::tiny(100);
    let learning_rate = 0.001;

    for seq_len in [16, 32].iter() {
        let input: Vec<u32> = (0..*seq_len as u32).map(|i| i % 100).collect();
        let target: Vec<u32> = (1..=*seq_len as u32).map(|i| i % 100).collect();

        group.throughput(Throughput::Elements(*seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("tiny_full_step", seq_len),
            &(&input, &target),
            |b, (input, target)| {
                let mut model = Transformer::new(tiny_config.clone()).unwrap();
                b.iter(|| {
                    // Forward pass
                    let logits = model.forward(black_box(*input));

                    // Compute loss (just for measurement)
                    let vocab_size = 100;
                    let mut total_loss = 0.0f32;
                    for (pos, &t) in target.iter().enumerate() {
                        let start = pos * vocab_size;
                        let end = start + vocab_size;
                        let mut probs = logits[start..end].to_vec();
                        softmax(&mut probs);
                        total_loss -= probs[t as usize].max(1e-7).ln();
                    }

                    // Backward pass with weight updates
                    model.backward(black_box(*target), learning_rate);

                    black_box(total_loss)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark token generation (autoregressive inference)
fn bench_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("generation");
    group.sample_size(20);

    let tiny_config = TransformerConfig::tiny(100);
    let model = Transformer::new(tiny_config).unwrap();

    for num_tokens in [4, 8, 16].iter() {
        let prompt: Vec<u32> = vec![1, 2, 3, 4];

        group.throughput(Throughput::Elements(*num_tokens as u64));
        group.bench_with_input(
            BenchmarkId::new("greedy", num_tokens),
            num_tokens,
            |b, &num_tokens| {
                let mut model = model.clone();
                b.iter(|| {
                    model.generate(black_box(&prompt), num_tokens, 1.0)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_softmax,
    bench_forward_pass,
    bench_backward_pass,
    bench_training_step,
    bench_generation,
);

criterion_main!(benches);
