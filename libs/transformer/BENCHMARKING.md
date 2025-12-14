# Transformer Benchmarking Guide

This document describes the evaluation framework and benchmarking tools for measuring transformer model performance.

## Quick Start

```bash
# Run all benchmark tests
cargo test -p transformer --test eval_benchmark -- --nocapture

# Run specific benchmark
cargo test -p transformer --test eval_benchmark test_comprehensive_benchmark -- --nocapture
```

## Evaluation Framework

The evaluation framework is located in `src/eval.rs` and provides comprehensive metrics for measuring model quality.

### Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **Accuracy** | Exact match rate for next-token prediction | 100% |
| **Top-3 Rate** | Target token in top 3 predictions | 100% |
| **Top-5 Rate** | Target token in top 5 predictions | 100% |
| **Avg Correct Prob** | Mean probability assigned to correct token | 1.0 |
| **Perplexity** | exp(avg cross-entropy loss) - lower is better | 1.0 |
| **Avg Loss** | Mean cross-entropy loss | 0.0 |
| **Pattern Accuracy** | Accuracy on pattern completion tasks | 100% |

### Understanding Perplexity

Perplexity measures how "surprised" the model is by the correct answer:

```
Perplexity = exp(average_loss)
```

| Perplexity | Interpretation |
|------------|----------------|
| 1.0 | Perfect prediction |
| 2.0 | ~50% confident on average |
| 10.0 | Like guessing among 10 options |
| vocab_size | Random guessing (untrained) |
| > vocab_size | Confidently wrong |

## Usage

### Basic Evaluation

```rust
use transformer::eval::{Evaluator, Benchmarks, EvalMetrics};
use transformer::{Transformer, TransformerConfig};

// Create model
let config = TransformerConfig::tiny(16);
let mut model = Transformer::new(config.clone()).unwrap();

// Create evaluator and dataset
let evaluator = Evaluator::new(config.max_seq_len);
let dataset = Benchmarks::counting_pattern();

// Run evaluation
let metrics = evaluator.evaluate(&mut model, &dataset);
metrics.print_report("My Model");
```

### Comparing Models

```rust
let baseline_metrics = evaluator.evaluate(&mut baseline_model, &dataset);
let improved_metrics = evaluator.evaluate(&mut improved_model, &dataset);

improved_metrics.compare(&baseline_metrics, "Improved vs Baseline");
```

### Custom Datasets

```rust
use transformer::eval::EvalDataset;

// From repeating pattern
let pattern = vec![1, 2, 3, 4, 5];
let dataset = EvalDataset::from_pattern(&pattern, 20, 3); // 20 repeats, context_len=3

// From raw tokens
let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
let dataset = EvalDataset::from_tokens(&tokens, 4); // context_len=4
```

## Available Benchmarks

### Standard Patterns

| Benchmark | Pattern | Description |
|-----------|---------|-------------|
| `counting_pattern()` | 1,2,3,4,5,1,2,3,4,5... | Simple sequential counting |
| `alternating_pattern()` | 1,2,1,2,1,2... | Binary alternation |
| `fibonacci_pattern()` | 1,1,2,3,5,8... (mod 10) | Fibonacci sequence |
| `long_counting_pattern()` | 0,1,2,3,4,5,6,7,8,9,0... | Full digit cycle |

### Running Standard Benchmarks

```rust
let counting = Benchmarks::counting_pattern();
let alternating = Benchmarks::alternating_pattern();
let fibonacci = Benchmarks::fibonacci_pattern();
```

## Benchmark Tests

Located in `tests/eval_benchmark.rs`:

### `test_baseline_evaluation`
Compares untrained vs trained model on pattern learning.

```bash
cargo test -p transformer --test eval_benchmark test_baseline_evaluation -- --nocapture
```

### `test_training_duration_impact`
Shows how metrics improve with more training epochs.

```bash
cargo test -p transformer --test eval_benchmark test_training_duration_impact -- --nocapture
```

**Example output:**
```
Epochs | Accuracy |  Top-3  |  Top-5  | Avg Prob | Perplexity
-------|----------|---------|---------|----------|------------
     5 |   85.0%  |  100.0% |  100.0% |   0.850  |     2.50
    10 |   95.0%  |  100.0% |  100.0% |   0.950  |     1.20
    20 |  100.0%  |  100.0% |  100.0% |   0.997  |     1.00
```

### `test_learning_rate_impact`
Compares different learning rates.

```bash
cargo test -p transformer --test eval_benchmark test_learning_rate_impact -- --nocapture
```

### `test_generation_quality`
Tests autoregressive generation accuracy.

```bash
cargo test -p transformer --test eval_benchmark test_generation_quality -- --nocapture
```

### `test_model_checkpoint_evaluation`
Verifies checkpoint save/load preserves model performance.

```bash
cargo test -p transformer --test eval_benchmark test_model_checkpoint_evaluation -- --nocapture
```

### `test_comprehensive_benchmark`
Full benchmark comparing single-pattern vs multi-pattern training.

```bash
cargo test -p transformer --test eval_benchmark test_comprehensive_benchmark -- --nocapture
```

## Example Results

### Single-Pattern vs Multi-Pattern Training

Training on counting pattern only vs training on both counting and alternating:

| Metric | Single-Pattern | Multi-Pattern | Improvement |
|--------|---------------|---------------|-------------|
| Overall Accuracy | 75.00% | 90.21% | +15.21% |
| Top-3 Rate | 75.00% | 100.00% | +25.00% |
| Perplexity | 1525.80 | 31.20 | -98% |
| Pattern Accuracy | 32.50% | 57.50% | +25.00% |

**Key insight**: Single-pattern training achieves 100% on its training distribution but 50% (random) on unseen patterns. Multi-pattern training improves generalization.

## Training Functions

### Single Pattern Training

```rust
fn train_model(
    config: TransformerConfig,
    pattern: &[u32],
    epochs: usize,
    learning_rate: f32,
) -> (Transformer, Vec<f32>)
```

### Multi-Pattern Training

```rust
fn train_model_multi_pattern(
    config: TransformerConfig,
    patterns: &[&[u32]],
    epochs: usize,
    learning_rate: f32,
) -> (Transformer, Vec<f32>)
```

**Usage:**
```rust
let counting = vec![1, 2, 3, 4, 5];
let alternating = vec![1, 2];
let patterns: Vec<&[u32]> = vec![&counting, &alternating];

let (model, losses) = train_model_multi_pattern(config, &patterns, 40, 0.05);
```

## Model Configuration for Benchmarks

The standard eval configuration:

```rust
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
```

## Interpreting Results

### Good Results
- Accuracy > 90%
- Perplexity < 5
- Top-3 Rate = 100%
- Loss < 0.1

### Warning Signs
- Perplexity > vocab_size: Model is confidently wrong
- Accuracy = 50% on binary patterns: Random guessing
- Loss not decreasing: Learning rate issues or gradient problems

### Debugging Tips

1. **High perplexity on test data**: Model may be overfitting to training distribution
2. **Loss plateaus early**: Try higher learning rate or more epochs
3. **Unstable training**: Reduce learning rate, add warmup
4. **Poor generalization**: Train on more diverse patterns
