//! Evaluation framework for transformer models
//!
//! Provides comprehensive metrics for measuring model quality:
//! - Next token prediction accuracy
//! - Top-k inclusion rates
//! - Perplexity on held-out data
//! - Pattern completion accuracy
//! - Generation quality metrics

use crate::{cross_entropy_loss, softmax, Transformer, TransformerConfig};
use serde::{Deserialize, Serialize};

/// Evaluation metrics for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// Next token prediction accuracy (exact match)
    pub accuracy: f32,
    /// Top-3 inclusion rate
    pub top3_rate: f32,
    /// Top-5 inclusion rate
    pub top5_rate: f32,
    /// Average probability assigned to correct token
    pub avg_correct_prob: f32,
    /// Perplexity on the evaluation set
    pub perplexity: f32,
    /// Average cross-entropy loss
    pub avg_loss: f32,
    /// Number of samples evaluated
    pub num_samples: usize,
    /// Pattern completion accuracy (for sequence prediction tasks)
    pub pattern_accuracy: f32,
}

impl Default for EvalMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            top3_rate: 0.0,
            top5_rate: 0.0,
            avg_correct_prob: 0.0,
            perplexity: f32::INFINITY,
            avg_loss: f32::INFINITY,
            num_samples: 0,
            pattern_accuracy: 0.0,
        }
    }
}

impl EvalMetrics {
    /// Print a formatted report of the metrics
    pub fn print_report(&self, label: &str) {
        println!("\n==================================================");
        println!("  Evaluation Report: {}", label);
        println!("==================================================");
        println!("  Accuracy:           {:6.2}%", self.accuracy * 100.0);
        println!("  Top-3 Rate:         {:6.2}%", self.top3_rate * 100.0);
        println!("  Top-5 Rate:         {:6.2}%", self.top5_rate * 100.0);
        println!("  Avg Correct Prob:   {:6.4}", self.avg_correct_prob);
        println!("  Perplexity:         {:6.2}", self.perplexity);
        println!("  Avg Loss:           {:6.4}", self.avg_loss);
        println!("  Pattern Accuracy:   {:6.2}%", self.pattern_accuracy * 100.0);
        println!("  Samples:            {}", self.num_samples);
        println!("==================================================\n");
    }

    /// Compare with another set of metrics and print improvement
    pub fn compare(&self, baseline: &EvalMetrics, label: &str) {
        println!("\n============================================================");
        println!("  Comparison: {} vs Baseline", label);
        println!("============================================================");
        println!(
            "  Accuracy:         {:6.2}% -> {:6.2}% ({:+.2}%)",
            baseline.accuracy * 100.0,
            self.accuracy * 100.0,
            (self.accuracy - baseline.accuracy) * 100.0
        );
        println!(
            "  Top-3 Rate:       {:6.2}% -> {:6.2}% ({:+.2}%)",
            baseline.top3_rate * 100.0,
            self.top3_rate * 100.0,
            (self.top3_rate - baseline.top3_rate) * 100.0
        );
        println!(
            "  Top-5 Rate:       {:6.2}% -> {:6.2}% ({:+.2}%)",
            baseline.top5_rate * 100.0,
            self.top5_rate * 100.0,
            (self.top5_rate - baseline.top5_rate) * 100.0
        );
        println!(
            "  Avg Correct Prob: {:6.4} -> {:6.4} ({:+.4})",
            baseline.avg_correct_prob,
            self.avg_correct_prob,
            self.avg_correct_prob - baseline.avg_correct_prob
        );
        println!(
            "  Perplexity:       {:6.2} -> {:6.2} ({:+.2})",
            baseline.perplexity,
            self.perplexity,
            self.perplexity - baseline.perplexity
        );
        println!(
            "  Pattern Accuracy: {:6.2}% -> {:6.2}% ({:+.2}%)",
            baseline.pattern_accuracy * 100.0,
            self.pattern_accuracy * 100.0,
            (self.pattern_accuracy - baseline.pattern_accuracy) * 100.0
        );
        println!("============================================================\n");
    }
}

/// Evaluation task: next token prediction
#[derive(Debug, Clone)]
pub struct NextTokenTask {
    /// Context tokens
    pub context: Vec<u32>,
    /// Expected next token
    pub expected: u32,
}

/// Evaluation task: pattern completion
#[derive(Debug, Clone)]
pub struct PatternTask {
    /// Prompt tokens
    pub prompt: Vec<u32>,
    /// Expected continuation
    pub expected_continuation: Vec<u32>,
}

/// Evaluation dataset
#[derive(Debug, Clone)]
pub struct EvalDataset {
    /// Next token prediction tasks
    pub next_token_tasks: Vec<NextTokenTask>,
    /// Pattern completion tasks
    pub pattern_tasks: Vec<PatternTask>,
    /// Raw token sequence for perplexity calculation
    pub perplexity_data: Vec<u32>,
}

impl EvalDataset {
    /// Create an empty dataset
    pub fn new() -> Self {
        Self {
            next_token_tasks: Vec::new(),
            pattern_tasks: Vec::new(),
            perplexity_data: Vec::new(),
        }
    }

    /// Create dataset from a repeating pattern (for testing pattern learning)
    pub fn from_pattern(pattern: &[u32], num_repeats: usize, context_len: usize) -> Self {
        let mut dataset = Self::new();

        // Create the full sequence
        let full_seq: Vec<u32> = (0..num_repeats)
            .flat_map(|_| pattern.iter().copied())
            .collect();

        // Create next token tasks
        for i in context_len..full_seq.len() {
            let context = full_seq[i - context_len..i].to_vec();
            let expected = full_seq[i];
            dataset.next_token_tasks.push(NextTokenTask { context, expected });
        }

        // Create pattern completion tasks
        for start in 0..pattern.len() {
            let prompt: Vec<u32> = (0..context_len)
                .map(|i| pattern[(start + i) % pattern.len()])
                .collect();
            let expected: Vec<u32> = (0..pattern.len())
                .map(|i| pattern[(start + context_len + i) % pattern.len()])
                .collect();
            dataset.pattern_tasks.push(PatternTask {
                prompt,
                expected_continuation: expected,
            });
        }

        // Perplexity data
        dataset.perplexity_data = full_seq;

        dataset
    }

    /// Create dataset from raw token sequence
    pub fn from_tokens(tokens: &[u32], context_len: usize) -> Self {
        let mut dataset = Self::new();

        // Create next token tasks
        for i in context_len..tokens.len() {
            let context = tokens[i - context_len..i].to_vec();
            let expected = tokens[i];
            dataset.next_token_tasks.push(NextTokenTask { context, expected });
        }

        dataset.perplexity_data = tokens.to_vec();
        dataset
    }
}

impl Default for EvalDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Model evaluator
pub struct Evaluator {
    /// Maximum sequence length for evaluation
    pub max_seq_len: usize,
}

impl Evaluator {
    /// Create a new evaluator
    pub fn new(max_seq_len: usize) -> Self {
        Self { max_seq_len }
    }

    /// Run full evaluation on a model
    pub fn evaluate(&self, model: &mut Transformer, dataset: &EvalDataset) -> EvalMetrics {
        let mut metrics = EvalMetrics::default();

        // Evaluate next token prediction
        if !dataset.next_token_tasks.is_empty() {
            let (accuracy, top3, top5, avg_prob) =
                self.eval_next_token(model, &dataset.next_token_tasks);
            metrics.accuracy = accuracy;
            metrics.top3_rate = top3;
            metrics.top5_rate = top5;
            metrics.avg_correct_prob = avg_prob;
            metrics.num_samples = dataset.next_token_tasks.len();
        }

        // Evaluate perplexity
        if dataset.perplexity_data.len() > 1 {
            let (perplexity, avg_loss) = self.eval_perplexity(model, &dataset.perplexity_data);
            metrics.perplexity = perplexity;
            metrics.avg_loss = avg_loss;
        }

        // Evaluate pattern completion
        if !dataset.pattern_tasks.is_empty() {
            metrics.pattern_accuracy = self.eval_pattern_completion(model, &dataset.pattern_tasks);
        }

        metrics
    }

    /// Evaluate next token prediction accuracy
    fn eval_next_token(
        &self,
        model: &mut Transformer,
        tasks: &[NextTokenTask],
    ) -> (f32, f32, f32, f32) {
        let mut correct = 0;
        let mut top3_correct = 0;
        let mut top5_correct = 0;
        let mut total_prob = 0.0f32;

        for task in tasks {
            // Truncate context if needed
            let context: Vec<u32> = if task.context.len() > self.max_seq_len {
                task.context[task.context.len() - self.max_seq_len..].to_vec()
            } else {
                task.context.clone()
            };

            model.forward(&context);
            let top_k = model.top_k_next_tokens(5);
            let probs = model.get_next_token_probs();

            // Check accuracy
            if !top_k.is_empty() && top_k[0].0 == task.expected {
                correct += 1;
            }

            // Check top-k
            if top_k.iter().take(3).any(|(t, _)| *t == task.expected) {
                top3_correct += 1;
            }
            if top_k.iter().any(|(t, _)| *t == task.expected) {
                top5_correct += 1;
            }

            // Get probability of correct token
            if let Some(&prob) = probs.get(task.expected as usize) {
                total_prob += prob;
            }
        }

        let n = tasks.len() as f32;
        (
            correct as f32 / n,
            top3_correct as f32 / n,
            top5_correct as f32 / n,
            total_prob / n,
        )
    }

    /// Evaluate perplexity on a token sequence
    fn eval_perplexity(&self, model: &mut Transformer, tokens: &[u32]) -> (f32, f32) {
        if tokens.len() < 2 {
            return (f32::INFINITY, f32::INFINITY);
        }

        let mut total_loss = 0.0f32;
        let mut count = 0;

        // Process in chunks to respect max_seq_len
        let chunk_size = self.max_seq_len.min(tokens.len() - 1);

        for start in (0..tokens.len() - 1).step_by(chunk_size) {
            let end = (start + chunk_size + 1).min(tokens.len());
            let chunk = &tokens[start..end];

            if chunk.len() < 2 {
                continue;
            }

            // Forward pass
            model.forward(&chunk[..chunk.len() - 1]);

            // Calculate loss for each position
            let vocab_size = model.get_config().vocab_size;
            let logits = model.get_logits();

            for (pos, &target) in chunk[1..].iter().enumerate() {
                let logit_start = pos * vocab_size;
                let logit_end = logit_start + vocab_size;

                if logit_end <= logits.len() {
                    let pos_logits = &logits[logit_start..logit_end];
                    let loss = cross_entropy_loss(pos_logits, target as usize);
                    if loss.is_finite() {
                        total_loss += loss;
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            return (f32::INFINITY, f32::INFINITY);
        }

        let avg_loss = total_loss / count as f32;
        let perplexity = avg_loss.exp();

        (perplexity, avg_loss)
    }

    /// Evaluate pattern completion accuracy
    fn eval_pattern_completion(&self, model: &mut Transformer, tasks: &[PatternTask]) -> f32 {
        let mut total_correct = 0;
        let mut total_tokens = 0;

        for task in tasks {
            let generated = model.generate(
                &task.prompt,
                task.expected_continuation.len(),
                0.0, // Greedy decoding
            );

            // Check how many continuation tokens match
            for (i, &expected) in task.expected_continuation.iter().enumerate() {
                let gen_idx = task.prompt.len() + i;
                if gen_idx < generated.len() && generated[gen_idx] == expected {
                    total_correct += 1;
                }
                total_tokens += 1;
            }
        }

        if total_tokens == 0 {
            return 0.0;
        }

        total_correct as f32 / total_tokens as f32
    }
}

/// Standard evaluation benchmarks
pub struct Benchmarks;

impl Benchmarks {
    /// Simple counting pattern: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...
    pub fn counting_pattern() -> EvalDataset {
        let pattern: Vec<u32> = vec![1, 2, 3, 4, 5];
        EvalDataset::from_pattern(&pattern, 20, 3)
    }

    /// Fibonacci-like pattern (mod vocab): each token is sum of previous two mod 10
    pub fn fibonacci_pattern() -> EvalDataset {
        let mut pattern = vec![1u32, 1];
        for _ in 2..20 {
            let next = (pattern[pattern.len() - 1] + pattern[pattern.len() - 2]) % 10;
            pattern.push(next);
        }
        EvalDataset::from_tokens(&pattern.repeat(5), 4)
    }

    /// Alternating pattern: 1, 2, 1, 2, 1, 2, ...
    pub fn alternating_pattern() -> EvalDataset {
        let pattern: Vec<u32> = vec![1, 2];
        EvalDataset::from_pattern(&pattern, 50, 2)
    }

    /// Longer pattern: 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, ...
    pub fn long_counting_pattern() -> EvalDataset {
        let pattern: Vec<u32> = (0..10).collect();
        EvalDataset::from_pattern(&pattern, 10, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_dataset_from_pattern() {
        let pattern = vec![1, 2, 3];
        let dataset = EvalDataset::from_pattern(&pattern, 3, 2);

        assert!(!dataset.next_token_tasks.is_empty());
        assert!(!dataset.pattern_tasks.is_empty());
        assert!(!dataset.perplexity_data.is_empty());
    }

    #[test]
    fn test_eval_metrics_default() {
        let metrics = EvalMetrics::default();
        assert_eq!(metrics.accuracy, 0.0);
        assert_eq!(metrics.num_samples, 0);
    }

    #[test]
    fn test_benchmarks() {
        let counting = Benchmarks::counting_pattern();
        assert!(!counting.next_token_tasks.is_empty());

        let fib = Benchmarks::fibonacci_pattern();
        assert!(!fib.next_token_tasks.is_empty());

        let alt = Benchmarks::alternating_pattern();
        assert!(!alt.next_token_tasks.is_empty());
    }

    #[test]
    fn test_evaluator_on_untrained_model() {
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

        let mut model = Transformer::new(config).unwrap();
        let dataset = Benchmarks::counting_pattern();
        let evaluator = Evaluator::new(16);

        let metrics = evaluator.evaluate(&mut model, &dataset);

        // Untrained model should have low accuracy (close to random)
        assert!(metrics.accuracy < 0.5, "Untrained model accuracy should be low");
        assert!(metrics.perplexity > 1.0, "Perplexity should be > 1");
    }
}
