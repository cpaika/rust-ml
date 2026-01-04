//! Tests for lm-viz crate

use crate::completion::CompletionState;
use crate::metrics::LMMetrics;

#[test]
fn test_metrics_roundtrip() {
    let mut metrics = LMMetrics::new();
    metrics.update_train(100, 2.5, 0.001, 1000);
    metrics.update_val(2.3);

    assert_eq!(metrics.step(), 100);
    assert!((metrics.train_loss() - 2.5).abs() < 0.001);
    assert!((metrics.val_loss() - 2.3).abs() < 0.001);
}

#[test]
fn test_completion_state_defaults() {
    let state = CompletionState::new();
    assert!(state.input.is_empty());
    assert!(state.generated_tokens.is_empty());
    assert!((state.temperature - 0.8).abs() < 0.001);
}

#[test]
fn test_loss_history_bounded() {
    let mut metrics = LMMetrics::new();

    // Add many entries
    for i in 0..15000 {
        metrics.update_train(i, 1.0, 0.001, 10);
    }

    // Should be bounded to 10000
    assert!(metrics.loss_history.len() <= 10000);
}
