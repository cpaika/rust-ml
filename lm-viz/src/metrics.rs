//! Training metrics tracking and display

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Training metrics for visualization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct LMMetrics {
    /// Current training step
    step: usize,
    /// Current epoch
    epoch: usize,
    /// Total steps in epoch
    steps_per_epoch: usize,
    /// Training loss
    train_loss: f32,
    /// Validation loss
    val_loss: f32,
    /// Training perplexity
    train_perplexity: f32,
    /// Validation perplexity
    val_perplexity: f32,
    /// Current learning rate
    learning_rate: f32,
    /// Tokens processed
    tokens_processed: usize,
    /// Training start time (ms)
    start_time: f64,
    /// Loss history for charting
    #[wasm_bindgen(skip)]
    pub loss_history: Vec<(usize, f32)>,
    /// Validation loss history
    #[wasm_bindgen(skip)]
    pub val_loss_history: Vec<(usize, f32)>,
    /// FLOPs per forward+backward pass (estimated)
    flops_per_step: usize,
}

#[wasm_bindgen]
impl LMMetrics {
    /// Create new metrics instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update metrics from a training step
    pub fn update_train(&mut self, step: usize, loss: f32, learning_rate: f32, tokens: usize) {
        self.step = step;
        self.train_loss = loss;
        self.train_perplexity = loss.exp();
        self.learning_rate = learning_rate;
        self.tokens_processed += tokens;
        self.loss_history.push((step, loss));

        // Keep history bounded for memory
        if self.loss_history.len() > 10000 {
            self.loss_history.remove(0);
        }
    }

    /// Update validation metrics
    pub fn update_val(&mut self, loss: f32) {
        self.val_loss = loss;
        self.val_perplexity = loss.exp();
        self.val_loss_history.push((self.step, loss));
    }

    /// Update epoch information
    pub fn set_epoch(&mut self, epoch: usize, steps_per_epoch: usize) {
        self.epoch = epoch;
        self.steps_per_epoch = steps_per_epoch;
    }

    /// Set training start time
    pub fn start_timer(&mut self, time_ms: f64) {
        self.start_time = time_ms;
    }

    /// Get current step
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get current epoch
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Get training loss
    pub fn train_loss(&self) -> f32 {
        self.train_loss
    }

    /// Get validation loss
    pub fn val_loss(&self) -> f32 {
        self.val_loss
    }

    /// Get training perplexity
    pub fn train_perplexity(&self) -> f32 {
        self.train_perplexity
    }

    /// Get validation perplexity
    pub fn val_perplexity(&self) -> f32 {
        self.val_perplexity
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Get tokens processed
    pub fn tokens_processed(&self) -> usize {
        self.tokens_processed
    }

    /// Get progress in current epoch (0.0 - 1.0)
    pub fn epoch_progress(&self) -> f32 {
        if self.steps_per_epoch == 0 {
            return 0.0;
        }
        let step_in_epoch = self.step % self.steps_per_epoch;
        step_in_epoch as f32 / self.steps_per_epoch as f32
    }

    /// Get tokens per second (throughput)
    pub fn tokens_per_second(&self, current_time_ms: f64) -> f32 {
        let elapsed_s = (current_time_ms - self.start_time) / 1000.0;
        if elapsed_s <= 0.0 {
            return 0.0;
        }
        self.tokens_processed as f32 / elapsed_s as f32
    }

    /// Set FLOPs per training step
    pub fn set_flops_per_step(&mut self, flops: usize) {
        self.flops_per_step = flops;
    }

    /// Get FLOPs per second (compute throughput)
    pub fn flops_per_second(&self, current_time_ms: f64) -> f64 {
        let elapsed_s = (current_time_ms - self.start_time) / 1000.0;
        if elapsed_s <= 0.0 || self.step == 0 {
            return 0.0;
        }
        (self.step as f64 * self.flops_per_step as f64) / elapsed_s
    }

    /// Format FLOPS with appropriate unit suffix
    pub fn format_flops(flops: f64) -> String {
        if flops >= 1e12 {
            format!("{:.2} TFLOPS", flops / 1e12)
        } else if flops >= 1e9 {
            format!("{:.2} GFLOPS", flops / 1e9)
        } else if flops >= 1e6 {
            format!("{:.2} MFLOPS", flops / 1e6)
        } else if flops >= 1e3 {
            format!("{:.2} KFLOPS", flops / 1e3)
        } else {
            format!("{:.0} FLOPS", flops)
        }
    }

    /// Get loss history as JSON for JavaScript charting
    pub fn loss_history_json(&self) -> String {
        serde_json::to_string(&self.loss_history).unwrap_or_default()
    }

    /// Get validation loss history as JSON
    pub fn val_loss_history_json(&self) -> String {
        serde_json::to_string(&self.val_loss_history).unwrap_or_default()
    }

    /// Reset metrics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Chart scale mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChartScale {
    #[default]
    Linear,
    Log,
    LogLog,
}

/// Render loss chart to canvas
pub fn render_loss_chart(
    ctx: &web_sys::CanvasRenderingContext2d,
    metrics: &LMMetrics,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    scale: ChartScale,
) {
    let padding = 40.0;
    let chart_x = x + padding;
    let chart_y = y + 20.0;
    let chart_width = width - padding * 2.0;
    let chart_height = height - padding - 20.0;

    // Background
    ctx.set_fill_style_str("#12121a");
    ctx.fill_rect(x, y, width, height);

    // Title
    ctx.set_fill_style_str("#e0e6ed");
    ctx.set_font("14px monospace");
    let _ = ctx.fill_text("Training Loss", x + 10.0, y + 16.0);

    if metrics.loss_history.is_empty() {
        ctx.set_fill_style_str("#6b7280");
        ctx.set_font("12px monospace");
        let _ = ctx.fill_text("No data yet", chart_x + chart_width / 2.0 - 30.0, chart_y + chart_height / 2.0);
        return;
    }

    // Find data range
    let (min_step, max_step) = (
        metrics.loss_history.first().map(|(s, _)| *s).unwrap_or(0),
        metrics.loss_history.last().map(|(s, _)| *s).unwrap_or(1),
    );

    let (min_loss, max_loss) = metrics.loss_history.iter()
        .fold((f32::MAX, f32::MIN), |(min, max), (_, loss)| {
            (min.min(*loss), max.max(*loss))
        });

    // Add validation loss to range
    let (min_loss, max_loss) = metrics.val_loss_history.iter()
        .fold((min_loss, max_loss), |(min, max), (_, loss)| {
            (min.min(*loss), max.max(*loss))
        });

    let loss_range = (max_loss - min_loss).max(0.1);
    let step_range = (max_step - min_step).max(1);

    // Draw axes
    ctx.set_stroke_style_str("#3d5a80");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(chart_x, chart_y);
    ctx.line_to(chart_x, chart_y + chart_height);
    ctx.line_to(chart_x + chart_width, chart_y + chart_height);
    ctx.stroke();

    // Draw axis labels
    ctx.set_fill_style_str("#6b7280");
    ctx.set_font("10px monospace");

    // Ensure min values are positive for log scales
    let safe_min_loss = min_loss.max(0.001);
    let safe_max_loss = max_loss.max(safe_min_loss + 0.001);
    let safe_min_step = min_step.max(1);
    let safe_max_step = max_step.max(safe_min_step + 1);

    // Y axis labels
    for i in 0..=4 {
        let frac = i as f32 / 4.0;
        let loss_val = match scale {
            ChartScale::Linear => min_loss + loss_range * (1.0 - frac),
            ChartScale::Log | ChartScale::LogLog => {
                let log_min = safe_min_loss.ln();
                let log_max = safe_max_loss.ln();
                (log_min + (log_max - log_min) * (1.0 - frac)).exp()
            }
        };
        let y_pos = chart_y + chart_height * frac as f64;
        let _ = ctx.fill_text(&format!("{:.2}", loss_val), x + 2.0, y_pos + 3.0);
    }

    // X axis labels
    for i in 0..=4 {
        let frac = i as f64 / 4.0;
        let step_val = match scale {
            ChartScale::LogLog => {
                // Logarithmic spacing for X axis in LogLog mode
                let log_min = (safe_min_step as f64).ln();
                let log_max = (safe_max_step as f64).ln();
                (log_min + (log_max - log_min) * frac).exp() as usize
            }
            _ => min_step + ((step_range as f64) * frac) as usize,
        };
        let x_pos = chart_x + chart_width * frac;
        let _ = ctx.fill_text(&format!("{}", step_val), x_pos - 10.0, chart_y + chart_height + 15.0);
    }

    // Map point to canvas coordinates
    let map_point = |step: usize, loss: f32| -> (f64, f64) {
        let x_frac = match scale {
            ChartScale::LogLog => {
                let safe_step = step.max(1) as f64;
                let log_step = safe_step.ln();
                let log_min = (safe_min_step as f64).ln();
                let log_max = (safe_max_step as f64).ln();
                if log_max > log_min {
                    ((log_step - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            _ => {
                if step_range > 0 {
                    ((step.saturating_sub(min_step)) as f64 / step_range as f64).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
        };
        let y_frac = match scale {
            ChartScale::Linear => {
                if loss_range > 0.0 {
                    ((loss - min_loss) / loss_range).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            ChartScale::Log | ChartScale::LogLog => {
                let safe_loss = loss.max(0.001);
                let log_loss = safe_loss.ln();
                let log_min = safe_min_loss.ln();
                let log_max = safe_max_loss.ln();
                if log_max > log_min {
                    ((log_loss - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
        };
        (
            chart_x + chart_width * x_frac,
            chart_y + chart_height * (1.0 - y_frac as f64),
        )
    };

    // Draw training loss line
    ctx.set_stroke_style_str("#4ecdc4");
    ctx.set_line_width(2.0);
    ctx.begin_path();
    for (i, (step, loss)) in metrics.loss_history.iter().enumerate() {
        let (px, py) = map_point(*step, *loss);
        if i == 0 {
            ctx.move_to(px, py);
        } else {
            ctx.line_to(px, py);
        }
    }
    ctx.stroke();

    // Draw validation loss line
    if !metrics.val_loss_history.is_empty() {
        ctx.set_stroke_style_str("#ff6b6b");
        ctx.set_line_width(2.0);
        ctx.begin_path();
        for (i, (step, loss)) in metrics.val_loss_history.iter().enumerate() {
            let (px, py) = map_point(*step, *loss);
            if i == 0 {
                ctx.move_to(px, py);
            } else {
                ctx.line_to(px, py);
            }
        }
        ctx.stroke();
    }

    // Legend
    ctx.set_fill_style_str("#4ecdc4");
    ctx.fill_rect(chart_x + chart_width - 100.0, chart_y + 5.0, 10.0, 3.0);
    ctx.set_fill_style_str("#e0e6ed");
    ctx.set_font("10px monospace");
    let _ = ctx.fill_text("Train", chart_x + chart_width - 85.0, chart_y + 10.0);

    ctx.set_fill_style_str("#ff6b6b");
    ctx.fill_rect(chart_x + chart_width - 100.0, chart_y + 20.0, 10.0, 3.0);
    let _ = ctx.fill_text("Val", chart_x + chart_width - 85.0, chart_y + 25.0);
}

/// Render metrics panel
pub fn render_metrics_panel(
    ctx: &web_sys::CanvasRenderingContext2d,
    metrics: &LMMetrics,
    x: f64,
    y: f64,
    width: f64,
    current_time_ms: f64,
) {
    let line_height = 22.0;
    let mut row = 0;

    ctx.set_fill_style_str("#12121a");
    ctx.fill_rect(x, y, width, line_height * 11.0);

    ctx.set_font("12px monospace");

    let draw_row = |ctx: &web_sys::CanvasRenderingContext2d, label: &str, value: &str, row: usize, color: &str| {
        ctx.set_fill_style_str("#6b7280");
        let _ = ctx.fill_text(label, x + 10.0, y + 18.0 + row as f64 * line_height);
        ctx.set_fill_style_str(color);
        let _ = ctx.fill_text(value, x + 140.0, y + 18.0 + row as f64 * line_height);
    };

    draw_row(ctx, "Step:", &format!("{}", metrics.step), row, "#e0e6ed");
    row += 1;
    draw_row(ctx, "Epoch:", &format!("{} ({:.1}%)", metrics.epoch, metrics.epoch_progress() * 100.0), row, "#e0e6ed");
    row += 1;
    draw_row(ctx, "Train Loss:", &format!("{:.4}", metrics.train_loss), row, "#4ecdc4");
    row += 1;
    draw_row(ctx, "Val Loss:", &format!("{:.4}", metrics.val_loss), row, "#ff6b6b");
    row += 1;
    draw_row(ctx, "Train PPL:", &format!("{:.2}", metrics.train_perplexity), row, "#4ecdc4");
    row += 1;
    draw_row(ctx, "Val PPL:", &format!("{:.2}", metrics.val_perplexity), row, "#ff6b6b");
    row += 1;
    draw_row(ctx, "Learning Rate:", &format!("{:.6}", metrics.learning_rate), row, "#e0e6ed");
    row += 1;
    draw_row(ctx, "Tokens:", &format_number(metrics.tokens_processed), row, "#e0e6ed");
    row += 1;
    let tps = metrics.tokens_per_second(current_time_ms);
    draw_row(ctx, "Tokens/sec:", &format!("{:.0}", tps), row, "#98c1d9");
    row += 1;
    let flops = metrics.flops_per_second(current_time_ms);
    draw_row(ctx, "FLOPS:", &LMMetrics::format_flops(flops), row, "#98c1d9");
}

/// Format large numbers with K, M suffixes
fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = LMMetrics::new();
        assert_eq!(metrics.step(), 0);
        assert_eq!(metrics.epoch(), 0);
    }

    #[test]
    fn test_metrics_update_train() {
        let mut metrics = LMMetrics::new();
        metrics.update_train(10, 2.5, 0.001, 100);

        assert_eq!(metrics.step(), 10);
        assert!((metrics.train_loss() - 2.5).abs() < 0.001);
        assert!((metrics.learning_rate() - 0.001).abs() < 0.0001);
        assert_eq!(metrics.tokens_processed(), 100);
        assert_eq!(metrics.loss_history.len(), 1);
    }

    #[test]
    fn test_metrics_update_val() {
        let mut metrics = LMMetrics::new();
        metrics.update_val(2.0);

        assert!((metrics.val_loss() - 2.0).abs() < 0.001);
        assert!((metrics.val_perplexity() - 2.0_f32.exp()).abs() < 0.1);
    }

    #[test]
    fn test_epoch_progress() {
        let mut metrics = LMMetrics::new();
        metrics.set_epoch(0, 100);
        metrics.update_train(50, 1.0, 0.001, 10);

        assert!((metrics.epoch_progress() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_tokens_per_second() {
        let mut metrics = LMMetrics::new();
        metrics.start_timer(0.0);
        metrics.update_train(1, 1.0, 0.001, 1000);

        let tps = metrics.tokens_per_second(1000.0); // 1 second later
        assert!((tps - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1500), "1.5K");
        assert_eq!(format_number(1_500_000), "1.50M");
    }

    #[test]
    fn test_metrics_reset() {
        let mut metrics = LMMetrics::new();
        metrics.update_train(100, 1.0, 0.001, 5000);
        metrics.reset();

        assert_eq!(metrics.step(), 0);
        assert_eq!(metrics.tokens_processed(), 0);
        assert!(metrics.loss_history.is_empty());
    }
}
