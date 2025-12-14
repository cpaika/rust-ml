//! Text completion sandbox with probability visualization

use serde::{Deserialize, Serialize};
use tokenizer::BpeTokenizer;
use transformer::Transformer;
use wasm_bindgen::prelude::*;

/// Token with its probability for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct TokenProb {
    /// Token ID
    id: u32,
    /// Token text
    #[wasm_bindgen(skip)]
    pub text: String,
    /// Probability (0.0 - 1.0)
    prob: f32,
}

#[wasm_bindgen]
impl TokenProb {
    /// Get token ID
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Get probability
    pub fn prob(&self) -> f32 {
        self.prob
    }

    /// Get token text
    pub fn text(&self) -> String {
        self.text.clone()
    }
}

/// Completion state for the sandbox
#[derive(Debug, Clone)]
pub struct CompletionState {
    /// Current input text
    pub input: String,
    /// Generated tokens
    pub generated_tokens: Vec<u32>,
    /// Top-k predictions with probabilities
    pub top_predictions: Vec<TokenProb>,
    /// Whether generation is in progress
    pub generating: bool,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-k for display
    pub top_k: usize,
    /// Enable partial word completion mode
    pub partial_word_completion: bool,
    /// Current partial word being typed (for display)
    pub current_partial_word: String,
}

impl Default for CompletionState {
    fn default() -> Self {
        Self {
            input: String::new(),
            generated_tokens: Vec::new(),
            top_predictions: Vec::new(),
            generating: false,
            temperature: 0.8,
            top_k: 10,
            partial_word_completion: true,
            current_partial_word: String::new(),
        }
    }
}

impl CompletionState {
    /// Create new completion state
    pub fn new() -> Self {
        Self::default()
    }

    /// Set input text
    pub fn set_input(&mut self, text: &str) {
        self.input = text.to_string();
        self.generated_tokens.clear();
        self.top_predictions.clear();
    }

    /// Generate next token predictions
    pub fn predict_next(&mut self, model: &mut Transformer, tokenizer: &BpeTokenizer) {
        // Tokenize input + generated
        let full_text = format!("{}{}", self.input, self.decode_generated(tokenizer));

        // Extract partial word for constrained completion
        let partial_word = self.extract_partial_word(&full_text);
        self.current_partial_word = partial_word.clone();

        let tokens = match tokenizer.encode(&full_text) {
            Ok(t) => t,
            Err(_) => return,
        };

        if tokens.is_empty() {
            return;
        }

        // Run forward pass
        model.forward(&tokens);

        // Get predictions - either constrained or unconstrained
        if self.partial_word_completion && !partial_word.is_empty() {
            self.predict_with_partial_word_constraint(model, tokenizer, &partial_word);
        } else {
            // Standard unconstrained prediction
            let top_k = model.top_k_next_tokens(self.top_k);
            self.top_predictions = top_k
                .into_iter()
                .map(|(id, prob)| TokenProb {
                    id,
                    text: tokenizer.decode(&[id]).unwrap_or_default(),
                    prob,
                })
                .collect();
        }
    }

    /// Extract the current partial word from the text
    /// Returns the text after the last whitespace character
    fn extract_partial_word(&self, text: &str) -> String {
        // Find the last whitespace or start of string
        if let Some(last_ws_pos) = text.rfind(|c: char| c.is_whitespace()) {
            text[last_ws_pos + 1..].to_string()
        } else {
            text.to_string()
        }
    }

    /// Generate predictions constrained to valid partial word completions
    fn predict_with_partial_word_constraint(
        &mut self,
        model: &mut Transformer,
        tokenizer: &BpeTokenizer,
        partial_word: &str,
    ) {
        // Get model's probability distribution
        let probs = model.get_next_token_probs();

        // Find tokens that could complete the partial word
        let completing_tokens = tokenizer.find_completing_tokens(partial_word);

        // Combine model probabilities with valid completions
        let mut candidates: Vec<TokenProb> = Vec::new();

        for (token_id, completion_text, priority) in completing_tokens.iter() {
            if let Some(&prob) = probs.get(*token_id as usize) {
                // Boost probability by priority (how well it matches the partial)
                let adjusted_prob = prob * priority;
                if adjusted_prob > 0.0001 {
                    // Show what the user will see when this token is added
                    candidates.push(TokenProb {
                        id: *token_id,
                        text: completion_text.clone(),
                        prob: adjusted_prob,
                    });
                }
            }
        }

        // Sort by adjusted probability
        candidates.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));

        // If we have valid completions, use them
        if !candidates.is_empty() {
            // Normalize probabilities for display
            let total: f32 = candidates.iter().map(|c| c.prob).sum();
            if total > 0.0 {
                for c in &mut candidates {
                    c.prob /= total;
                }
            }
            self.top_predictions = candidates.into_iter().take(self.top_k).collect();
        } else {
            // Fall back to standard predictions if no valid completions found
            let top_k = model.top_k_next_tokens(self.top_k);
            self.top_predictions = top_k
                .into_iter()
                .map(|(id, prob)| TokenProb {
                    id,
                    text: tokenizer.decode(&[id]).unwrap_or_default(),
                    prob,
                })
                .collect();
        }
    }

    /// Toggle partial word completion mode
    pub fn set_partial_word_completion(&mut self, enabled: bool) {
        self.partial_word_completion = enabled;
    }

    /// Sample and add a single token (internal, doesn't update predictions)
    fn sample_one(&mut self, model: &mut Transformer, tokenizer: &BpeTokenizer) -> bool {
        // Tokenize input + generated
        let full_text = format!("{}{}", self.input, self.decode_generated(tokenizer));
        let tokens = match tokenizer.encode(&full_text) {
            Ok(t) => t,
            Err(_) => return false,
        };

        if tokens.is_empty() {
            return false;
        }

        // Run forward pass and sample
        model.forward(&tokens);

        let next_token = if self.temperature > 0.0 {
            model.sample_next_token(self.temperature)
        } else {
            model.greedy_next_token()
        };

        if let Some(token) = next_token {
            // Check for EOS token (typically 3)
            if token == 3 {
                return false; // Stop generation
            }
            self.generated_tokens.push(token);
            true
        } else {
            false
        }
    }

    /// Sample and add a token (updates predictions after)
    pub fn sample_next(&mut self, model: &mut Transformer, tokenizer: &BpeTokenizer) {
        if self.sample_one(model, tokenizer) {
            // Update predictions only after successful sample
            self.predict_next(model, tokenizer);
        }
    }

    /// Generate multiple tokens efficiently (only updates predictions at the end)
    pub fn generate(&mut self, model: &mut Transformer, tokenizer: &BpeTokenizer, num_tokens: usize) {
        self.generating = true;

        for _ in 0..num_tokens {
            if !self.sample_one(model, tokenizer) {
                break; // EOS or error
            }
        }

        self.generating = false;

        // Update predictions only once at the end
        self.predict_next(model, tokenizer);
    }

    /// Decode generated tokens to text
    pub fn decode_generated(&self, tokenizer: &BpeTokenizer) -> String {
        if self.generated_tokens.is_empty() {
            return String::new();
        }
        tokenizer.decode(&self.generated_tokens).unwrap_or_default()
    }

    /// Get full text (input + generated)
    pub fn full_text(&self, tokenizer: &BpeTokenizer) -> String {
        format!("{}{}", self.input, self.decode_generated(tokenizer))
    }

    /// Clear generated tokens
    pub fn clear_generated(&mut self) {
        self.generated_tokens.clear();
        self.top_predictions.clear();
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temp: f32) {
        self.temperature = temp.max(0.0).min(2.0);
    }
}

/// Render probability bar for a token
pub fn render_token_prob_bar(
    ctx: &web_sys::CanvasRenderingContext2d,
    token: &TokenProb,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    is_top: bool,
) {
    let bar_area_start = 100.0;
    let bar_area_width = width - bar_area_start - 60.0;
    let bar_width = bar_area_width * token.prob as f64;

    // Background - subtle highlight for top prediction
    let bg_color = if is_top { "#161b22" } else { "#0d1117" };
    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(x, y, width, height);

    // Left border accent for top prediction
    if is_top {
        ctx.set_fill_style_str("#4ecdc4");
        ctx.fill_rect(x, y, 3.0, height);
    }

    // Probability bar with gradient effect
    let bar_color = probability_color(token.prob);
    ctx.set_fill_style_str(&bar_color);
    ctx.fill_rect(x + bar_area_start, y + 6.0, bar_width, height - 12.0);

    // Bar background (track)
    ctx.set_fill_style_str("#21262d");
    ctx.fill_rect(x + bar_area_start + bar_width, y + 6.0, bar_area_width - bar_width, height - 12.0);

    // Token text with rank indicator
    ctx.set_font("12px monospace");
    ctx.set_fill_style_str(if is_top { "#4ecdc4" } else { "#c9d1d9" });

    // Escape and truncate token display
    let display_text = escape_token_display(&token.text, 12);
    let _ = ctx.fill_text(&display_text, x + 8.0, y + height / 2.0 + 4.0);

    // Probability text
    ctx.set_fill_style_str(if is_top { "#4ecdc4" } else { "#8b949e" });
    ctx.set_font(if is_top { "bold 12px monospace" } else { "12px monospace" });
    let prob_text = format!("{:.1}%", token.prob * 100.0);
    let _ = ctx.fill_text(&prob_text, x + width - 50.0, y + height / 2.0 + 4.0);
}

/// Render the completion panel
pub fn render_completion_panel(
    ctx: &web_sys::CanvasRenderingContext2d,
    state: &CompletionState,
    tokenizer: &BpeTokenizer,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) {
    // Background
    ctx.set_fill_style_str("#0a0a0f");
    ctx.fill_rect(x, y, width, height);

    // Title with icon
    ctx.set_fill_style_str("#4ecdc4");
    ctx.set_font("bold 16px monospace");
    let _ = ctx.fill_text("Generated Output", x + 10.0, y + 24.0);

    // Input/output text area - larger and more prominent
    let text_area_y = y + 40.0;
    let text_area_height = 120.0;

    // Text area background with gradient-like effect
    ctx.set_fill_style_str("#0d1117");
    ctx.fill_rect(x + 10.0, text_area_y, width - 20.0, text_area_height);

    // Border
    ctx.set_stroke_style_str("#30363d");
    ctx.set_line_width(1.0);
    ctx.stroke_rect(x + 10.0, text_area_y, width - 20.0, text_area_height);

    // Render text with word wrapping
    let input_text = &state.input;
    let generated_text = state.decode_generated(tokenizer);

    ctx.set_font("14px monospace");

    // Word wrap and render text
    let max_width = width - 40.0;
    let line_height = 20.0;
    let mut current_y = text_area_y + 25.0;
    let start_x = x + 20.0;

    // Combine and wrap text
    let full_text = format!("{}{}", input_text, generated_text);
    let input_len = input_text.len();

    let mut line_start = 0;
    let mut current_x = start_x;

    for (char_idx, c) in full_text.chars().enumerate() {
        let char_width = if c == ' ' { 8.4 } else { 8.4 }; // Monospace font

        if current_x + char_width > start_x + max_width || c == '\n' {
            // Render the line
            let line = &full_text[line_start..char_idx];
            render_colored_line(ctx, line, start_x, current_y, input_len.saturating_sub(line_start), line_start);
            current_y += line_height;
            current_x = start_x;
            line_start = if c == '\n' { char_idx + 1 } else { char_idx };

            if current_y > text_area_y + text_area_height - 10.0 {
                // Truncate with ellipsis
                ctx.set_fill_style_str("#6b7280");
                let _ = ctx.fill_text("...", start_x, current_y);
                break;
            }
        }
        current_x += char_width;
    }

    // Render remaining text
    if line_start < full_text.len() && current_y <= text_area_y + text_area_height - 10.0 {
        let line = &full_text[line_start..];
        render_colored_line(ctx, line, start_x, current_y, input_len.saturating_sub(line_start), line_start);
    }

    // If no text, show placeholder
    if full_text.is_empty() {
        ctx.set_fill_style_str("#484f58");
        ctx.set_font("italic 14px monospace");
        let _ = ctx.fill_text("Type a prompt above to generate text...", start_x, text_area_y + 60.0);
    }

    // Predictions panel
    let pred_y = text_area_y + text_area_height + 25.0;

    // Section header
    ctx.set_fill_style_str("#8b949e");
    ctx.set_font("12px monospace");
    let _ = ctx.fill_text("NEXT TOKEN PREDICTIONS", x + 10.0, pred_y);

    // Underline
    ctx.set_stroke_style_str("#30363d");
    ctx.begin_path();
    ctx.move_to(x + 10.0, pred_y + 5.0);
    ctx.line_to(x + 200.0, pred_y + 5.0);
    ctx.stroke();

    // Render probability bars - improved layout
    let bar_height = 28.0;
    let bar_y_start = pred_y + 15.0;

    if state.top_predictions.is_empty() {
        ctx.set_fill_style_str("#484f58");
        ctx.set_font("12px monospace");
        let _ = ctx.fill_text("No predictions yet", x + 10.0, bar_y_start + 20.0);
    } else {
        for (i, token) in state.top_predictions.iter().take(10).enumerate() {
            render_token_prob_bar(
                ctx,
                token,
                x + 10.0,
                bar_y_start + i as f64 * bar_height,
                width - 20.0,
                bar_height - 2.0,
                i == 0, // Highlight top prediction
            );
        }
    }

    // Stats footer
    let footer_y = bar_y_start + 10.0 * bar_height + 15.0;
    ctx.set_fill_style_str("#484f58");
    ctx.set_font("11px monospace");
    let stats = format!(
        "Generated: {} tokens  |  Temperature: {:.2}  |  Top-k: {}",
        state.generated_tokens.len(),
        state.temperature,
        state.top_k
    );
    let _ = ctx.fill_text(&stats, x + 10.0, footer_y);
}

/// Render a line with input (white) and generated (teal) portions
fn render_colored_line(
    ctx: &web_sys::CanvasRenderingContext2d,
    line: &str,
    x: f64,
    y: f64,
    input_chars_remaining: usize,
    line_start_in_full: usize,
) {
    ctx.set_font("14px monospace");
    let char_width = 8.4;

    let mut current_x = x;
    for (i, c) in line.chars().enumerate() {
        let global_idx = line_start_in_full + i;
        let color = if global_idx < input_chars_remaining + line_start_in_full {
            "#c9d1d9" // Input: light gray
        } else {
            "#4ecdc4" // Generated: teal
        };
        ctx.set_fill_style_str(color);
        let _ = ctx.fill_text(&c.to_string(), current_x, y);
        current_x += char_width;
    }
}

/// Map probability to color (red for low, green for high)
fn probability_color(prob: f32) -> String {
    if prob > 0.5 {
        "#4ecdc4".to_string() // Teal for high prob
    } else if prob > 0.2 {
        "#98c1d9".to_string() // Light blue for medium
    } else if prob > 0.05 {
        "#e0e6ed".to_string() // Light gray for low
    } else {
        "#6b7280".to_string() // Dark gray for very low
    }
}

/// Escape special characters in token display
fn escape_token_display(text: &str, max_len: usize) -> String {
    let escaped: String = text
        .chars()
        .map(|c| match c {
            '\n' => '\\'.to_string() + "n",
            '\t' => '\\'.to_string() + "t",
            '\r' => '\\'.to_string() + "r",
            ' ' => '\u{2423}'.to_string(), // Open box for space
            c if c.is_control() => format!("\\x{:02x}", c as u32),
            c => c.to_string(),
        })
        .collect();

    if escaped.len() > max_len {
        format!("{}...", &escaped[..max_len - 3])
    } else {
        escaped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizer::{BpeTokenizer, BpeConfig};
    use transformer::{Transformer, TransformerConfig};
    use std::time::Instant;

    fn create_test_model_and_tokenizer() -> (Transformer, BpeTokenizer) {
        // Use a larger vocab to match what the tokenizer produces
        let vocab_size = 500;

        let config = TransformerConfig {
            vocab_size,
            d_model: 32,
            n_heads: 2,
            n_layers: 1,
            d_ff: 64,
            max_seq_len: 64,
            dropout: 0.0,
            ..Default::default()
        };
        let model = Transformer::new(config).expect("Should create model");

        let bpe_config = BpeConfig {
            vocab_size,
            ..Default::default()
        };
        let mut tokenizer = BpeTokenizer::new(bpe_config);
        // Train on simple repeating text
        let training_text = "a b c d e f g h i j k l m n o p q r s t u v w x y z ";
        tokenizer.train(&[&training_text.repeat(10)]).unwrap();

        (model, tokenizer)
    }

    #[test]
    fn test_generation_completes_in_reasonable_time() {
        // This test ensures that generating tokens doesn't take too long
        // which would cause UI freezes when the slider is adjusted
        let (mut model, tokenizer) = create_test_model_and_tokenizer();

        let mut state = CompletionState::new();
        state.set_input("a b c");

        // Test generating 50 tokens (max slider value)
        let start = Instant::now();
        state.generate(&mut model, &tokenizer, 50);
        let elapsed = start.elapsed();

        // Generation should complete in under 2 seconds to not freeze UI
        // On a typical machine this should be much faster
        assert!(
            elapsed.as_secs() < 2,
            "Generation of 50 tokens took {:?}, which is too slow and will freeze the UI",
            elapsed
        );

        // Verify tokens were actually generated
        assert!(!state.generated_tokens.is_empty(), "Should have generated some tokens");
    }

    #[test]
    fn test_rapid_slider_changes_dont_accumulate() {
        // Simulates what happens when user drags slider rapidly
        // Each change should not regenerate from scratch
        let (mut model, tokenizer) = create_test_model_and_tokenizer();

        let mut state = CompletionState::new();
        state.set_input("a b");

        // Simulate rapid slider changes (10 -> 20 -> 30 -> 40 -> 50)
        let start = Instant::now();
        for num_tokens in [10, 20, 30, 40, 50] {
            state.set_input("a b"); // This clears generated tokens
            state.generate(&mut model, &tokenizer, num_tokens);
        }
        let elapsed = start.elapsed();

        // All these regenerations should complete in reasonable time
        assert!(
            elapsed.as_secs() < 5,
            "Rapid slider changes took {:?}, which is too slow",
            elapsed
        );
    }

    #[test]
    fn test_completion_state_new() {
        let state = CompletionState::new();
        assert!(state.input.is_empty());
        assert!(state.generated_tokens.is_empty());
        assert!((state.temperature - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_set_input() {
        let mut state = CompletionState::new();
        state.set_input("Hello world");
        assert_eq!(state.input, "Hello world");
    }

    #[test]
    fn test_set_temperature() {
        let mut state = CompletionState::new();

        state.set_temperature(1.5);
        assert!((state.temperature - 1.5).abs() < 0.001);

        // Clamp to valid range
        state.set_temperature(-0.5);
        assert!((state.temperature - 0.0).abs() < 0.001);

        state.set_temperature(3.0);
        assert!((state.temperature - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_clear_generated() {
        let mut state = CompletionState::new();
        state.generated_tokens.push(1);
        state.generated_tokens.push(2);
        state.top_predictions.push(TokenProb {
            id: 1,
            text: "test".to_string(),
            prob: 0.5,
        });

        state.clear_generated();
        assert!(state.generated_tokens.is_empty());
        assert!(state.top_predictions.is_empty());
    }

    #[test]
    fn test_escape_token_display() {
        assert_eq!(escape_token_display("\n", 10), "\\n");
        assert_eq!(escape_token_display(" ", 10), "\u{2423}");
        assert_eq!(escape_token_display("hello", 10), "hello");
        // max_len=10: "verylon" (7 chars) + "..." (3 chars) = 10 chars total
        assert_eq!(escape_token_display("verylongtoken", 10), "verylon...");
    }

    #[test]
    fn test_probability_color() {
        let high = probability_color(0.8);
        assert_eq!(high, "#4ecdc4");

        let low = probability_color(0.01);
        assert_eq!(low, "#6b7280");
    }

    #[test]
    fn test_token_prob_accessors() {
        let tp = TokenProb {
            id: 42,
            text: "test".to_string(),
            prob: 0.75,
        };

        assert_eq!(tp.id(), 42);
        assert_eq!(tp.text(), "test");
        assert!((tp.prob() - 0.75).abs() < 0.001);
    }
}
