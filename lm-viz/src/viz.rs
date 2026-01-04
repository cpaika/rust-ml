//! Main visualization module for the language model app

use crate::completion::{render_completion_panel, CompletionState};
use crate::metrics::{render_loss_chart, render_metrics_panel, ChartScale, LMMetrics};
use tokenizer::{BpeTokenizer, BpeConfig};
use transformer::training::{TextDataset, TrainingConfig, LearningRateScheduler};
use transformer::{Transformer, TransformerConfig};
use transformer::gpu::GpuContext;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, MouseEvent,
};
use std::cell::RefCell;
use std::rc::Rc;
use serde::{Deserialize, Serialize};

/// Saved model state for checkpoint files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedModelState {
    /// Model configuration
    pub config: TransformerConfig,
    /// Model weights (the entire transformer)
    pub model: Transformer,
    /// Tokenizer
    pub tokenizer: BpeTokenizer,
    /// Training metrics
    pub metrics: LMMetrics,
    /// Current epoch
    pub epoch: usize,
    /// Current step
    pub step: usize,
}

// Color palette
const BG_COLOR: &str = "#0a0a0f";
const PANEL_BG: &str = "#12121a";
const ACCENT_COLOR: &str = "#4ecdc4";
const ACCENT_SECONDARY: &str = "#ff6b6b";
const TEXT_COLOR: &str = "#e0e6ed";
const TEXT_DIM: &str = "#6b7280";
const NEURON_STROKE: &str = "#3d5a80";

// Layout constants
const SIDEBAR_WIDTH: f64 = 320.0;
const HEADER_HEIGHT: f64 = 50.0;

/// Training state
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrainingState {
    Idle,
    Training,
    Paused,
}

/// Application mode
#[derive(Clone, Copy, PartialEq, Eq)]
enum AppMode {
    Training,
    Completion,
}

/// Accelerator mode for training
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AcceleratorMode {
    Cpu,
    Gpu,
}

/// GPU initialization state
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GpuInitState {
    NotStarted,
    Initializing,
    Ready,
    Failed,
}

/// Calculate approximate FLOPs for a transformer forward + backward pass
/// Formula based on "Scaling Laws for Neural Language Models" (Kaplan et al.)
/// FLOPs ≈ 6 * N * seq_len for forward+backward, where N is number of parameters
fn calculate_transformer_flops(config: &TransformerConfig, seq_len: usize) -> usize {
    let d_model = config.d_model;
    let n_layers = config.n_layers;
    let d_ff = config.d_ff;
    let vocab_size = config.vocab_size;

    // Embedding: vocab_size * d_model (lookup, no multiply)
    // Per layer:
    //   - Attention: 4 * d_model^2 (Q, K, V, O projections) * seq_len
    //   - Attention scores: seq_len^2 * d_model
    //   - FFN: 2 * d_model * d_ff * seq_len
    // Output projection: d_model * vocab_size * seq_len

    let per_layer = (4 * d_model * d_model + 2 * d_model * d_ff) * seq_len;
    let attention_scores = seq_len * seq_len * d_model * n_layers;
    let embedding_and_output = d_model * vocab_size * seq_len;

    // Multiply by 2 for forward+backward, and 3 for compute vs memory operations
    let total = (per_layer * n_layers + attention_scores + embedding_and_output) * 6;
    total
}

/// Main application state
struct AppState {
    // Model and tokenizer
    model: Option<Transformer>,
    tokenizer: Option<BpeTokenizer>,
    config: TransformerConfig,

    // Training
    training_state: TrainingState,
    training_data: Option<TextDataset>,
    val_data: Option<TextDataset>,
    lr_scheduler: Option<LearningRateScheduler>,
    metrics: LMMetrics,
    batches_per_frame: usize,

    // Completion
    completion: CompletionState,
    num_tokens_to_generate: usize,

    // UI state
    mode: AppMode,
    chart_scale: ChartScale,
    canvas_width: f64,
    canvas_height: f64,

    // GPU acceleration
    accelerator_mode: AcceleratorMode,
    gpu_available: bool,
    gpu_init_state: GpuInitState,
}

impl AppState {
    fn new(canvas_width: f64, canvas_height: f64, gpu_available: bool) -> Self {
        // Use a small config suitable for browser training
        let config = TransformerConfig::tiny(1000);

        // Default to GPU if available
        let accelerator_mode = if gpu_available {
            AcceleratorMode::Gpu
        } else {
            AcceleratorMode::Cpu
        };

        Self {
            model: None,
            tokenizer: None,
            config,
            training_state: TrainingState::Idle,
            training_data: None,
            val_data: None,
            lr_scheduler: None,
            metrics: LMMetrics::new(),
            batches_per_frame: 1,
            completion: CompletionState::new(),
            num_tokens_to_generate: 10,
            mode: AppMode::Training,
            chart_scale: ChartScale::Linear,
            canvas_width,
            canvas_height,
            accelerator_mode,
            gpu_available,
            gpu_init_state: GpuInitState::NotStarted,
        }
    }

    /// Initialize model and tokenizer with training data
    fn initialize(&mut self, text: &str) -> Result<(), String> {
        // Create and train tokenizer
        let bpe_config = BpeConfig {
            vocab_size: self.config.vocab_size,
            ..Default::default()
        };
        let mut tokenizer = BpeTokenizer::new(bpe_config);
        tokenizer.train(&[text]).map_err(|e| e.to_string())?;

        // Tokenize text
        let tokens = tokenizer.encode(text).map_err(|e| e.to_string())?;
        if tokens.len() < self.config.max_seq_len * 2 {
            return Err("Not enough training data".to_string());
        }

        // Split into train/val
        let split_point = (tokens.len() as f32 * 0.9) as usize;
        let train_tokens = tokens[..split_point].to_vec();
        let val_tokens = tokens[split_point..].to_vec();

        // Create datasets
        let training_config = TrainingConfig::browser_friendly();
        let train_dataset = TextDataset::new(train_tokens, training_config.seq_length);
        let val_dataset = TextDataset::new(val_tokens, training_config.seq_length);

        // Calculate steps
        let steps_per_epoch = train_dataset.len() / training_config.seq_length;
        let total_steps = steps_per_epoch * training_config.epochs;

        // Create model
        let model = Transformer::new(self.config.clone())?;

        // Create LR scheduler
        let lr_scheduler = LearningRateScheduler::new(
            training_config.learning_rate,
            training_config.warmup_steps,
            total_steps,
        );

        // Update state
        self.tokenizer = Some(tokenizer);
        self.model = Some(model);
        self.training_data = Some(train_dataset);
        self.val_data = Some(val_dataset);
        self.lr_scheduler = Some(lr_scheduler);
        self.metrics.set_epoch(0, steps_per_epoch);

        // Calculate and set FLOPs per step
        let flops_per_step = calculate_transformer_flops(&self.config, training_config.seq_length);
        self.metrics.set_flops_per_step(flops_per_step);

        log(&format!(
            "Initialized with {} tokens, {} vocab, {} params, {} FLOPS/step",
            tokens.len(),
            self.config.vocab_size,
            self.model.as_ref().map(|m| m.num_parameters()).unwrap_or(0),
            LMMetrics::format_flops(flops_per_step as f64)
        ));

        Ok(())
    }

    /// Run a single training step
    fn train_step(&mut self) -> Option<f32> {
        let model = self.model.as_mut()?;
        let dataset = self.training_data.as_mut()?;
        let scheduler = self.lr_scheduler.as_mut()?;

        // Get next batch
        let (input, target) = dataset.next_example()?;

        // Get learning rate
        let lr = scheduler.get_lr();

        // Forward pass
        let loss = model.compute_loss(&input, &target);

        // Backward pass with full propagation (includes weight updates)
        model.backward(&target, lr);

        // Update metrics
        let step = scheduler.current_step();
        scheduler.step();

        self.metrics.update_train(step, loss, lr, input.len());

        Some(loss)
    }

    /// Run validation
    fn evaluate(&mut self) -> Option<f32> {
        let model = self.model.as_mut()?;
        let val_data = self.val_data.as_mut()?;

        // Sample a few validation batches
        let mut total_loss = 0.0;
        let mut count = 0;

        val_data.reset();
        for _ in 0..10 {
            if let Some((input, target)) = val_data.next_example() {
                total_loss += model.compute_loss(&input, &target);
                count += 1;
            }
        }

        if count > 0 {
            let avg_loss = total_loss / count as f32;
            self.metrics.update_val(avg_loss);
            Some(avg_loss)
        } else {
            None
        }
    }

    /// Generate completion
    fn generate_completion(&mut self, num_tokens: usize) {
        if let (Some(model), Some(tokenizer)) = (self.model.as_mut(), self.tokenizer.as_ref()) {
            self.completion.generate(model, tokenizer, num_tokens);
        }
    }
}

// Global state (needed for WASM callbacks)
thread_local! {
    static STATE: RefCell<Option<AppState>> = const { RefCell::new(None) };
    static CANVAS: RefCell<Option<HtmlCanvasElement>> = const { RefCell::new(None) };
    static CTX: RefCell<Option<CanvasRenderingContext2d>> = const { RefCell::new(None) };
    // Debounce timeout handle for slider changes
    static DEBOUNCE_HANDLE: RefCell<Option<i32>> = const { RefCell::new(None) };
}

/// Check if WebGPU is available in the browser
fn check_webgpu_available() -> bool {
    if let Some(window) = web_sys::window() {
        let navigator = window.navigator();
        // Check if gpu property exists on navigator
        let gpu = js_sys::Reflect::get(&navigator, &"gpu".into());
        if let Ok(gpu_val) = gpu {
            return !gpu_val.is_undefined() && !gpu_val.is_null();
        }
    }
    false
}

/// Initialize the visualization
pub fn init() -> Result<(), JsValue> {
    log("Initializing LM Visualization");

    let window = web_sys::window().ok_or("No window")?;
    let document = window.document().ok_or("No document")?;

    // Check GPU availability
    let gpu_available = check_webgpu_available();
    log(&format!("WebGPU available: {}", gpu_available));

    // Create canvas
    let canvas = create_canvas(&document)?;
    let ctx = get_context(&canvas)?;

    let width = canvas.width() as f64;
    let height = canvas.height() as f64;

    // Initialize state with GPU availability
    let state = AppState::new(width, height, gpu_available);

    // Store globally
    STATE.with(|s| *s.borrow_mut() = Some(state));
    CANVAS.with(|c| *c.borrow_mut() = Some(canvas.clone()));
    CTX.with(|c| *c.borrow_mut() = Some(ctx));

    // Set up event handlers
    setup_event_handlers(&canvas)?;

    // Create UI buttons
    create_ui(&document)?;

    // Initial render
    render();

    // Start animation loop
    start_animation_loop()?;

    // Auto-initialize GPU if available and GPU mode is default
    if gpu_available {
        init_gpu_async(document.clone());
    }

    Ok(())
}

/// Initialize GPU asynchronously
fn init_gpu_async(document: Document) {
    use wasm_bindgen_futures::spawn_local;

    // Mark as initializing
    STATE.with(|state| {
        if let Some(ref mut s) = *state.borrow_mut() {
            s.gpu_init_state = GpuInitState::Initializing;
        }
    });
    let _ = update_gpu_button(&document);

    spawn_local(async move {
        // Try to initialize GPU context
        match GpuContext::new_async().await {
            Ok(_ctx) => {
                // GPU initialized successfully
                STATE.with(|state| {
                    if let Some(ref mut s) = *state.borrow_mut() {
                        s.gpu_init_state = GpuInitState::Ready;
                    }
                });
                web_sys::console::log_1(&"GPU initialized successfully".into());
            }
            Err(e) => {
                // GPU init failed, fallback to CPU
                STATE.with(|state| {
                    if let Some(ref mut s) = *state.borrow_mut() {
                        s.gpu_init_state = GpuInitState::Failed;
                        s.accelerator_mode = AcceleratorMode::Cpu;
                    }
                });
                web_sys::console::error_1(&format!("GPU init failed, using CPU: {}", e).into());
            }
        }
        let _ = update_gpu_button(&document);
    });
}

/// Create the main canvas
fn create_canvas(document: &Document) -> Result<HtmlCanvasElement, JsValue> {
    let canvas: HtmlCanvasElement = document
        .create_element("canvas")?
        .dyn_into()?;

    canvas.set_id("lm-canvas");
    canvas.set_width(1200);
    canvas.set_height(800);

    let style = canvas.style();
    style.set_property("display", "block")?;
    style.set_property("margin", "0 auto")?;
    style.set_property("background", BG_COLOR)?;

    document.body().ok_or("No body")?.append_child(&canvas)?;

    Ok(canvas)
}

/// Get 2D rendering context
fn get_context(canvas: &HtmlCanvasElement) -> Result<CanvasRenderingContext2d, JsValue> {
    let obj = canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("No context"))?;
    obj.dyn_into().map_err(|e: js_sys::Object| JsValue::from(e))
}

/// Create UI buttons and controls
fn create_ui(document: &Document) -> Result<(), JsValue> {
    // Main container for all controls at the top
    let main_container: HtmlElement = document.create_element("div")?.dyn_into()?;
    main_container.set_id("main-controls");
    main_container.set_attribute("style", &format!(
        "background: {}; padding: 15px 20px; display: flex; flex-direction: column; gap: 10px;",
        PANEL_BG
    ))?;

    // Top row: Title, main buttons, accelerator toggle
    let top_row: HtmlElement = document.create_element("div")?.dyn_into()?;
    top_row.set_attribute("style", "display: flex; align-items: center; gap: 15px; flex-wrap: wrap;")?;

    // Title
    let title: HtmlElement = document.create_element("h1")?.dyn_into()?;
    title.set_inner_text("Language Model Trainer");
    title.set_attribute("style", &format!(
        "margin: 0; font-size: 18px; font-weight: bold; color: {}; font-family: monospace;",
        TEXT_COLOR
    ))?;
    top_row.append_child(&title)?;

    // Separator
    let sep: HtmlElement = document.create_element("div")?.dyn_into()?;
    sep.set_attribute("style", &format!("width: 1px; height: 24px; background: {};", NEURON_STROKE))?;
    top_row.append_child(&sep)?;

    // Initialize button
    let init_btn = create_button(document, "Initialize", "init-btn")?;
    let init_closure = Closure::wrap(Box::new(move || {
        initialize_with_sample_data();
    }) as Box<dyn Fn()>);
    init_btn.set_onclick(Some(init_closure.as_ref().unchecked_ref()));
    init_closure.forget();
    top_row.append_child(&init_btn)?;

    // Train button
    let train_btn = create_button(document, "Train", "train-btn")?;
    let train_closure = Closure::wrap(Box::new(move || {
        toggle_training();
    }) as Box<dyn Fn()>);
    train_btn.set_onclick(Some(train_closure.as_ref().unchecked_ref()));
    train_closure.forget();
    top_row.append_child(&train_btn)?;

    // Mode toggle
    let mode_btn = create_button(document, "Completion", "mode-btn")?;
    let mode_closure = Closure::wrap(Box::new(move || {
        toggle_mode();
    }) as Box<dyn Fn()>);
    mode_btn.set_onclick(Some(mode_closure.as_ref().unchecked_ref()));
    mode_closure.forget();
    top_row.append_child(&mode_btn)?;

    // Separator
    let sep2: HtmlElement = document.create_element("div")?.dyn_into()?;
    sep2.set_attribute("style", &format!("width: 1px; height: 24px; background: {};", NEURON_STROKE))?;
    top_row.append_child(&sep2)?;

    // Save Model button
    let save_btn: HtmlElement = document.create_element("button")?.dyn_into()?;
    save_btn.set_id("save-btn");
    save_btn.set_inner_text("💾 Save");
    save_btn.set_attribute("style", &format!(
        "padding: 6px 12px; font-size: 12px; cursor: pointer; \
        border-radius: 4px; border: 1px solid {}; \
        background: #1a2a3a; color: {}; font-weight: 500;",
        NEURON_STROKE, TEXT_COLOR
    ))?;
    let save_closure = Closure::wrap(Box::new(move || {
        save_model();
    }) as Box<dyn Fn()>);
    save_btn.set_onclick(Some(save_closure.as_ref().unchecked_ref()));
    save_closure.forget();
    top_row.append_child(&save_btn)?;

    // Load Model button
    let load_btn: HtmlElement = document.create_element("button")?.dyn_into()?;
    load_btn.set_id("load-btn");
    load_btn.set_inner_text("📂 Load");
    load_btn.set_attribute("style", &format!(
        "padding: 6px 12px; font-size: 12px; cursor: pointer; \
        border-radius: 4px; border: 1px solid {}; \
        background: #1a2a3a; color: {}; font-weight: 500;",
        NEURON_STROKE, TEXT_COLOR
    ))?;
    let load_closure = Closure::wrap(Box::new(move || {
        trigger_load_model();
    }) as Box<dyn Fn()>);
    load_btn.set_onclick(Some(load_closure.as_ref().unchecked_ref()));
    load_closure.forget();
    top_row.append_child(&load_btn)?;

    // Hidden file input for loading models
    let file_input: web_sys::HtmlInputElement = document.create_element("input")?.dyn_into()?;
    file_input.set_id("model-file-input");
    file_input.set_type("file");
    file_input.set_accept(".json");
    file_input.set_attribute("style", "display: none;")?;
    let file_change_closure = Closure::wrap(Box::new(move || {
        handle_model_file_selected();
    }) as Box<dyn Fn()>);
    file_input.add_event_listener_with_callback("change", file_change_closure.as_ref().unchecked_ref())?;
    file_change_closure.forget();
    top_row.append_child(&file_input)?;

    // GPU/CPU toggle container
    let accel_container: HtmlElement = document.create_element("div")?.dyn_into()?;
    accel_container.set_attribute("style", "display: flex; gap: 0; align-items: center; margin-left: auto;")?;

    // GPU button
    let gpu_btn: HtmlElement = document.create_element("button")?.dyn_into()?;
    gpu_btn.set_id("gpu-btn");
    gpu_btn.set_inner_text("GPU");
    gpu_btn.set_attribute("style", &format!(
        "padding: 6px 12px; font-size: 12px; cursor: pointer; \
        border-radius: 4px 0 0 4px; border: 1px solid {}; \
        background: {}; color: {}; font-weight: 600;",
        NEURON_STROKE, ACCENT_COLOR, BG_COLOR
    ))?;

    let doc_clone = document.clone();
    let gpu_closure = Closure::wrap(Box::new(move || {
        toggle_accelerator(AcceleratorMode::Gpu);
        let _ = update_gpu_button(&doc_clone);
    }) as Box<dyn Fn()>);
    gpu_btn.set_onclick(Some(gpu_closure.as_ref().unchecked_ref()));
    gpu_closure.forget();
    accel_container.append_child(&gpu_btn)?;

    // CPU button
    let cpu_btn: HtmlElement = document.create_element("button")?.dyn_into()?;
    cpu_btn.set_id("cpu-btn");
    cpu_btn.set_inner_text("CPU");
    cpu_btn.set_attribute("style", &format!(
        "padding: 6px 12px; font-size: 12px; cursor: pointer; \
        border-radius: 0 4px 4px 0; border: 1px solid {}; \
        background: #1a2a3a; color: {}; font-weight: 500;",
        NEURON_STROKE, TEXT_COLOR
    ))?;

    let doc_clone = document.clone();
    let cpu_closure = Closure::wrap(Box::new(move || {
        toggle_accelerator(AcceleratorMode::Cpu);
        let _ = update_gpu_button(&doc_clone);
    }) as Box<dyn Fn()>);
    cpu_btn.set_onclick(Some(cpu_closure.as_ref().unchecked_ref()));
    cpu_closure.forget();
    accel_container.append_child(&cpu_btn)?;

    // GPU Status indicator
    let gpu_status: HtmlElement = document.create_element("span")?.dyn_into()?;
    gpu_status.set_id("gpu-status");
    gpu_status.set_attribute("style", &format!(
        "color: {}; font-size: 10px; padding: 2px 8px; margin-left: 5px;",
        TEXT_DIM
    ))?;
    accel_container.append_child(&gpu_status)?;

    top_row.append_child(&accel_container)?;
    main_container.append_child(&top_row)?;

    // Second row: Chart scale buttons, completion controls
    let second_row: HtmlElement = document.create_element("div")?.dyn_into()?;
    second_row.set_id("second-row");
    second_row.set_attribute("style", "display: flex; align-items: center; gap: 15px; flex-wrap: wrap;")?;

    // Chart scale buttons (shown in training mode)
    let scale_container: HtmlElement = document.create_element("div")?.dyn_into()?;
    scale_container.set_id("scale-controls");
    scale_container.set_attribute("style", "display: flex; gap: 0; align-items: center;")?;

    let scale_label: HtmlElement = document.create_element("span")?.dyn_into()?;
    scale_label.set_inner_text("Scale:");
    scale_label.set_attribute("style", &format!(
        "color: {}; font-size: 12px; margin-right: 8px; font-family: monospace;",
        TEXT_DIM
    ))?;
    scale_container.append_child(&scale_label)?;

    // Linear button
    let linear_btn: HtmlElement = document.create_element("button")?.dyn_into()?;
    linear_btn.set_id("scale-linear");
    linear_btn.set_inner_text("Linear");
    linear_btn.set_attribute("style", &format!(
        "padding: 4px 10px; font-size: 11px; cursor: pointer; \
        border-radius: 4px 0 0 4px; border: 1px solid {}; \
        background: {}; color: {}; font-weight: 600;",
        NEURON_STROKE, ACCENT_COLOR, BG_COLOR
    ))?;

    let doc_clone = document.clone();
    let linear_closure = Closure::wrap(Box::new(move || {
        set_chart_scale(ChartScale::Linear);
        let _ = update_scale_buttons(&doc_clone);
    }) as Box<dyn Fn()>);
    linear_btn.set_onclick(Some(linear_closure.as_ref().unchecked_ref()));
    linear_closure.forget();
    scale_container.append_child(&linear_btn)?;

    // Log button
    let log_btn: HtmlElement = document.create_element("button")?.dyn_into()?;
    log_btn.set_id("scale-log");
    log_btn.set_inner_text("Log");
    log_btn.set_attribute("style", &format!(
        "padding: 4px 10px; font-size: 11px; cursor: pointer; \
        border-radius: 0; border: 1px solid {}; border-left: none; \
        background: #1a2a3a; color: {}; font-weight: 500;",
        NEURON_STROKE, TEXT_COLOR
    ))?;

    let doc_clone = document.clone();
    let log_closure = Closure::wrap(Box::new(move || {
        set_chart_scale(ChartScale::Log);
        let _ = update_scale_buttons(&doc_clone);
    }) as Box<dyn Fn()>);
    log_btn.set_onclick(Some(log_closure.as_ref().unchecked_ref()));
    log_closure.forget();
    scale_container.append_child(&log_btn)?;

    // LogLog button
    let loglog_btn: HtmlElement = document.create_element("button")?.dyn_into()?;
    loglog_btn.set_id("scale-loglog");
    loglog_btn.set_inner_text("Log-Log");
    loglog_btn.set_attribute("style", &format!(
        "padding: 4px 10px; font-size: 11px; cursor: pointer; \
        border-radius: 0 4px 4px 0; border: 1px solid {}; border-left: none; \
        background: #1a2a3a; color: {}; font-weight: 500;",
        NEURON_STROKE, TEXT_COLOR
    ))?;

    let doc_clone = document.clone();
    let loglog_closure = Closure::wrap(Box::new(move || {
        set_chart_scale(ChartScale::LogLog);
        let _ = update_scale_buttons(&doc_clone);
    }) as Box<dyn Fn()>);
    loglog_btn.set_onclick(Some(loglog_closure.as_ref().unchecked_ref()));
    loglog_closure.forget();
    scale_container.append_child(&loglog_btn)?;

    second_row.append_child(&scale_container)?;

    // Completion controls (shown in completion mode)
    let completion_controls: HtmlElement = document.create_element("div")?.dyn_into()?;
    completion_controls.set_id("completion-controls");
    completion_controls.set_attribute("style", "display: none; align-items: center; gap: 10px; flex: 1;")?;

    // Note: The main prompt input is now in the completions panel overlay (textarea)
    // This control bar just has sliders and buttons

    // Token count slider label
    let slider_label: HtmlElement = document.create_element("span")?.dyn_into()?;
    slider_label.set_id("token-count-label");
    slider_label.set_inner_text("Tokens: 10");
    slider_label.set_attribute("style", &format!(
        "color: {}; font-size: 12px; font-family: monospace; white-space: nowrap;",
        TEXT_DIM
    ))?;
    completion_controls.append_child(&slider_label)?;

    // Token count slider
    let slider: web_sys::HtmlInputElement = document.create_element("input")?.dyn_into()?;
    slider.set_id("token-slider");
    slider.set_type("range");
    slider.set_min("1");
    slider.set_max("50");
    slider.set_value("10");
    slider.set_attribute("style", "width: 100px; cursor: pointer;")?;

    let slider_change = Closure::wrap(Box::new(move || {
        on_token_slider_change();
    }) as Box<dyn Fn()>);
    slider.add_event_listener_with_callback("input", slider_change.as_ref().unchecked_ref())?;
    slider_change.forget();

    completion_controls.append_child(&slider)?;

    // Temperature label
    let temp_label: HtmlElement = document.create_element("span")?.dyn_into()?;
    temp_label.set_id("temp-label");
    temp_label.set_inner_text("Temp: 0.8");
    temp_label.set_attribute("style", &format!(
        "color: {}; font-size: 12px; font-family: monospace; white-space: nowrap;",
        TEXT_DIM
    ))?;
    completion_controls.append_child(&temp_label)?;

    // Temperature slider
    let temp_slider: web_sys::HtmlInputElement = document.create_element("input")?.dyn_into()?;
    temp_slider.set_id("temp-slider");
    temp_slider.set_type("range");
    temp_slider.set_min("0");
    temp_slider.set_max("200");
    temp_slider.set_value("80");
    temp_slider.set_attribute("style", "width: 80px; cursor: pointer;")?;

    let temp_change = Closure::wrap(Box::new(move || {
        on_temp_slider_change();
    }) as Box<dyn Fn()>);
    temp_slider.add_event_listener_with_callback("input", temp_change.as_ref().unchecked_ref())?;
    temp_change.forget();

    completion_controls.append_child(&temp_slider)?;

    // Regenerate button
    let regen_btn = create_button(document, "Generate", "regen-btn")?;
    regen_btn.style().set_property("padding", "6px 12px")?;
    regen_btn.style().set_property("font-size", "12px")?;
    let regen_closure = Closure::wrap(Box::new(move || {
        regenerate_completion();
    }) as Box<dyn Fn()>);
    regen_btn.set_onclick(Some(regen_closure.as_ref().unchecked_ref()));
    regen_closure.forget();
    completion_controls.append_child(&regen_btn)?;

    // Clear button
    let clear_btn: HtmlElement = document.create_element("button")?.dyn_into()?;
    clear_btn.set_id("clear-btn");
    clear_btn.set_inner_text("Clear");
    clear_btn.style().set_property("padding", "6px 12px")?;
    clear_btn.style().set_property("background", "#30363d")?;
    clear_btn.style().set_property("color", TEXT_COLOR)?;
    clear_btn.style().set_property("border", "none")?;
    clear_btn.style().set_property("border-radius", "4px")?;
    clear_btn.style().set_property("cursor", "pointer")?;
    clear_btn.style().set_property("font-size", "12px")?;
    let clear_closure = Closure::wrap(Box::new(move || {
        clear_completion();
    }) as Box<dyn Fn()>);
    clear_btn.set_onclick(Some(clear_closure.as_ref().unchecked_ref()));
    clear_closure.forget();
    completion_controls.append_child(&clear_btn)?;

    second_row.append_child(&completion_controls)?;
    main_container.append_child(&second_row)?;

    // Insert at the beginning of body (before canvas)
    let body = document.body().ok_or("No body")?;
    body.insert_before(&main_container, body.first_child().as_ref())?;

    // Create completions panel overlay (positioned over the main content area)
    create_completions_panel(document)?;

    // Initialize button states
    update_gpu_button(document)?;
    update_scale_buttons(document)?;
    update_mode_controls();

    Ok(())
}

/// Clear completion text
fn clear_completion() {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.completion.set_input("");
            state.completion.clear_generated();
        }
    });

    // Clear the textarea
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(textarea) = document.get_element_by_id("completion-input") {
                if let Ok(ta) = textarea.dyn_into::<web_sys::HtmlTextAreaElement>() {
                    ta.set_value("");
                }
            }
            // Clear output
            if let Some(output) = document.get_element_by_id("completion-output") {
                output.set_inner_html("");
            }
            // Clear predictions
            if let Some(pred) = document.get_element_by_id("predictions-container") {
                pred.set_inner_html("<div style=\"color: #484f58; font-size: 12px;\">Type something to see predictions...</div>");
            }
        }
    }
}

/// Regenerate completion from current input
fn regenerate_completion() {
    do_generate_completion();
}

/// Create the completions panel overlay
fn create_completions_panel(document: &Document) -> Result<(), JsValue> {
    // Main panel container - positioned over the content area
    let panel: HtmlElement = document.create_element("div")?.dyn_into()?;
    panel.set_id("completions-panel");
    panel.set_attribute("style", &format!(
        "display: none; \
        position: absolute; \
        left: {}px; \
        top: 130px; \
        right: 20px; \
        bottom: 20px; \
        background: {}; \
        border: 1px solid #30363d; \
        border-radius: 8px; \
        flex-direction: column; \
        gap: 12px; \
        padding: 16px; \
        overflow: hidden;",
        SIDEBAR_WIDTH as i32 + 20,
        BG_COLOR
    ))?;

    // Title section
    let title: HtmlElement = document.create_element("div")?.dyn_into()?;
    title.set_inner_html(&format!(
        "<span style=\"color: {}; font-size: 16px; font-weight: bold; font-family: monospace;\">✨ Text Completion</span>\
        <span style=\"color: {}; font-size: 12px; margin-left: 12px;\">Type your prompt and watch the model generate text</span>",
        ACCENT_COLOR, TEXT_DIM
    ));
    panel.append_child(&title)?;

    // Input section with label
    let input_section: HtmlElement = document.create_element("div")?.dyn_into()?;
    input_section.set_attribute("style", "display: flex; flex-direction: column; flex: 1; min-height: 0;")?;

    let input_label: HtmlElement = document.create_element("div")?.dyn_into()?;
    input_label.set_inner_html(&format!(
        "<span style=\"color: {}; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;\">📝 Your Prompt</span>",
        TEXT_DIM
    ));
    input_label.style().set_property("margin-bottom", "6px")?;
    input_section.append_child(&input_label)?;

    // Large textarea for input
    let textarea: web_sys::HtmlTextAreaElement = document.create_element("textarea")?.dyn_into()?;
    textarea.set_id("completion-input");
    textarea.set_placeholder("Type your prompt here...\n\nTry something like:\n• \"The quick brown fox\"\n• \"Once upon a time\"\n• \"In the beginning\"");
    textarea.set_attribute("style", &format!(
        "flex: 1; \
        min-height: 120px; \
        padding: 16px; \
        background: #0d1117; \
        border: 1px solid #30363d; \
        border-radius: 6px; \
        color: {}; \
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; \
        font-size: 15px; \
        line-height: 1.6; \
        resize: none; \
        outline: none;",
        TEXT_COLOR
    ))?;

    // Add input event listener
    let input_closure = Closure::wrap(Box::new(move || {
        on_completion_input();
    }) as Box<dyn Fn()>);
    textarea.add_event_listener_with_callback("input", input_closure.as_ref().unchecked_ref())?;
    input_closure.forget();

    // Focus style - use simple closure without event type
    let focus_in = Closure::wrap(Box::new(move || {
        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                if let Some(el) = document.get_element_by_id("completion-input") {
                    let _ = el.dyn_ref::<HtmlElement>().map(|e| {
                        e.style().set_property("border-color", ACCENT_COLOR)
                    });
                }
            }
        }
    }) as Box<dyn Fn()>);
    textarea.add_event_listener_with_callback("focus", focus_in.as_ref().unchecked_ref())?;
    focus_in.forget();

    let focus_out = Closure::wrap(Box::new(move || {
        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                if let Some(el) = document.get_element_by_id("completion-input") {
                    let _ = el.dyn_ref::<HtmlElement>().map(|e| {
                        e.style().set_property("border-color", "#30363d")
                    });
                }
            }
        }
    }) as Box<dyn Fn()>);
    textarea.add_event_listener_with_callback("blur", focus_out.as_ref().unchecked_ref())?;
    focus_out.forget();

    input_section.append_child(&textarea)?;
    panel.append_child(&input_section)?;

    // Output section with label
    let output_section: HtmlElement = document.create_element("div")?.dyn_into()?;
    output_section.set_attribute("style", "display: flex; flex-direction: column; flex: 1; min-height: 0;")?;

    let output_label: HtmlElement = document.create_element("div")?.dyn_into()?;
    output_label.set_inner_html(&format!(
        "<span style=\"color: {}; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;\">🤖 Generated Output</span>\
        <span style=\"color: {}; font-size: 10px; margin-left: 8px;\">(selectable text)</span>",
        TEXT_DIM, "#484f58"
    ));
    output_label.style().set_property("margin-bottom", "6px")?;
    output_section.append_child(&output_label)?;

    // Output container (selectable text)
    let output: HtmlElement = document.create_element("div")?.dyn_into()?;
    output.set_id("completion-output");
    output.set_attribute("style", &format!(
        "flex: 1; \
        min-height: 100px; \
        padding: 16px; \
        background: #0d1117; \
        border: 1px solid #30363d; \
        border-radius: 6px; \
        color: {}; \
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; \
        font-size: 15px; \
        line-height: 1.6; \
        overflow-y: auto; \
        user-select: text; \
        cursor: text; \
        white-space: pre-wrap; \
        word-wrap: break-word;",
        TEXT_COLOR
    ))?;
    output.set_inner_html(&format!(
        "<span style=\"color: #484f58; font-style: italic;\">Generated text will appear here...</span>"
    ));
    output_section.append_child(&output)?;
    panel.append_child(&output_section)?;

    // Predictions section
    let pred_section: HtmlElement = document.create_element("div")?.dyn_into()?;
    pred_section.set_attribute("style", "display: flex; flex-direction: column;")?;

    let pred_label: HtmlElement = document.create_element("div")?.dyn_into()?;
    pred_label.set_inner_html(&format!(
        "<span style=\"color: {}; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;\">🎯 Next Token Predictions</span>",
        TEXT_DIM
    ));
    pred_label.style().set_property("margin-bottom", "6px")?;
    pred_section.append_child(&pred_label)?;

    // Predictions container
    let predictions: HtmlElement = document.create_element("div")?.dyn_into()?;
    predictions.set_id("predictions-container");
    predictions.set_attribute("style", &format!(
        "display: flex; \
        flex-wrap: wrap; \
        gap: 8px; \
        padding: 12px; \
        background: #0d1117; \
        border: 1px solid #30363d; \
        border-radius: 6px; \
        max-height: 120px; \
        overflow-y: auto;"
    ))?;
    predictions.set_inner_html("<div style=\"color: #484f58; font-size: 12px;\">Type something to see predictions...</div>");
    pred_section.append_child(&predictions)?;
    panel.append_child(&pred_section)?;

    // Add panel to body
    let body = document.body().ok_or("No body")?;
    body.append_child(&panel)?;

    Ok(())
}

/// Handle input in the completion textarea
fn on_completion_input() {
    // Get the input text
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(textarea) = document.get_element_by_id("completion-input") {
                if let Ok(ta) = textarea.dyn_into::<web_sys::HtmlTextAreaElement>() {
                    let text = ta.value();

                    // Update state
                    STATE.with(|s| {
                        if let Some(state) = s.borrow_mut().as_mut() {
                            state.completion.set_input(&text);
                        }
                    });

                    // Debounced generation
                    schedule_debounced_generation();
                }
            }
        }
    }
}

/// Toggle accelerator mode
fn toggle_accelerator(mode: AcceleratorMode) {
    STATE.with(|state| {
        if let Some(ref mut s) = *state.borrow_mut() {
            // Don't allow switching to GPU if not available or failed
            if mode == AcceleratorMode::Gpu {
                if !s.gpu_available || s.gpu_init_state == GpuInitState::Failed {
                    return;
                }
                // Only allow GPU if ready
                if s.gpu_init_state != GpuInitState::Ready {
                    return;
                }
            }
            s.accelerator_mode = mode;
        }
    });
}

/// Update GPU/CPU button styles based on current state
fn update_gpu_button(document: &Document) -> Result<(), JsValue> {
    let gpu_btn = document.get_element_by_id("gpu-btn").ok_or("no gpu button")?;
    let cpu_btn = document.get_element_by_id("cpu-btn").ok_or("no cpu button")?;
    let status = document.get_element_by_id("gpu-status").ok_or("no gpu status")?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            let gpu_btn_el = gpu_btn.dyn_ref::<HtmlElement>().unwrap();
            let cpu_btn_el = cpu_btn.dyn_ref::<HtmlElement>().unwrap();

            // Check if GPU is usable
            let gpu_usable = s.gpu_available && s.gpu_init_state != GpuInitState::Failed;

            match s.accelerator_mode {
                AcceleratorMode::Gpu => {
                    // GPU is active
                    let gpu_bg = match s.gpu_init_state {
                        GpuInitState::NotStarted => ACCENT_COLOR,
                        GpuInitState::Initializing => "#f0ad4e",
                        GpuInitState::Ready => ACCENT_COLOR,
                        GpuInitState::Failed => ACCENT_SECONDARY,
                    };
                    let _ = gpu_btn_el.style().set_property("background", gpu_bg);
                    let _ = gpu_btn_el.style().set_property("border", "none");
                    let _ = gpu_btn_el.style().set_property("color", BG_COLOR);
                    let _ = gpu_btn_el.style().set_property("font-weight", "600");
                    let _ = gpu_btn_el.style().set_property("opacity", "1");
                    let _ = gpu_btn_el.style().set_property("cursor", "pointer");

                    // CPU is inactive
                    let _ = cpu_btn_el.style().set_property("background", "#1a2a3a");
                    let _ = cpu_btn_el.style().set_property("border", &format!("1px solid {}", NEURON_STROKE));
                    let _ = cpu_btn_el.style().set_property("color", TEXT_COLOR);
                    let _ = cpu_btn_el.style().set_property("font-weight", "500");
                }
                AcceleratorMode::Cpu => {
                    // CPU is active
                    let _ = cpu_btn_el.style().set_property("background", ACCENT_COLOR);
                    let _ = cpu_btn_el.style().set_property("border", "none");
                    let _ = cpu_btn_el.style().set_property("color", BG_COLOR);
                    let _ = cpu_btn_el.style().set_property("font-weight", "600");

                    // GPU button styling
                    if gpu_usable {
                        let _ = gpu_btn_el.style().set_property("background", "#1a2a3a");
                        let _ = gpu_btn_el.style().set_property("border", &format!("1px solid {}", NEURON_STROKE));
                        let _ = gpu_btn_el.style().set_property("color", TEXT_COLOR);
                        let _ = gpu_btn_el.style().set_property("font-weight", "500");
                        let _ = gpu_btn_el.style().set_property("opacity", "1");
                        let _ = gpu_btn_el.style().set_property("cursor", "pointer");
                    } else {
                        // Disabled style
                        let _ = gpu_btn_el.style().set_property("background", "#1a1a2a");
                        let _ = gpu_btn_el.style().set_property("border", "1px solid #2a2a3a");
                        let _ = gpu_btn_el.style().set_property("color", "#4a4a5a");
                        let _ = gpu_btn_el.style().set_property("font-weight", "500");
                        let _ = gpu_btn_el.style().set_property("opacity", "0.5");
                        let _ = gpu_btn_el.style().set_property("cursor", "not-allowed");
                    }
                }
            };

            // Update status text
            let (status_text, status_color) = if !s.gpu_available {
                ("WebGPU unavailable".to_string(), ACCENT_SECONDARY)
            } else {
                match s.gpu_init_state {
                    GpuInitState::NotStarted => ("Initializing...".to_string(), TEXT_DIM),
                    GpuInitState::Initializing => ("Initializing GPU...".to_string(), "#f0ad4e"),
                    GpuInitState::Ready => {
                        if s.accelerator_mode == AcceleratorMode::Gpu {
                            ("WebGPU active".to_string(), ACCENT_COLOR)
                        } else {
                            ("WebGPU ready".to_string(), TEXT_DIM)
                        }
                    }
                    GpuInitState::Failed => ("GPU failed".to_string(), ACCENT_SECONDARY),
                }
            };
            status.set_text_content(Some(&status_text));
            let _ = status.dyn_ref::<HtmlElement>().unwrap().style().set_property("color", status_color);
        }
    });

    Ok(())
}

/// Set chart scale mode
fn set_chart_scale(scale: ChartScale) {
    STATE.with(|state| {
        if let Some(ref mut s) = *state.borrow_mut() {
            s.chart_scale = scale;
        }
    });
    render();
}

/// Update chart scale button styles
fn update_scale_buttons(document: &Document) -> Result<(), JsValue> {
    let linear_btn = document.get_element_by_id("scale-linear");
    let log_btn = document.get_element_by_id("scale-log");
    let loglog_btn = document.get_element_by_id("scale-loglog");

    let current_scale = STATE.with(|state| {
        state.borrow().as_ref().map(|s| s.chart_scale).unwrap_or(ChartScale::Linear)
    });

    let active_style = format!(
        "padding: 4px 10px; font-size: 11px; cursor: pointer; \
        border: 1px solid {}; background: {}; color: {}; font-weight: 600;",
        NEURON_STROKE, ACCENT_COLOR, BG_COLOR
    );
    let inactive_style = format!(
        "padding: 4px 10px; font-size: 11px; cursor: pointer; \
        border: 1px solid {}; background: #1a2a3a; color: {}; font-weight: 500;",
        NEURON_STROKE, TEXT_COLOR
    );

    if let Some(btn) = linear_btn {
        let style = if current_scale == ChartScale::Linear {
            format!("{}; border-radius: 4px 0 0 4px;", active_style)
        } else {
            format!("{}; border-radius: 4px 0 0 4px;", inactive_style)
        };
        let _ = btn.set_attribute("style", &style);
    }

    if let Some(btn) = log_btn {
        let style = if current_scale == ChartScale::Log {
            format!("{}; border-radius: 0; border-left: none;", active_style)
        } else {
            format!("{}; border-radius: 0; border-left: none;", inactive_style)
        };
        let _ = btn.set_attribute("style", &style);
    }

    if let Some(btn) = loglog_btn {
        let style = if current_scale == ChartScale::LogLog {
            format!("{}; border-radius: 0 4px 4px 0; border-left: none;", active_style)
        } else {
            format!("{}; border-radius: 0 4px 4px 0; border-left: none;", inactive_style)
        };
        let _ = btn.set_attribute("style", &style);
    }

    Ok(())
}

/// Cancel any pending debounced generation
fn cancel_debounce() {
    DEBOUNCE_HANDLE.with(|handle| {
        if let Some(id) = handle.borrow_mut().take() {
            if let Some(window) = web_sys::window() {
                window.clear_timeout_with_handle(id);
            }
        }
    });
}

/// Schedule a debounced completion generation
fn schedule_debounced_generation() {
    cancel_debounce();

    if let Some(window) = web_sys::window() {
        let closure = Closure::once(Box::new(move || {
            do_generate_completion();
        }) as Box<dyn FnOnce()>);

        // 150ms debounce delay
        let result = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            closure.as_ref().unchecked_ref(),
            150,
        );

        if let Ok(id) = result {
            DEBOUNCE_HANDLE.with(|handle| {
                *handle.borrow_mut() = Some(id);
            });
        }

        closure.forget();
    }
}

/// Actually perform the completion generation (called after debounce)
fn do_generate_completion() {
    // Get prompt from the new textarea
    let prompt = get_completion_textarea_input();

    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            if !prompt.is_empty() {
                let num_tokens = state.num_tokens_to_generate;
                state.completion.set_input(&prompt);
                state.generate_completion(num_tokens);
            }
        }
    });

    // Update the HTML output elements
    update_completion_output();
    render();
}

/// Get input from the completion textarea
fn get_completion_textarea_input() -> String {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(textarea) = document.get_element_by_id("completion-input") {
                if let Ok(ta) = textarea.dyn_into::<web_sys::HtmlTextAreaElement>() {
                    return ta.value();
                }
            }
        }
    }
    String::new()
}

/// Update the HTML completion output elements
fn update_completion_output() {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            STATE.with(|s| {
                if let Some(state) = s.borrow().as_ref() {
                    // Get tokenizer reference
                    if let Some(tokenizer) = state.tokenizer.as_ref() {
                        // Update output with colored text
                        if let Some(output) = document.get_element_by_id("completion-output") {
                            let input_text = &state.completion.input;
                            let generated_text = state.completion.decode_generated(tokenizer);

                            if input_text.is_empty() && generated_text.is_empty() {
                                output.set_inner_html("<span style=\"color: #484f58; font-style: italic;\">Generated text will appear here...</span>");
                            } else {
                                // Input in white, generated in teal
                                let html = format!(
                                    "<span style=\"color: #c9d1d9;\">{}</span><span style=\"color: #4ecdc4; background: rgba(78, 205, 196, 0.1); border-radius: 2px;\">{}</span>",
                                    html_escape(&input_text),
                                    html_escape(&generated_text)
                                );
                                output.set_inner_html(&html);
                            }
                        }

                        // Update predictions
                        if let Some(pred_container) = document.get_element_by_id("predictions-container") {
                            if state.completion.top_predictions.is_empty() {
                                pred_container.set_inner_html("<div style=\"color: #484f58; font-size: 12px;\">Type something to see predictions...</div>");
                            } else {
                                let mut html = String::new();
                                for (i, pred) in state.completion.top_predictions.iter().enumerate() {
                                    let prob_pct = pred.prob() * 100.0;
                                    let (bg_color, text_color, border_color) = if i == 0 {
                                        ("#1a3a3a", "#4ecdc4", "#4ecdc4")
                                    } else if prob_pct > 10.0 {
                                        ("#1a2634", "#98c1d9", "#3d5a80")
                                    } else {
                                        ("#161b22", "#8b949e", "#30363d")
                                    };

                                    let token_display = html_escape(&escape_token_for_display(&pred.text()));
                                    html.push_str(&format!(
                                        "<div style=\"display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; \
                                        background: {}; border: 1px solid {}; border-radius: 6px; font-family: monospace;\">\
                                        <span style=\"color: {}; font-weight: {};\">{}</span>\
                                        <span style=\"color: {}; font-size: 11px; opacity: 0.8;\">{:.1}%</span>\
                                        </div>",
                                        bg_color, border_color, text_color,
                                        if i == 0 { "bold" } else { "normal" },
                                        token_display, text_color, prob_pct
                                    ));
                                }
                                pred_container.set_inner_html(&html);
                            }
                        }
                    }
                }
            });
        }
    }
}

/// Escape HTML special characters
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
        .replace('\n', "<br>")
}

/// Escape special characters for display in token predictions
fn escape_token_for_display(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\n' => "↵".to_string(),
            '\t' => "⇥".to_string(),
            '\r' => "↲".to_string(),
            ' ' => "␣".to_string(),
            c if c.is_control() => format!("\\x{:02x}", c as u32),
            c => c.to_string(),
        })
        .collect()
}

/// Handle token slider change
fn on_token_slider_change() {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(slider) = document.get_element_by_id("token-slider") {
                if let Ok(slider) = slider.dyn_into::<web_sys::HtmlInputElement>() {
                    let value: usize = slider.value().parse().unwrap_or(10);

                    // Update label immediately (responsive feedback)
                    if let Some(label) = document.get_element_by_id("token-count-label") {
                        label.set_text_content(Some(&format!("Tokens: {}", value)));
                    }

                    // Update state value immediately
                    STATE.with(|s| {
                        if let Some(state) = s.borrow_mut().as_mut() {
                            state.num_tokens_to_generate = value;
                        }
                    });

                    // Schedule debounced regeneration
                    schedule_debounced_generation();
                }
            }
        }
    }
}

/// Handle temperature slider change
fn on_temp_slider_change() {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(slider) = document.get_element_by_id("temp-slider") {
                if let Ok(slider) = slider.dyn_into::<web_sys::HtmlInputElement>() {
                    let value: f32 = slider.value().parse::<f32>().unwrap_or(80.0) / 100.0;

                    // Update label immediately (responsive feedback)
                    if let Some(label) = document.get_element_by_id("temp-label") {
                        label.set_text_content(Some(&format!("Temp: {:.1}", value)));
                    }

                    // Update state value immediately
                    STATE.with(|s| {
                        if let Some(state) = s.borrow_mut().as_mut() {
                            state.completion.set_temperature(value);
                        }
                    });

                    // Schedule debounced regeneration
                    schedule_debounced_generation();
                }
            }
        }
    }
}

/// Update mode-specific controls visibility
fn update_mode_controls() {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            let mode = STATE.with(|s| {
                s.borrow().as_ref().map(|state| state.mode).unwrap_or(AppMode::Training)
            });

            // Scale controls (training mode)
            if let Some(scale_controls) = document.get_element_by_id("scale-controls") {
                let display = if mode == AppMode::Training { "flex" } else { "none" };
                let _ = scale_controls.dyn_ref::<HtmlElement>().map(|el| {
                    el.style().set_property("display", display)
                });
            }

            // Completion controls (completion mode)
            if let Some(completion_controls) = document.get_element_by_id("completion-controls") {
                let display = if mode == AppMode::Completion { "flex" } else { "none" };
                let _ = completion_controls.dyn_ref::<HtmlElement>().map(|el| {
                    el.style().set_property("display", display)
                });
            }

            // Completions panel overlay (completion mode)
            if let Some(panel) = document.get_element_by_id("completions-panel") {
                let display = if mode == AppMode::Completion { "flex" } else { "none" };
                let _ = panel.dyn_ref::<HtmlElement>().map(|el| {
                    el.style().set_property("display", display)
                });
            }
        }
    }
}

/// Create a styled button
fn create_button(document: &Document, text: &str, id: &str) -> Result<HtmlElement, JsValue> {
    let button: HtmlElement = document
        .create_element("button")?
        .dyn_into()?;

    button.set_id(id);
    button.set_inner_text(text);

    let style = button.style();
    style.set_property("padding", "8px 16px")?;
    style.set_property("background", ACCENT_COLOR)?;
    style.set_property("color", "#0a0a0f")?;
    style.set_property("border", "none")?;
    style.set_property("border-radius", "4px")?;
    style.set_property("cursor", "pointer")?;
    style.set_property("font-weight", "bold")?;

    Ok(button)
}

/// Set up mouse and keyboard event handlers
fn setup_event_handlers(canvas: &HtmlCanvasElement) -> Result<(), JsValue> {
    // Mouse click
    let click_closure = Closure::wrap(Box::new(move |event: MouseEvent| {
        handle_click(event.offset_x() as f64, event.offset_y() as f64);
    }) as Box<dyn Fn(MouseEvent)>);

    canvas.add_event_listener_with_callback("click", click_closure.as_ref().unchecked_ref())?;
    click_closure.forget();

    Ok(())
}

/// Handle mouse click
fn handle_click(x: f64, _y: f64) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            // Check if click is in completion area to select a prediction
            if state.mode == AppMode::Completion && x > SIDEBAR_WIDTH {
                // Could implement token selection here
            }
        }
    });
    render();
}

/// Initialize with sample training data
fn initialize_with_sample_data() {
    let sample_text = generate_sample_text();

    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            match state.initialize(&sample_text) {
                Ok(()) => log("Model initialized successfully"),
                Err(e) => log(&format!("Init error: {}", e)),
            }
        }
    });

    render();
}

/// Generate sample training text
fn generate_sample_text() -> String {
    // Generate a bunch of simple repetitive text for training
    let mut text = String::new();

    for i in 0..1000 {
        text.push_str(&format!("The number {} is interesting. ", i % 100));
        if i % 10 == 0 {
            text.push_str("Numbers are everywhere in mathematics. ");
        }
        if i % 20 == 0 {
            text.push_str("Learning to count is fundamental. ");
        }
        if i % 30 == 0 {
            text.push_str("Math helps us understand the world. ");
        }
    }

    text
}

/// Toggle training state
fn toggle_training() {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.training_state = match state.training_state {
                TrainingState::Idle => {
                    if state.model.is_some() {
                        let window = web_sys::window().unwrap();
                        let performance = window.performance().unwrap();
                        state.metrics.start_timer(performance.now());
                        TrainingState::Training
                    } else {
                        log("Initialize model first");
                        TrainingState::Idle
                    }
                }
                TrainingState::Training => TrainingState::Paused,
                TrainingState::Paused => TrainingState::Training,
            };

            // Update button text
            if let Some(window) = web_sys::window() {
                if let Some(document) = window.document() {
                    if let Some(btn) = document.get_element_by_id("train-btn") {
                        let text = match state.training_state {
                            TrainingState::Idle => "Start Training",
                            TrainingState::Training => "Pause",
                            TrainingState::Paused => "Resume",
                        };
                        btn.set_inner_html(text);
                    }
                }
            }
        }
    });

    render();
}

/// Toggle between training and completion modes
fn toggle_mode() {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.mode = match state.mode {
                AppMode::Training => AppMode::Completion,
                AppMode::Completion => AppMode::Training,
            };

            // Update button text
            if let Some(window) = web_sys::window() {
                if let Some(document) = window.document() {
                    if let Some(btn) = document.get_element_by_id("mode-btn") {
                        let text = match state.mode {
                            AppMode::Training => "Completion",
                            AppMode::Completion => "Training",
                        };
                        btn.set_inner_html(text);
                    }
                }
            }
        }
    });

    // Update which controls are visible
    update_mode_controls();
    render();
}

/// Start animation loop
fn start_animation_loop() -> Result<(), JsValue> {
    let f = Rc::new(RefCell::new(None::<Closure<dyn FnMut()>>));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        // Training step
        STATE.with(|s| {
            if let Some(state) = s.borrow_mut().as_mut() {
                if state.training_state == TrainingState::Training {
                    // Run a few steps per frame
                    for _ in 0..state.batches_per_frame {
                        if state.train_step().is_none() {
                            // Dataset exhausted, reset for next epoch
                            if let Some(dataset) = state.training_data.as_mut() {
                                dataset.reset();
                            }
                            let epoch = state.metrics.epoch() + 1;
                            let steps_per_epoch = state.training_data.as_ref()
                                .map(|d| d.len() / 32)
                                .unwrap_or(100);
                            state.metrics.set_epoch(epoch, steps_per_epoch);
                        }
                    }

                    // Periodic evaluation
                    if state.metrics.step() % 100 == 0 {
                        state.evaluate();
                    }
                }
            }
        });

        render();

        // Schedule next frame
        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}

/// Request animation frame
fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

/// Render the visualization
fn render() {
    CTX.with(|ctx_cell| {
        STATE.with(|state_cell| {
            let ctx_ref = ctx_cell.borrow();
            let state_ref = state_cell.borrow();

            if let (Some(ctx), Some(state)) = (ctx_ref.as_ref(), state_ref.as_ref()) {
                render_app(ctx, state);
            }
        });
    });
}

/// Main render function
fn render_app(ctx: &CanvasRenderingContext2d, state: &AppState) {
    let width = state.canvas_width;
    let height = state.canvas_height;

    // Clear canvas
    ctx.set_fill_style_str(BG_COLOR);
    ctx.fill_rect(0.0, 0.0, width, height);

    // Header
    render_header(ctx, state, width);

    // Sidebar with metrics
    render_sidebar(ctx, state, height);

    // Main content area
    let content_x = SIDEBAR_WIDTH;
    let content_y = HEADER_HEIGHT;
    let content_width = width - SIDEBAR_WIDTH;
    let content_height = height - HEADER_HEIGHT;

    match state.mode {
        AppMode::Training => {
            // Loss chart
            render_loss_chart(
                ctx,
                &state.metrics,
                content_x + 20.0,
                content_y + 20.0,
                content_width - 40.0,
                content_height - 40.0,
                state.chart_scale,
            );
        }
        AppMode::Completion => {
            // Completion panel
            if let Some(tokenizer) = state.tokenizer.as_ref() {
                render_completion_panel(
                    ctx,
                    &state.completion,
                    tokenizer,
                    content_x + 20.0,
                    content_y + 20.0,
                    content_width - 40.0,
                    content_height - 40.0,
                );
            } else {
                // No model yet
                ctx.set_fill_style_str(TEXT_DIM);
                ctx.set_font("14px monospace");
                let _ = ctx.fill_text(
                    "Initialize model to use completions",
                    content_x + 100.0,
                    content_y + 100.0,
                );
            }
        }
    }
}

/// Render header - now just shows status info since main controls are in HTML
fn render_header(ctx: &CanvasRenderingContext2d, state: &AppState, width: f64) {
    ctx.set_fill_style_str(PANEL_BG);
    ctx.fill_rect(0.0, 0.0, width, HEADER_HEIGHT);

    // Mode indicator on left
    let mode_text = match state.mode {
        AppMode::Training => "TRAINING",
        AppMode::Completion => "COMPLETION",
    };
    ctx.set_fill_style_str(ACCENT_COLOR);
    ctx.set_font("bold 12px monospace");
    let _ = ctx.fill_text(mode_text, 20.0, 30.0);

    // Training status
    let (status, status_color) = match state.training_state {
        TrainingState::Idle => ("Ready", TEXT_DIM),
        TrainingState::Training => ("Training", "#4ecdc4"),
        TrainingState::Paused => ("Paused", "#f0ad4e"),
    };
    ctx.set_fill_style_str(status_color);
    ctx.set_font("12px monospace");
    let _ = ctx.fill_text(status, 120.0, 30.0);

    // Epoch and step info on right
    if state.model.is_some() {
        ctx.set_fill_style_str(TEXT_DIM);
        ctx.set_font("11px monospace");
        let info = format!(
            "Epoch {} | Step {} | Loss: {:.3}",
            state.metrics.epoch(),
            state.metrics.step(),
            state.metrics.train_loss()
        );
        let _ = ctx.fill_text(&info, width - 280.0, 30.0);
    }
}

/// Render sidebar with metrics
fn render_sidebar(ctx: &CanvasRenderingContext2d, state: &AppState, height: f64) {
    ctx.set_fill_style_str(PANEL_BG);
    ctx.fill_rect(0.0, HEADER_HEIGHT, SIDEBAR_WIDTH, height - HEADER_HEIGHT);

    // Get current time for throughput calculation
    let current_time = web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    render_metrics_panel(
        ctx,
        &state.metrics,
        10.0,
        HEADER_HEIGHT + 10.0,
        SIDEBAR_WIDTH - 20.0,
        current_time,
    );

    // Model info
    if let Some(model) = state.model.as_ref() {
        let info_y = HEADER_HEIGHT + 250.0;

        ctx.set_fill_style_str(TEXT_COLOR);
        ctx.set_font("12px monospace");
        let _ = ctx.fill_text("Model Info:", 20.0, info_y);

        ctx.set_fill_style_str(TEXT_DIM);
        let _ = ctx.fill_text(
            &format!("Parameters: {}", format_number(model.num_parameters())),
            20.0,
            info_y + 20.0,
        );
        let _ = ctx.fill_text(
            &format!("Layers: {}", state.config.n_layers),
            20.0,
            info_y + 40.0,
        );
        let _ = ctx.fill_text(
            &format!("d_model: {}", state.config.d_model),
            20.0,
            info_y + 60.0,
        );
        let _ = ctx.fill_text(
            &format!("Vocab: {}", state.config.vocab_size),
            20.0,
            info_y + 80.0,
        );
    }
}

/// Format large numbers
fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

/// Log to browser console
fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

// ==================== MODEL SAVE/LOAD FUNCTIONS ====================

/// Save the current model to a downloadable JSON file
fn save_model() {
    STATE.with(|s| {
        let state = s.borrow();
        if let Some(state) = state.as_ref() {
            // Check if model exists
            let model = match state.model.as_ref() {
                Some(m) => m,
                None => {
                    log("No model to save");
                    return;
                }
            };

            let tokenizer = match state.tokenizer.as_ref() {
                Some(t) => t,
                None => {
                    log("No tokenizer to save");
                    return;
                }
            };

            // Create saved state
            let saved_state = SavedModelState {
                config: state.config.clone(),
                model: model.clone(),
                tokenizer: tokenizer.clone(),
                metrics: state.metrics.clone(),
                epoch: state.metrics.epoch(),
                step: state.metrics.step(),
            };

            // Serialize to JSON
            let json = match serde_json::to_string_pretty(&saved_state) {
                Ok(j) => j,
                Err(e) => {
                    log(&format!("Failed to serialize model: {}", e));
                    return;
                }
            };

            // Trigger download
            if let Err(e) = trigger_download(&json, "lm-checkpoint.json") {
                log(&format!("Failed to download: {:?}", e));
            } else {
                log(&format!("Model saved! (Step {}, Epoch {})", state.metrics.step(), state.metrics.epoch()));
            }
        }
    });
}

/// Trigger file download in browser
fn trigger_download(content: &str, filename: &str) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("No window")?;
    let document = window.document().ok_or("No document")?;

    // Create Blob from content
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&JsValue::from_str(content));

    let blob_options = web_sys::BlobPropertyBag::new();
    blob_options.set_type("application/json");

    let blob = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &blob_options)?;

    // Create object URL
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;

    // Create download link
    let link: HtmlElement = document.create_element("a")?.dyn_into()?;
    link.set_attribute("href", &url)?;
    link.set_attribute("download", filename)?;
    link.style().set_property("display", "none")?;

    // Add to document, click, and remove
    document.body().ok_or("No body")?.append_child(&link)?;
    link.click();
    document.body().ok_or("No body")?.remove_child(&link)?;

    // Revoke URL
    web_sys::Url::revoke_object_url(&url)?;

    Ok(())
}

/// Trigger file input click for loading model
fn trigger_load_model() {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(input) = document.get_element_by_id("model-file-input") {
                if let Ok(input) = input.dyn_into::<web_sys::HtmlInputElement>() {
                    // Reset the input so same file can be selected again
                    input.set_value("");
                    input.click();
                }
            }
        }
    }
}

/// Handle file selection for model loading
fn handle_model_file_selected() {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(input) = document.get_element_by_id("model-file-input") {
                if let Ok(input) = input.dyn_into::<web_sys::HtmlInputElement>() {
                    if let Some(files) = input.files() {
                        if files.length() > 0 {
                            if let Some(file) = files.get(0) {
                                // Read file content using FileReader
                                let reader = match web_sys::FileReader::new() {
                                    Ok(r) => r,
                                    Err(e) => {
                                        log(&format!("Failed to create FileReader: {:?}", e));
                                        return;
                                    }
                                };

                                // Set up onload handler
                                let onload = Closure::wrap(Box::new(move |event: web_sys::Event| {
                                    if let Some(target) = event.target() {
                                        if let Ok(reader) = target.dyn_into::<web_sys::FileReader>() {
                                            if let Ok(result) = reader.result() {
                                                if let Some(text) = result.as_string() {
                                                    load_model_from_json(&text);
                                                }
                                            }
                                        }
                                    }
                                }) as Box<dyn Fn(web_sys::Event)>);

                                reader.set_onload(Some(onload.as_ref().unchecked_ref()));
                                onload.forget();

                                // Start reading
                                if let Err(e) = reader.read_as_text(&file) {
                                    log(&format!("Failed to read file: {:?}", e));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Load model from JSON string
fn load_model_from_json(json: &str) {
    // Parse JSON
    let saved_state: SavedModelState = match serde_json::from_str(json) {
        Ok(s) => s,
        Err(e) => {
            log(&format!("Failed to parse model file: {}", e));
            return;
        }
    };

    // Stop any ongoing training
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.training_state = TrainingState::Idle;
        }
    });

    // Update state with loaded model
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.config = saved_state.config;
            state.model = Some(saved_state.model);
            state.tokenizer = Some(saved_state.tokenizer);
            state.metrics = saved_state.metrics;

            // Reinitialize training data if needed (user will need to click Initialize to train again)
            state.training_data = None;
            state.val_data = None;
            state.lr_scheduler = None;

            log(&format!(
                "Model loaded! Epoch: {}, Step: {}, Loss: {:.4}",
                saved_state.epoch,
                saved_state.step,
                state.metrics.train_loss()
            ));
        }
    });

    // Update UI
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            // Update train button
            if let Some(btn) = document.get_element_by_id("train-btn") {
                btn.set_text_content(Some("Train"));
            }
        }
    }

    render();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1500), "1.5K");
        assert_eq!(format_number(1_500_000), "1.50M");
    }

    #[test]
    fn test_sample_text_generation() {
        let text = generate_sample_text();
        assert!(!text.is_empty());
        assert!(text.len() > 1000);
    }
}
