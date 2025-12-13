use common::{parse_digits_from_bytes, Digit};
use neural::{Network, mnist::MnistSample};
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, MouseEvent, WheelEvent};

const INPUT_PANEL_WIDTH: f64 = 200.0;
const NEURON_RADIUS: f64 = 8.0;  // Smaller neurons
const LAYER_SPACING: f64 = 250.0;

// Modern color palette
const BG_COLOR: &str = "#0a0a0f";
const PANEL_BG: &str = "#12121a";
const NEURON_BASE: &str = "#1a2a4a";
const NEURON_STROKE: &str = "#3d5a80";
const NEURON_HOVER: &str = "#98c1d9";
const WEIGHT_COLOR: &str = "#2a3a4a";
const WEIGHT_HIGHLIGHT: &str = "#4ecdc4";
const ACCENT_COLOR: &str = "#4ecdc4";
const ACCENT_SECONDARY: &str = "#ff6b6b";
const TEXT_COLOR: &str = "#e0e6ed";
const TEXT_DIM: &str = "#6b7280";

// Training parameters
const LEARNING_RATE: f32 = 0.1;
const BATCH_SIZE: usize = 32;
const BATCHES_PER_FRAME: usize = 5;

struct NeuronPosition {
    layer_idx: usize,
    neuron_idx: usize,
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum TrainingState {
    Idle,
    Training,
    Paused,
}

struct TrainingMetrics {
    epoch: usize,
    batch: usize,
    total_batches: usize,
    samples_seen: usize,
    loss: f32,
    accuracy: f32,
    loss_history: Vec<f32>,
    log_scale: bool,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            epoch: 0,
            batch: 0,
            total_batches: 0,
            samples_seen: 0,
            loss: 0.0,
            accuracy: 0.0,
            loss_history: Vec::new(),
            log_scale: false,
        }
    }
}

struct State {
    network: Network,
    digits: Vec<Digit>,
    samples: Vec<MnistSample>,
    current_digit_idx: Option<usize>,
    neuron_positions: Vec<NeuronPosition>,
    hovered_neuron: Option<(usize, usize)>,
    // Hover from digit canvas (left panel) - pixel index 0-783
    hovered_digit_pixel: Option<usize>,
    mouse_x: f64,
    mouse_y: f64,
    canvas_width: f64,
    canvas_height: f64,
    offset_x: f64,
    offset_y: f64,
    zoom: f64,
    is_dragging: bool,
    drag_start_x: f64,
    drag_start_y: f64,
    // Training state
    training_state: TrainingState,
    metrics: TrainingMetrics,
    current_batch_idx: usize,
    // Current activations for visualization
    current_activations: Vec<Vec<f32>>,
    current_prediction: Option<usize>,
}

impl State {
    fn new(network: Network, digits: Vec<Digit>, canvas_width: f64, canvas_height: f64) -> Self {
        let neuron_positions = Self::calculate_positions(&network, canvas_width, canvas_height);
        let samples: Vec<MnistSample> = digits.iter().map(|d| MnistSample::new(d.clone())).collect();
        let num_layers = network.layer_sizes().len();

        Self {
            network,
            digits,
            samples,
            current_digit_idx: None,
            neuron_positions,
            hovered_neuron: None,
            hovered_digit_pixel: None,
            mouse_x: 0.0,
            mouse_y: 0.0,
            canvas_width,
            canvas_height,
            offset_x: 0.0,
            offset_y: 0.0,
            zoom: 1.0,
            is_dragging: false,
            drag_start_x: 0.0,
            drag_start_y: 0.0,
            training_state: TrainingState::Idle,
            metrics: TrainingMetrics::default(),
            current_batch_idx: 0,
            current_activations: vec![Vec::new(); num_layers],
            current_prediction: None,
        }
    }

    fn calculate_positions(network: &Network, _canvas_width: f64, canvas_height: f64) -> Vec<NeuronPosition> {
        let mut positions = Vec::new();
        let layer_sizes = network.layer_sizes();

        let available_height = canvas_height - 100.0;

        for (layer_idx, &actual_size) in layer_sizes.iter().enumerate() {
            if layer_idx == 0 {
                // Input layer: 28x28 grid
                let grid_size = 28;
                let cell_size = 4.0; // 4px per cell
                let grid_height = grid_size as f64 * cell_size;
                let grid_x = INPUT_PANEL_WIDTH + 100.0;
                let grid_y = (canvas_height - grid_height) / 2.0;

                for neuron_idx in 0..actual_size {
                    let row = neuron_idx / grid_size;
                    let col = neuron_idx % grid_size;
                    let x = grid_x + (col as f64) * cell_size + cell_size / 2.0;
                    let y = grid_y + (row as f64) * cell_size + cell_size / 2.0;
                    positions.push(NeuronPosition {
                        layer_idx,
                        neuron_idx,
                        x,
                        y,
                    });
                }
            } else if layer_idx == 1 && actual_size > 20 {
                // Hidden layer: column of small dots
                let layer_x = INPUT_PANEL_WIDTH + 100.0 + (layer_idx as f64) * LAYER_SPACING;
                let dot_size = 4.0;
                let dot_spacing = 1.0;
                let total_height = actual_size as f64 * (dot_size + dot_spacing);
                let start_y = (canvas_height - total_height) / 2.0;

                for neuron_idx in 0..actual_size {
                    let y = start_y + (neuron_idx as f64) * (dot_size + dot_spacing) + dot_size / 2.0;
                    positions.push(NeuronPosition {
                        layer_idx,
                        neuron_idx,
                        x: layer_x,
                        y,
                    });
                }
            } else {
                // Output layer: column layout (only 10 neurons)
                let layer_x = INPUT_PANEL_WIDTH + 100.0 + (layer_idx as f64) * LAYER_SPACING;

                // Calculate spacing to fit all neurons with room
                let neuron_spacing = (NEURON_RADIUS * 2.5).min(available_height / actual_size as f64);
                let total_height = (actual_size - 1) as f64 * neuron_spacing;
                let start_y = (canvas_height - total_height) / 2.0;

                for neuron_idx in 0..actual_size {
                    let y = start_y + (neuron_idx as f64) * neuron_spacing;
                    positions.push(NeuronPosition {
                        layer_idx,
                        neuron_idx,
                        x: layer_x,
                        y,
                    });
                }
            }
        }

        positions
    }

    fn screen_to_world(&self, screen_x: f64, screen_y: f64) -> (f64, f64) {
        let world_x = (screen_x - self.offset_x) / self.zoom;
        let world_y = (screen_y - self.offset_y) / self.zoom;
        (world_x, world_y)
    }

    fn get_neuron_at(&self, screen_x: f64, screen_y: f64) -> Option<(usize, usize)> {
        let (world_x, world_y) = self.screen_to_world(screen_x, screen_y);
        let hit_radius = NEURON_RADIUS / self.zoom.sqrt();

        for pos in &self.neuron_positions {
            let dx = world_x - pos.x;
            let dy = world_y - pos.y;
            if dx * dx + dy * dy <= (NEURON_RADIUS + hit_radius) * (NEURON_RADIUS + hit_radius) {
                return Some((pos.layer_idx, pos.neuron_idx));
            }
        }
        None
    }

    fn get_position(&self, layer_idx: usize, neuron_idx: usize) -> Option<(f64, f64)> {
        self.neuron_positions
            .iter()
            .find(|p| p.layer_idx == layer_idx && p.neuron_idx == neuron_idx)
            .map(|p| (p.x, p.y))
    }

    /// Returns the hovered input neuron index (0-783) from either digit canvas or main canvas
    fn get_hovered_input(&self) -> Option<usize> {
        // Digit canvas hover takes priority
        if let Some(idx) = self.hovered_digit_pixel {
            return Some(idx);
        }
        // Check if hovering input layer on main canvas
        if let Some((layer_idx, neuron_idx)) = self.hovered_neuron {
            if layer_idx == 0 {
                return Some(neuron_idx);
            }
        }
        None
    }

    fn zoom_at(&mut self, screen_x: f64, screen_y: f64, delta: f64) {
        let zoom_factor = if delta > 0.0 { 0.9 } else { 1.1 };
        let new_zoom = (self.zoom * zoom_factor).clamp(0.2, 4.0);
        let (world_x, world_y) = self.screen_to_world(screen_x, screen_y);
        self.zoom = new_zoom;
        self.offset_x = screen_x - world_x * self.zoom;
        self.offset_y = screen_y - world_y * self.zoom;
    }

    fn start_drag(&mut self, screen_x: f64, screen_y: f64) {
        self.is_dragging = true;
        self.drag_start_x = screen_x - self.offset_x;
        self.drag_start_y = screen_y - self.offset_y;
    }

    fn drag(&mut self, screen_x: f64, screen_y: f64) {
        if self.is_dragging {
            self.offset_x = screen_x - self.drag_start_x;
            self.offset_y = screen_y - self.drag_start_y;
        }
    }

    fn end_drag(&mut self) {
        self.is_dragging = false;
    }

    fn load_random_digit(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let next_idx = rng.gen_range(0..self.digits.len());
        self.current_digit_idx = Some(next_idx);

        // Run forward pass using the DIGIT (not shuffled samples array)
        // This ensures the displayed digit matches what we're predicting
        let digit = &self.digits[next_idx];
        let input: Vec<f32> = digit.pixels().iter().map(|&p| p as f32 / 255.0).collect();
        let output = self.network.forward(&input);

        // Store activations for each layer
        self.current_activations = vec![input.clone()]; // Input layer
        for i in 0..self.network.num_layers() {
            if let Some(acts) = self.network.get_activations(i) {
                self.current_activations.push(acts.clone());
            }
        }

        // Find prediction
        self.current_prediction = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx);
    }

    fn reset_network(&mut self) {
        self.network = Network::mnist_default();
        self.training_state = TrainingState::Idle;
        self.metrics = TrainingMetrics::default();
        self.current_batch_idx = 0;
        self.current_activations.clear();
        self.current_prediction = None;
    }

    fn train_batches(&mut self, num_batches: usize) {
        let total_samples = self.samples.len();
        let num_total_batches = total_samples / BATCH_SIZE;

        for _ in 0..num_batches {
            if self.training_state != TrainingState::Training {
                break;
            }

            // Get batch
            let start = self.current_batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(total_samples);

            if start >= total_samples {
                // End of epoch
                self.metrics.epoch += 1;
                self.current_batch_idx = 0;

                // Stop training if accuracy is good enough or max epochs reached
                if self.metrics.accuracy >= 0.99 || self.metrics.epoch >= 10 {
                    self.training_state = TrainingState::Paused;
                    break;
                }

                // Shuffle samples for next epoch (simple shuffle)
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                self.samples.shuffle(&mut rng);
                continue;
            }

            let inputs: Vec<Vec<f32>> = self.samples[start..end]
                .iter()
                .map(|s| s.normalized_pixels_f32())
                .collect();
            let labels: Vec<u8> = self.samples[start..end].iter().map(|s| s.label()).collect();

            // Train batch
            let loss = self.network.train_batch(&inputs, &labels, LEARNING_RATE);

            // Update metrics
            self.metrics.batch = self.current_batch_idx;
            self.metrics.total_batches = num_total_batches;
            self.metrics.samples_seen += inputs.len();
            self.metrics.loss = loss;
            self.metrics.loss_history.push(loss);

            // Evaluate accuracy periodically (every 10 batches)
            // Use digits array (not shuffled) from the END as validation set
            // This gives us a true measure of generalization
            if self.current_batch_idx % 10 == 0 {
                let val_size = 500;
                let val_start = self.digits.len().saturating_sub(val_size);
                let eval_inputs: Vec<Vec<f32>> = self.digits[val_start..]
                    .iter()
                    .map(|d| d.pixels().iter().map(|&p| p as f32 / 255.0).collect())
                    .collect();
                let eval_labels: Vec<u8> = self.digits[val_start..]
                    .iter()
                    .map(|d| d.label())
                    .collect();
                let (_, acc) = self.network.evaluate(&eval_inputs, &eval_labels);
                self.metrics.accuracy = acc;
            }

            self.current_batch_idx += 1;
        }
    }
}

thread_local! {
    static STATE: std::cell::RefCell<Option<State>> = const { std::cell::RefCell::new(None) };
}

pub fn init(csv_data: &[u8]) -> Result<(), JsValue> {
    let digits = parse_digits_from_bytes(csv_data).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let network = Network::mnist_default();

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;

    let canvas_width = window.inner_width()?.as_f64().unwrap_or(1200.0) - INPUT_PANEL_WIDTH;
    let canvas_height = window.inner_height()?.as_f64().unwrap_or(800.0);

    STATE.with(|state| {
        *state.borrow_mut() = Some(State::new(network, digits, canvas_width, canvas_height));
    });

    setup_ui(&document, canvas_width as u32, canvas_height as u32)?;
    setup_handlers(&document)?;

    // Load first digit
    STATE.with(|state| {
        if let Some(ref mut s) = *state.borrow_mut() {
            s.load_random_digit();
        }
    });
    render_digit(&document)?;
    render(&document)?;

    Ok(())
}

fn setup_ui(document: &Document, canvas_width: u32, canvas_height: u32) -> Result<(), JsValue> {
    let body = document.body().ok_or("no body")?;
    body.set_attribute(
        "style",
        &format!("margin: 0; overflow: hidden; background: {}; font-family: 'Inter', 'Segoe UI', sans-serif;", BG_COLOR),
    )?;

    // Container
    let container = document.create_element("div")?;
    container.set_attribute("style", "display: flex; height: 100vh;")?;

    // Left panel
    let left_panel = document.create_element("div")?;
    left_panel.set_id("left-panel");
    left_panel.set_attribute(
        "style",
        &format!(
            "width: {}px; background: {}; padding: 20px; box-sizing: border-box; \
             display: flex; flex-direction: column; gap: 16px; border-right: 1px solid #1a1a2e;",
            INPUT_PANEL_WIDTH as u32, PANEL_BG
        ),
    )?;

    // Title
    let title = document.create_element("div")?;
    title.set_attribute(
        "style",
        &format!("color: {}; font-size: 18px; font-weight: 600; margin-bottom: 8px;", ACCENT_COLOR),
    )?;
    title.set_text_content(Some("MNIST Trainer"));
    left_panel.append_child(&title)?;

    // Training Controls Section
    let controls_section = document.create_element("div")?;
    controls_section.set_attribute("style", "display: flex; flex-direction: column; gap: 8px;")?;

    // Start/Stop Button
    let train_btn = document.create_element("button")?;
    train_btn.set_id("train-btn");
    train_btn.set_text_content(Some("Start Training"));
    train_btn.set_attribute(
        "style",
        &format!(
            "width: 100%; padding: 12px; font-size: 14px; cursor: pointer; \
             background: {}; color: {}; border: none; border-radius: 8px; \
             font-weight: 600; transition: all 0.2s;",
            ACCENT_COLOR, BG_COLOR
        ),
    )?;
    controls_section.append_child(&train_btn)?;

    // Reset Button
    let reset_btn = document.create_element("button")?;
    reset_btn.set_id("reset-btn");
    reset_btn.set_text_content(Some("Reset Network"));
    reset_btn.set_attribute(
        "style",
        &format!(
            "width: 100%; padding: 10px; font-size: 13px; cursor: pointer; \
             background: transparent; color: {}; border: 1px solid {}; border-radius: 8px; \
             transition: all 0.2s;",
            ACCENT_SECONDARY, ACCENT_SECONDARY
        ),
    )?;
    controls_section.append_child(&reset_btn)?;

    left_panel.append_child(&controls_section)?;

    // Metrics Section
    let metrics_section = document.create_element("div")?;
    metrics_section.set_id("metrics-section");
    metrics_section.set_attribute(
        "style",
        &format!(
            "background: {}; border-radius: 8px; padding: 12px; \
             border: 1px solid #1a1a2e;",
            BG_COLOR
        ),
    )?;
    metrics_section.set_inner_html(&format!(
        "<div style='color: {}; font-size: 12px; margin-bottom: 8px; font-weight: 600;'>TRAINING METRICS</div>\
         <div id='metric-epoch' style='color: {}; font-size: 13px; margin-bottom: 4px;'>Epoch: 0</div>\
         <div id='metric-batch' style='color: {}; font-size: 13px; margin-bottom: 4px;'>Batch: 0/0</div>\
         <div id='metric-loss' style='color: {}; font-size: 13px; margin-bottom: 4px;'>Loss: -</div>\
         <div id='metric-accuracy' style='color: {}; font-size: 13px;'>Accuracy: -</div>",
        TEXT_DIM, TEXT_COLOR, TEXT_COLOR, TEXT_COLOR, TEXT_COLOR
    ));
    left_panel.append_child(&metrics_section)?;

    // Loss Chart
    let chart_canvas = document.create_element("canvas")?.dyn_into::<HtmlCanvasElement>()?;
    chart_canvas.set_id("loss-chart");
    chart_canvas.set_width(160);
    chart_canvas.set_height(80);
    chart_canvas.set_attribute(
        "style",
        &format!("background: {}; border-radius: 8px; border: 1px solid #1a1a2e;", BG_COLOR),
    )?;
    left_panel.append_child(&chart_canvas)?;

    // Scale toggle button
    let scale_btn = document.create_element("button")?;
    scale_btn.set_id("scale-btn");
    scale_btn.set_text_content(Some("Linear Scale"));
    scale_btn.set_attribute(
        "style",
        &format!(
            "width: 100%; padding: 8px; font-size: 12px; cursor: pointer; \
             background: #1a2a3a; color: {}; border: 1px solid {}; border-radius: 6px; \
             margin-top: 6px; font-weight: 500; transition: all 0.2s;",
            TEXT_COLOR, NEURON_STROKE
        ),
    )?;
    left_panel.append_child(&scale_btn)?;

    // Divider
    let divider = document.create_element("div")?;
    divider.set_attribute("style", "height: 1px; background: #1a1a2e; margin: 8px 0;")?;
    left_panel.append_child(&divider)?;

    // Sample Section
    let sample_label = document.create_element("div")?;
    sample_label.set_attribute(
        "style",
        &format!("color: {}; font-size: 12px; font-weight: 600;", TEXT_DIM),
    )?;
    sample_label.set_text_content(Some("CURRENT SAMPLE"));
    left_panel.append_child(&sample_label)?;

    // Random Image Button
    let load_btn = document.create_element("button")?;
    load_btn.set_id("load-btn");
    load_btn.set_text_content(Some("Random Image"));
    load_btn.set_attribute(
        "style",
        &format!(
            "width: 100%; padding: 10px; font-size: 13px; cursor: pointer; \
             background: transparent; color: {}; border: 1px solid {}; border-radius: 8px; \
             transition: all 0.2s;",
            NEURON_STROKE, NEURON_STROKE
        ),
    )?;
    left_panel.append_child(&load_btn)?;

    // Digit display
    let digit_canvas = document.create_element("canvas")?.dyn_into::<HtmlCanvasElement>()?;
    digit_canvas.set_id("digit-canvas");
    digit_canvas.set_width(140);
    digit_canvas.set_height(140);
    digit_canvas.set_attribute(
        "style",
        &format!("background: {}; border: 2px solid {}; border-radius: 8px; align-self: center;", BG_COLOR, NEURON_STROKE),
    )?;
    left_panel.append_child(&digit_canvas)?;

    // Prediction display
    let prediction_div = document.create_element("div")?;
    prediction_div.set_id("prediction-div");
    prediction_div.set_attribute(
        "style",
        &format!(
            "color: {}; text-align: center; font-size: 14px; padding: 8px; \
             background: {}; border-radius: 8px;",
            TEXT_COLOR, BG_COLOR
        ),
    )?;
    prediction_div.set_text_content(Some("Prediction: -"));
    left_panel.append_child(&prediction_div)?;

    // Spacer
    let spacer = document.create_element("div")?;
    spacer.set_attribute("style", "flex: 1;")?;
    left_panel.append_child(&spacer)?;

    // Instructions
    let instructions = document.create_element("div")?;
    instructions.set_attribute(
        "style",
        &format!("color: {}; font-size: 11px; line-height: 1.6;", TEXT_DIM),
    )?;
    instructions.set_inner_html("Scroll: Zoom<br>Drag: Pan<br>Hover: Details");
    left_panel.append_child(&instructions)?;

    container.append_child(&left_panel)?;

    // Main canvas
    let canvas = document.create_element("canvas")?.dyn_into::<HtmlCanvasElement>()?;
    canvas.set_id("main-canvas");
    canvas.set_width(canvas_width);
    canvas.set_height(canvas_height);
    canvas.set_attribute("style", "display: block; cursor: grab;")?;
    container.append_child(&canvas)?;

    body.append_child(&container)?;

    // Tooltip
    let tooltip = document.create_element("div")?;
    tooltip.set_id("tooltip");
    tooltip.set_attribute(
        "style",
        &format!(
            "position: fixed; \
             background: rgba(18, 18, 26, 0.95); \
             color: {}; \
             padding: 12px 16px; \
             border-radius: 8px; \
             font-size: 13px; \
             pointer-events: none; \
             display: none; \
             z-index: 1000; \
             border: 1px solid {}; \
             max-width: 280px; \
             box-shadow: 0 8px 32px rgba(0,0,0,0.4);",
            TEXT_COLOR, NEURON_STROKE
        ),
    )?;
    body.append_child(&tooltip)?;

    Ok(())
}

fn setup_handlers(document: &Document) -> Result<(), JsValue> {
    let canvas = document
        .get_element_by_id("main-canvas")
        .ok_or("no canvas")?
        .dyn_into::<HtmlCanvasElement>()?;

    // Mouse wheel for zoom
    let doc_clone = document.clone();
    let wheel_closure = Closure::wrap(Box::new(move |event: WheelEvent| {
        event.prevent_default();
        let x = event.offset_x() as f64;
        let y = event.offset_y() as f64;
        let delta = event.delta_y();

        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.zoom_at(x, y, delta);
            }
        });
        let _ = render(&doc_clone);
    }) as Box<dyn Fn(WheelEvent)>);
    canvas.add_event_listener_with_callback("wheel", wheel_closure.as_ref().unchecked_ref())?;
    wheel_closure.forget();

    // Mouse down
    let doc_clone = document.clone();
    let canvas_clone = canvas.clone();
    let mousedown_closure = Closure::wrap(Box::new(move |event: MouseEvent| {
        let x = event.offset_x() as f64;
        let y = event.offset_y() as f64;
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.start_drag(x, y);
            }
        });
        canvas_clone.style().set_property("cursor", "grabbing").unwrap();
        let _ = render(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    canvas.add_event_listener_with_callback("mousedown", mousedown_closure.as_ref().unchecked_ref())?;
    mousedown_closure.forget();

    // Mouse move
    let doc_clone = document.clone();
    let mousemove_closure = Closure::wrap(Box::new(move |event: MouseEvent| {
        let x = event.offset_x() as f64;
        let y = event.offset_y() as f64;

        let needs_render = STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.mouse_x = x;
                s.mouse_y = y;
                let was_dragging = s.is_dragging;
                s.drag(x, y);
                let old_hover = s.hovered_neuron;
                if !s.is_dragging {
                    s.hovered_neuron = s.get_neuron_at(x, y);
                }
                was_dragging || old_hover != s.hovered_neuron
            } else {
                false
            }
        });

        let _ = update_tooltip(&doc_clone, event.client_x() as f64, event.client_y() as f64);
        if needs_render {
            let _ = render(&doc_clone);
        }
    }) as Box<dyn Fn(MouseEvent)>);
    canvas.add_event_listener_with_callback("mousemove", mousemove_closure.as_ref().unchecked_ref())?;
    mousemove_closure.forget();

    // Mouse up
    let doc_clone = document.clone();
    let canvas_clone = canvas.clone();
    let mouseup_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.end_drag();
            }
        });
        canvas_clone.style().set_property("cursor", "grab").unwrap();
        let _ = render(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    canvas.add_event_listener_with_callback("mouseup", mouseup_closure.as_ref().unchecked_ref())?;
    mouseup_closure.forget();

    // Mouse leave
    let doc_clone = document.clone();
    let canvas_clone = canvas.clone();
    let mouseleave_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.end_drag();
                s.hovered_neuron = None;
            }
        });
        canvas_clone.style().set_property("cursor", "grab").unwrap();
        if let Some(tooltip) = doc_clone.get_element_by_id("tooltip") {
            let _ = tooltip.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "none");
        }
        let _ = render(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    canvas.add_event_listener_with_callback("mouseleave", mouseleave_closure.as_ref().unchecked_ref())?;
    mouseleave_closure.forget();

    // Load button
    let doc_clone = document.clone();
    let load_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.load_random_digit();
            }
        });
        let _ = render_digit(&doc_clone);
        let _ = render(&doc_clone);
        let _ = update_prediction(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("load-btn")
        .ok_or("no load button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(load_closure.as_ref().unchecked_ref()));
    load_closure.forget();

    // Train button
    let doc_clone = document.clone();
    let train_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        let should_start = STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                match s.training_state {
                    TrainingState::Idle | TrainingState::Paused => {
                        s.training_state = TrainingState::Training;
                        true
                    }
                    TrainingState::Training => {
                        s.training_state = TrainingState::Paused;
                        false
                    }
                }
            } else {
                false
            }
        });

        let _ = update_train_button(&doc_clone);

        if should_start {
            start_training_loop(doc_clone.clone());
        }
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("train-btn")
        .ok_or("no train button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(train_closure.as_ref().unchecked_ref()));
    train_closure.forget();

    // Reset button
    let doc_clone = document.clone();
    let reset_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.reset_network();
            }
        });
        let _ = update_train_button(&doc_clone);
        let _ = update_metrics(&doc_clone);
        let _ = render_loss_chart(&doc_clone);
        let _ = render(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("reset-btn")
        .ok_or("no reset button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(reset_closure.as_ref().unchecked_ref()));
    reset_closure.forget();

    // Scale toggle button
    let doc_clone = document.clone();
    let scale_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.metrics.log_scale = !s.metrics.log_scale;
            }
        });
        let _ = update_scale_button(&doc_clone);
        let _ = render_loss_chart(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("scale-btn")
        .ok_or("no scale button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(scale_closure.as_ref().unchecked_ref()));
    scale_closure.forget();

    // Digit canvas mouse handlers (for weight visualization on hover)
    let digit_canvas = document
        .get_element_by_id("digit-canvas")
        .ok_or("no digit canvas")?
        .dyn_into::<HtmlCanvasElement>()?;

    // Digit canvas mouse move - track which pixel is being hovered
    let doc_clone = document.clone();
    let digit_mousemove_closure = Closure::wrap(Box::new(move |event: MouseEvent| {
        let x = event.offset_x() as f64;
        let y = event.offset_y() as f64;

        // Convert canvas coords (140x140) to pixel index (28x28)
        // Each pixel is displayed at 5x scale
        let col = (x / 5.0).floor() as usize;
        let row = (y / 5.0).floor() as usize;

        if col < 28 && row < 28 {
            let pixel_idx = row * 28 + col;
            STATE.with(|state| {
                if let Some(ref mut s) = *state.borrow_mut() {
                    s.hovered_digit_pixel = Some(pixel_idx);
                }
            });
            let _ = render_digit(&doc_clone);
            let _ = render(&doc_clone);
            let _ = update_weight_tooltip(&doc_clone, event.client_x() as f64, event.client_y() as f64);
        }
    }) as Box<dyn Fn(MouseEvent)>);
    digit_canvas.add_event_listener_with_callback("mousemove", digit_mousemove_closure.as_ref().unchecked_ref())?;
    digit_mousemove_closure.forget();

    // Digit canvas mouse leave - clear hover
    let doc_clone = document.clone();
    let digit_mouseleave_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.hovered_digit_pixel = None;
            }
        });
        if let Some(tooltip) = doc_clone.get_element_by_id("tooltip") {
            let _ = tooltip.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "none");
        }
        let _ = render_digit(&doc_clone);
        let _ = render(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    digit_canvas.add_event_listener_with_callback("mouseleave", digit_mouseleave_closure.as_ref().unchecked_ref())?;
    digit_mouseleave_closure.forget();

    Ok(())
}

fn start_training_loop(document: Document) {
    let window = web_sys::window().unwrap();

    let closure = Closure::wrap(Box::new(move || {
        let is_training = STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                if s.training_state == TrainingState::Training {
                    s.train_batches(BATCHES_PER_FRAME);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        });

        if is_training {
            let _ = update_metrics(&document);
            let _ = render_loss_chart(&document);
            let _ = render(&document);

            // Schedule next frame
            start_training_loop(document.clone());
        }
    }) as Box<dyn Fn()>);

    let _ = window.request_animation_frame(closure.as_ref().unchecked_ref());
    closure.forget();
}

fn update_train_button(document: &Document) -> Result<(), JsValue> {
    let btn = document.get_element_by_id("train-btn").ok_or("no train button")?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            let (text, bg_color) = match s.training_state {
                TrainingState::Idle => ("Start Training", ACCENT_COLOR),
                TrainingState::Training => ("Pause Training", ACCENT_SECONDARY),
                TrainingState::Paused => ("Resume Training", ACCENT_COLOR),
            };
            btn.set_text_content(Some(text));
            let _ = btn.dyn_ref::<HtmlElement>().unwrap().style().set_property(
                "background",
                bg_color,
            );
        }
    });

    Ok(())
}

fn update_scale_button(document: &Document) -> Result<(), JsValue> {
    let btn = document.get_element_by_id("scale-btn").ok_or("no scale button")?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            let text = if s.metrics.log_scale { "Log Scale" } else { "Linear Scale" };
            btn.set_text_content(Some(text));
        }
    });

    Ok(())
}

fn update_metrics(document: &Document) -> Result<(), JsValue> {
    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            if let Some(el) = document.get_element_by_id("metric-epoch") {
                el.set_text_content(Some(&format!("Epoch: {}", s.metrics.epoch + 1)));
            }
            if let Some(el) = document.get_element_by_id("metric-batch") {
                el.set_text_content(Some(&format!(
                    "Batch: {}/{}",
                    s.metrics.batch + 1,
                    s.metrics.total_batches
                )));
            }
            if let Some(el) = document.get_element_by_id("metric-loss") {
                el.set_text_content(Some(&format!("Loss: {:.4}", s.metrics.loss)));
            }
            if let Some(el) = document.get_element_by_id("metric-accuracy") {
                el.set_text_content(Some(&format!("Accuracy: {:.1}%", s.metrics.accuracy * 100.0)));
            }
        }
    });

    Ok(())
}

fn render_loss_chart(document: &Document) -> Result<(), JsValue> {
    let canvas = document
        .get_element_by_id("loss-chart")
        .ok_or("no loss chart")?
        .dyn_into::<HtmlCanvasElement>()?;

    let ctx = canvas
        .get_context("2d")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()?;

    let width = canvas.width() as f64;
    let height = canvas.height() as f64;

    // Clear
    ctx.set_fill_style_str(BG_COLOR);
    ctx.fill_rect(0.0, 0.0, width, height);

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            let history = &s.metrics.loss_history;
            if history.len() < 2 {
                return;
            }

            let log_scale = s.metrics.log_scale;

            // Transform values if log scale
            let values: Vec<f64> = if log_scale {
                history.iter().map(|&v| (v.max(1e-7) as f64).ln()).collect()
            } else {
                history.iter().map(|&v| v as f64).collect()
            };

            // Find min/max for scaling (always show full history)
            let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = (max_val - min_val).max(0.1);

            // Draw line
            ctx.set_stroke_style_str(ACCENT_COLOR);
            ctx.set_line_width(1.5);
            ctx.begin_path();

            let padding = 10.0;
            let chart_width = width - 2.0 * padding;
            let chart_height = height - 2.0 * padding;

            for (i, &val) in values.iter().enumerate() {
                let x = padding + (i as f64 / (values.len() - 1) as f64) * chart_width;
                let y = padding + (1.0 - (val - min_val) / range) * chart_height;

                if i == 0 {
                    ctx.move_to(x, y);
                } else {
                    ctx.line_to(x, y);
                }
            }
            ctx.stroke();

            // Draw axis labels
            ctx.set_fill_style_str(TEXT_DIM);
            ctx.set_font("9px sans-serif");
            ctx.set_text_align("left");

            let max_display = if log_scale { max_val.exp() } else { max_val };
            let min_display = if log_scale { min_val.exp() } else { min_val };

            let _ = ctx.fill_text(&format!("{:.2}", max_display), 2.0, 10.0);
            let _ = ctx.fill_text(&format!("{:.2}", min_display), 2.0, height - 2.0);
        }
    });

    Ok(())
}

fn update_prediction(document: &Document) -> Result<(), JsValue> {
    let pred_div = document.get_element_by_id("prediction-div").ok_or("no prediction div")?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            if let (Some(digit_idx), Some(pred)) = (s.current_digit_idx, s.current_prediction) {
                let actual = s.digits[digit_idx].label();
                let correct = pred as u8 == actual;
                let color = if correct { ACCENT_COLOR } else { ACCENT_SECONDARY };

                pred_div.set_inner_html(&format!(
                    "<div>Predicted: <span style='color: {}; font-weight: 600;'>{}</span></div>\
                     <div style='font-size: 12px; color: {};'>Actual: {}</div>",
                    color, pred, TEXT_DIM, actual
                ));
            }
        }
    });

    Ok(())
}

fn update_tooltip(document: &Document, mouse_x: f64, mouse_y: f64) -> Result<(), JsValue> {
    let tooltip = document
        .get_element_by_id("tooltip")
        .ok_or("no tooltip")?
        .dyn_into::<HtmlElement>()?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            if let Some((layer_idx, neuron_idx)) = s.hovered_neuron {
                let layer_sizes = s.network.layer_sizes();
                let num_layers = layer_sizes.len();

                let layer_name = if layer_idx == 0 {
                    "Input Layer"
                } else if layer_idx == num_layers - 1 {
                    "Output Layer"
                } else {
                    "Hidden Layer"
                };

                let bias = s.network.get_bias(layer_idx, neuron_idx);

                // Get activation if available
                let activation = s.current_activations
                    .get(layer_idx)
                    .and_then(|a| a.get(neuron_idx))
                    .copied();

                let mut html = format!(
                    "<div style='margin-bottom: 8px; font-weight: 600; color: {};'>{}</div>\
                     <div style='margin-bottom: 4px;'>Neuron <b>#{}</b> of {}</div>",
                    NEURON_HOVER, layer_name, neuron_idx, layer_sizes[layer_idx]
                );

                if layer_idx > 0 {
                    html.push_str(&format!(
                        "<div style='margin-bottom: 4px;'>Bias: <b>{:.4}</b></div>",
                        bias
                    ));
                }

                if let Some(act) = activation {
                    html.push_str(&format!(
                        "<div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #2a3a4a;'>\
                         Activation: <b style='color: {};'>{:.4}</b></div>",
                        ACCENT_COLOR, act
                    ));
                }

                // Input layer: show weight stats
                if layer_idx == 0 {
                    let layer = &s.network.layers[0];
                    let mut min_w = f32::INFINITY;
                    let mut max_w = f32::NEG_INFINITY;
                    let mut sum_w = 0.0f32;
                    for j in 0..layer.output_size {
                        let w = layer.get_weight(neuron_idx, j);
                        min_w = min_w.min(w);
                        max_w = max_w.max(w);
                        sum_w += w;
                    }
                    let avg_w = sum_w / layer.output_size as f32;

                    html.push_str(&format!(
                        "<div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #2a3a4a;'>\
                         <div style='color: {}; font-size: 11px; margin-bottom: 4px;'>WEIGHTS TO HIDDEN</div>\
                         <div>Min: <span style='color: {};'>{:.4}</span></div>\
                         <div>Max: <span style='color: {};'>{:.4}</span></div>\
                         <div>Avg: <b>{:.4}</b></div>\
                         </div>",
                        TEXT_DIM,
                        ACCENT_SECONDARY, min_w,
                        ACCENT_COLOR, max_w,
                        avg_w
                    ));
                }

                // Output layer: show class name
                if layer_idx == num_layers - 1 {
                    html.push_str(&format!(
                        "<div style='margin-top: 4px; color: {};'>Class: Digit {}</div>",
                        TEXT_DIM, neuron_idx
                    ));
                }

                tooltip.set_inner_html(&html);
                tooltip.style().set_property("display", "block").unwrap();
                tooltip.style().set_property("left", &format!("{}px", mouse_x + 15.0)).unwrap();
                tooltip.style().set_property("top", &format!("{}px", mouse_y + 15.0)).unwrap();
            } else {
                tooltip.style().set_property("display", "none").unwrap();
            }
        }
    });

    Ok(())
}

/// Update tooltip for weight visualization when hovering digit canvas
fn update_weight_tooltip(document: &Document, mouse_x: f64, mouse_y: f64) -> Result<(), JsValue> {
    let tooltip = document
        .get_element_by_id("tooltip")
        .ok_or("no tooltip")?
        .dyn_into::<HtmlElement>()?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            if let Some(pixel_idx) = s.hovered_digit_pixel {
                let row = pixel_idx / 28;
                let col = pixel_idx % 28;

                // Get pixel value
                let pixel_val = s.current_digit_idx
                    .and_then(|idx| s.digits.get(idx))
                    .map(|d| d.pixels()[pixel_idx])
                    .unwrap_or(0);

                // Get weight stats from this input neuron to hidden layer
                let layer = &s.network.layers[0];
                let mut min_w = f32::INFINITY;
                let mut max_w = f32::NEG_INFINITY;
                let mut sum_w = 0.0f32;
                let mut sum_abs_w = 0.0f32;

                for j in 0..layer.output_size {
                    let w = layer.get_weight(pixel_idx, j);
                    min_w = min_w.min(w);
                    max_w = max_w.max(w);
                    sum_w += w;
                    sum_abs_w += w.abs();
                }
                let avg_w = sum_w / layer.output_size as f32;
                let avg_abs_w = sum_abs_w / layer.output_size as f32;

                let html = format!(
                    "<div style='margin-bottom: 8px; font-weight: 600; color: {};'>Input Pixel</div>\
                     <div style='margin-bottom: 4px;'>Position: <b>({}, {})</b></div>\
                     <div style='margin-bottom: 4px;'>Index: <b>#{}</b></div>\
                     <div style='margin-bottom: 8px;'>Value: <b>{}</b> ({:.1}%)</div>\
                     <div style='padding-top: 8px; border-top: 1px solid #2a3a4a;'>\
                     <div style='color: {}; font-size: 11px; margin-bottom: 4px;'>WEIGHTS TO HIDDEN ({} connections)</div>\
                     <div>Min: <span style='color: {};'>{:.4}</span></div>\
                     <div>Max: <span style='color: {};'>{:.4}</span></div>\
                     <div>Avg: <b>{:.4}</b></div>\
                     <div>Avg |w|: <b>{:.4}</b></div>\
                     </div>",
                    NEURON_HOVER,
                    col, row,
                    pixel_idx,
                    pixel_val, pixel_val as f32 / 255.0 * 100.0,
                    TEXT_DIM, layer.output_size,
                    ACCENT_SECONDARY, min_w,
                    ACCENT_COLOR, max_w,
                    avg_w,
                    avg_abs_w
                );

                tooltip.set_inner_html(&html);
                tooltip.style().set_property("display", "block").unwrap();
                tooltip.style().set_property("left", &format!("{}px", mouse_x + 15.0)).unwrap();
                tooltip.style().set_property("top", &format!("{}px", mouse_y + 15.0)).unwrap();
            }
        }
    });

    Ok(())
}

fn render_digit(document: &Document) -> Result<(), JsValue> {
    let canvas = document
        .get_element_by_id("digit-canvas")
        .ok_or("no digit canvas")?
        .dyn_into::<HtmlCanvasElement>()?;

    let ctx = canvas
        .get_context("2d")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            if let Some(digit_idx) = s.current_digit_idx {
                let digit = &s.digits[digit_idx];

                // Clear
                ctx.set_fill_style_str(BG_COLOR);
                ctx.fill_rect(0.0, 0.0, 140.0, 140.0);

                // Draw digit scaled 5x
                let pixels = digit.pixels();
                for row in 0..28 {
                    for col in 0..28 {
                        let pixel_idx = row * 28 + col;
                        let pixel = pixels[pixel_idx];
                        let is_hovered = s.hovered_digit_pixel == Some(pixel_idx);

                        if pixel > 10 || is_hovered {
                            // Color based on intensity
                            let intensity = pixel as f64 / 255.0;
                            let r = (78.0 * (1.0 - intensity) + 78.0 * intensity) as u8;
                            let g = (205.0 * (1.0 - intensity) + 205.0 * intensity) as u8;
                            let b = (196.0 * (1.0 - intensity) + 196.0 * intensity) as u8;
                            ctx.set_fill_style_str(&format!("rgb({},{},{})", r, g, b));
                            ctx.fill_rect(col as f64 * 5.0, row as f64 * 5.0, 5.0, 5.0);
                        }

                        // Draw highlight border for hovered pixel
                        if is_hovered {
                            ctx.set_stroke_style_str(ACCENT_COLOR);
                            ctx.set_line_width(2.0);
                            ctx.stroke_rect(col as f64 * 5.0, row as f64 * 5.0, 5.0, 5.0);
                        }
                    }
                }
            }
        }
    });

    let _ = update_prediction(document);
    Ok(())
}

fn render(document: &Document) -> Result<(), JsValue> {
    let canvas = document
        .get_element_by_id("main-canvas")
        .ok_or("no canvas")?
        .dyn_into::<HtmlCanvasElement>()?;

    let ctx = canvas
        .get_context("2d")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()?;

    STATE.with(|state| {
        let state = state.borrow();
        let state = state.as_ref().ok_or("state not initialized")?;

        // Clear canvas
        ctx.set_fill_style_str(BG_COLOR);
        ctx.fill_rect(0.0, 0.0, state.canvas_width, state.canvas_height);

        // Apply transform
        ctx.save();
        ctx.translate(state.offset_x, state.offset_y)?;
        ctx.scale(state.zoom, state.zoom)?;

        // Draw layer labels
        let layer_sizes = state.network.layer_sizes();
        let num_layers = layer_sizes.len();

        ctx.set_fill_style_str(TEXT_COLOR);
        ctx.set_font("bold 16px 'Inter', sans-serif");
        ctx.set_text_align("center");

        for (i, &size) in layer_sizes.iter().enumerate() {
            let label = if i == 0 {
                format!("Input\n({})", size)
            } else if i == num_layers - 1 {
                format!("Output\n({})", size)
            } else {
                format!("Hidden\n({})", size)
            };

            if let Some((x, _)) = state.get_position(i, 0) {
                ctx.set_fill_style_str(TEXT_DIM);
                let _ = ctx.fill_text(&label.split('\n').next().unwrap_or(""), x, 25.0);
                ctx.set_fill_style_str(TEXT_DIM);
                ctx.set_font("12px 'Inter', sans-serif");
                let _ = ctx.fill_text(&format!("({})", size), x, 45.0);
                ctx.set_font("bold 16px 'Inter', sans-serif");
            }
        }

        // Draw weights (connections)
        let hovered = state.hovered_neuron;
        let hovered_input = state.get_hovered_input();

        ctx.set_line_width(0.5 / state.zoom);
        ctx.set_global_alpha(0.15);

        // Draw input->hidden weights when an input neuron is hovered
        if let Some(input_idx) = hovered_input {
            let layer = &state.network.layers[0];
            let hidden_size = layer_sizes[1]; // Show all hidden neurons

            // Find weight magnitude range for color scaling
            let mut max_abs_weight = 0.0f32;
            for j in 0..layer.output_size {
                let w = layer.get_weight(input_idx, j).abs();
                if w > max_abs_weight {
                    max_abs_weight = w;
                }
            }

            // Get input neuron position
            if let Some((x1, y1)) = state.get_position(0, input_idx) {
                for to_idx in 0..hidden_size {
                    if let Some((x2, y2)) = state.get_position(1, to_idx) {
                        let weight = layer.get_weight(input_idx, to_idx);
                        let normalized = if max_abs_weight > 0.0 {
                            weight.abs() / max_abs_weight
                        } else {
                            0.0
                        };

                        // Color: positive weights = teal, negative weights = red
                        if weight >= 0.0 {
                            ctx.set_stroke_style_str(ACCENT_COLOR);
                        } else {
                            ctx.set_stroke_style_str(ACCENT_SECONDARY);
                        }

                        // Alpha and line width based on magnitude
                        let alpha = 0.3 + normalized as f64 * 0.6;
                        let line_width = (0.5 + normalized as f64 * 2.5) / state.zoom;

                        ctx.set_global_alpha(alpha);
                        ctx.set_line_width(line_width);

                        ctx.begin_path();
                        ctx.move_to(x1, y1);
                        ctx.line_to(x2, y2);
                        ctx.stroke();
                    }
                }
            }
        }

        // Draw hidden->output weights
        for layer_idx in 1..state.network.num_layers() {
            let from_size = layer_sizes[layer_idx]; // Show all neurons
            let to_size = layer_sizes[layer_idx + 1];

            for from_idx in 0..from_size {
                for to_idx in 0..to_size {
                    if let (Some((x1, y1)), Some((x2, y2))) = (
                        state.get_position(layer_idx, from_idx),
                        state.get_position(layer_idx + 1, to_idx),
                    ) {
                        let is_connected = hovered.map_or(false, |(l, n)| {
                            (l == layer_idx && n == from_idx) || (l == layer_idx + 1 && n == to_idx)
                        });

                        if is_connected {
                            ctx.set_stroke_style_str(WEIGHT_HIGHLIGHT);
                            ctx.set_global_alpha(0.8);
                            ctx.set_line_width(2.0 / state.zoom);
                        } else {
                            ctx.set_stroke_style_str(WEIGHT_COLOR);
                            ctx.set_global_alpha(0.15);
                            ctx.set_line_width(0.5 / state.zoom);
                        }

                        ctx.begin_path();
                        ctx.move_to(x1, y1);
                        ctx.line_to(x2, y2);
                        ctx.stroke();
                    }
                }
            }
        }

        ctx.set_global_alpha(1.0);

        // Draw neurons
        let input_cell_size = 4.0;
        let hidden_cell_size = 4.0;

        for pos in &state.neuron_positions {
            let is_hovered = hovered.map_or(false, |(l, n)| l == pos.layer_idx && n == pos.neuron_idx);
            let is_input_layer = pos.layer_idx == 0;
            let is_hidden_layer = pos.layer_idx == 1 && layer_sizes[1] > 20;
            let is_output_layer = pos.layer_idx == num_layers - 1;
            // Check if this input neuron is hovered (either from main canvas or digit canvas)
            let is_input_hovered = pos.layer_idx == 0 && hovered_input == Some(pos.neuron_idx);

            // Get activation value for this neuron
            let activation = state.current_activations
                .get(pos.layer_idx)
                .and_then(|a| a.get(pos.neuron_idx))
                .copied()
                .unwrap_or(0.0);

            // Color based on activation
            let fill_color = if activation > 0.01 {
                let t = activation.min(1.0) as f64;
                let r = (26.0 * (1.0 - t) + 78.0 * t) as u8;
                let g = (42.0 * (1.0 - t) + 205.0 * t) as u8;
                let b = (74.0 * (1.0 - t) + 196.0 * t) as u8;
                format!("rgb({},{},{})", r, g, b)
            } else {
                NEURON_BASE.to_string()
            };

            if is_input_layer {
                // Input layer: draw as small square (pixel)
                ctx.set_fill_style_str(&fill_color);
                ctx.fill_rect(
                    pos.x - input_cell_size / 2.0,
                    pos.y - input_cell_size / 2.0,
                    input_cell_size,
                    input_cell_size,
                );

                // Highlight hovered input pixel
                if is_input_hovered {
                    ctx.set_stroke_style_str(ACCENT_COLOR);
                    ctx.set_line_width(2.0 / state.zoom);
                    ctx.stroke_rect(
                        pos.x - input_cell_size / 2.0 - 1.0,
                        pos.y - input_cell_size / 2.0 - 1.0,
                        input_cell_size + 2.0,
                        input_cell_size + 2.0,
                    );
                }
            } else if is_hidden_layer {
                // Hidden layer: draw as small squares in column
                ctx.set_fill_style_str(&fill_color);
                ctx.fill_rect(
                    pos.x - hidden_cell_size / 2.0,
                    pos.y - hidden_cell_size / 2.0,
                    hidden_cell_size,
                    hidden_cell_size,
                );

                // Highlight if hovered
                if is_hovered {
                    ctx.set_stroke_style_str(ACCENT_COLOR);
                    ctx.set_line_width(2.0 / state.zoom);
                    ctx.stroke_rect(
                        pos.x - hidden_cell_size / 2.0 - 2.0,
                        pos.y - hidden_cell_size / 2.0 - 2.0,
                        hidden_cell_size + 4.0,
                        hidden_cell_size + 4.0,
                    );
                }
            } else if is_output_layer {
                // Output layer: draw as circles with labels

                // Draw glow for active neurons
                if activation > 0.1 {
                    let glow_alpha = (activation * 0.4).min(0.4) as f64;
                    let glow_radius = NEURON_RADIUS * (1.5 + activation as f64 * 0.5);

                    ctx.begin_path();
                    let _ = ctx.arc(pos.x, pos.y, glow_radius, 0.0, std::f64::consts::TAU);
                    ctx.set_fill_style_str(ACCENT_COLOR);
                    ctx.set_global_alpha(glow_alpha);
                    ctx.fill();
                    ctx.set_global_alpha(1.0);
                }

                // Neuron circle
                ctx.begin_path();
                let _ = ctx.arc(pos.x, pos.y, NEURON_RADIUS, 0.0, std::f64::consts::TAU);

                if is_hovered {
                    ctx.set_fill_style_str(NEURON_HOVER);
                    ctx.set_stroke_style_str("#ffffff");
                    ctx.set_line_width(2.5 / state.zoom);
                } else {
                    ctx.set_fill_style_str(&fill_color);
                    ctx.set_stroke_style_str(NEURON_STROKE);
                    ctx.set_line_width(1.5 / state.zoom);
                }

                ctx.fill();
                ctx.stroke();

                // Draw output layer labels
                ctx.set_fill_style_str(TEXT_COLOR);
                ctx.set_font(&format!("{}px 'Inter', sans-serif", (10.0 / state.zoom.sqrt()).max(8.0)));
                ctx.set_text_align("center");
                ctx.set_text_baseline("middle");
                let _ = ctx.fill_text(&pos.neuron_idx.to_string(), pos.x, pos.y);
            }
        }

        ctx.restore();

        Ok(())
    })
}
