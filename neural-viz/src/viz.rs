use common::{parse_digits_from_bytes, Digit};
use neural::{Network, mnist::MnistSample};
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, MouseEvent, WheelEvent};

use neural::{GpuNetwork, WeightSyncState};

const INPUT_PANEL_WIDTH: f64 = 200.0;
const NEURON_RADIUS: f64 = 8.0;  // Smaller neurons
const OUTPUT_NEURON_RADIUS: f64 = 18.0;  // Larger output neurons for visibility
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
    Completed,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum AcceleratorMode {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, PartialEq)]
enum GpuInitState {
    NotStarted,
    Initializing,
    Ready,
    Failed,
}

#[derive(Clone, Copy, PartialEq)]
enum ChartScale {
    Linear,
    Log,
    LogLog,
}

struct TrainingMetrics {
    epoch: usize,
    batch: usize,
    total_batches: usize,
    samples_seen: usize,
    loss: f32,
    accuracy: f32,
    loss_history: Vec<f32>,
    chart_scale: ChartScale,
    // Timing for FLOPS calculation
    training_start_time: Option<f64>,
    flops_per_sample: usize,
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
            chart_scale: ChartScale::Linear,
            training_start_time: None,
            flops_per_sample: 0,
        }
    }
}

/// Calculate FLOPs per sample for a neural network
/// For each layer: forward + backward pass
fn calculate_flops_per_sample(layer_sizes: &[usize]) -> usize {
    let mut total_flops = 0;

    for i in 0..layer_sizes.len() - 1 {
        let input_size = layer_sizes[i];
        let output_size = layer_sizes[i + 1];

        // Forward pass:
        // - Matrix multiply: 2 * in * out (multiply + add)
        // - Bias addition: out
        // - Activation: out (ReLU comparison or softmax)
        let forward = 2 * input_size * output_size + 2 * output_size;

        // Backward pass (roughly 3x forward):
        // - Error backprop: 2 * in * out
        // - Weight gradients: 2 * in * out
        // - Weight updates: 2 * in * out
        // - Bias updates: 2 * out
        let backward = 6 * input_size * output_size + 2 * output_size;

        total_flops += forward + backward;
    }

    total_flops
}

struct State {
    network: Network,
    // Training data - used only for training
    train_digits: Vec<Digit>,
    samples: Vec<MnistSample>,
    // Validation data - never trained on, used for accuracy metrics and Random Image
    validation_digits: Vec<Digit>,
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
    // Accelerator mode (CPU vs GPU)
    accelerator_mode: AcceleratorMode,
    gpu_available: bool,
    gpu_init_state: GpuInitState,
    // Drawing mode for custom digit input
    drawing_mode: bool,
    drawn_pixels: [u8; 784],
    is_drawing: bool,
    last_draw_pos: Option<(f64, f64)>,
}

// Number of samples to hold out for validation (never trained on)
const VALIDATION_SIZE: usize = 4000;

impl State {
    fn new(network: Network, all_digits: Vec<Digit>, canvas_width: f64, canvas_height: f64, gpu_available: bool) -> Self {
        let neuron_positions = Self::calculate_positions(&network, canvas_width, canvas_height);
        let num_layers = network.layer_sizes().len();

        // Split data: last VALIDATION_SIZE samples for validation, rest for training
        let split_point = all_digits.len().saturating_sub(VALIDATION_SIZE);
        let mut all_digits = all_digits;
        let validation_digits: Vec<Digit> = all_digits.split_off(split_point);
        let train_digits = all_digits; // Remaining digits are for training

        // Only create training samples from training data
        let samples: Vec<MnistSample> = train_digits.iter().map(|d| MnistSample::new(d.clone())).collect();

        // Calculate FLOPs per sample for this network architecture
        let flops_per_sample = calculate_flops_per_sample(&network.layer_sizes());

        let mut metrics = TrainingMetrics::default();
        metrics.flops_per_sample = flops_per_sample;

        Self {
            network,
            train_digits,
            samples,
            validation_digits,
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
            metrics,
            current_batch_idx: 0,
            current_activations: vec![Vec::new(); num_layers],
            current_prediction: None,
            accelerator_mode: get_default_accelerator_mode(gpu_available),
            gpu_available,
            gpu_init_state: GpuInitState::NotStarted,
            drawing_mode: false,
            drawn_pixels: [0u8; 784],
            is_drawing: false,
            last_draw_pos: None,
        }
    }

    /// Preprocess drawn pixels to match MNIST format:
    /// - Center by center of mass
    /// - Scale to fit in ~20x20 box (MNIST standard)
    fn preprocess_drawn(&self) -> [u8; 784] {
        let mut result = [0u8; 784];

        // Find bounding box and center of mass
        let mut min_x = 28i32;
        let mut max_x = 0i32;
        let mut min_y = 28i32;
        let mut max_y = 0i32;
        let mut mass_x = 0.0f64;
        let mut mass_y = 0.0f64;
        let mut total_mass = 0.0f64;

        for y in 0..28 {
            for x in 0..28 {
                let val = self.drawn_pixels[y * 28 + x] as f64;
                if val > 0.0 {
                    min_x = min_x.min(x as i32);
                    max_x = max_x.max(x as i32);
                    min_y = min_y.min(y as i32);
                    max_y = max_y.max(y as i32);
                    mass_x += x as f64 * val;
                    mass_y += y as f64 * val;
                    total_mass += val;
                }
            }
        }

        // If nothing drawn, return empty
        if total_mass == 0.0 {
            return result;
        }

        // Center of mass
        let com_x = mass_x / total_mass;
        let com_y = mass_y / total_mass;

        // Bounding box size
        let width = (max_x - min_x + 1) as f64;
        let height = (max_y - min_y + 1) as f64;

        // MNIST digits fit in ~20x20 box, centered in 28x28
        // Scale factor to fit the larger dimension into 20 pixels
        let target_size = 20.0;
        let scale = if width > height {
            target_size / width
        } else {
            target_size / height
        };

        // Don't upscale small drawings too much (max 2x)
        let scale = scale.min(2.0);

        // Target center is 13.5, 13.5 (center of 28x28 grid)
        let target_cx = 13.5;
        let target_cy = 13.5;

        // For each pixel in output, sample from input with bilinear interpolation
        for out_y in 0..28 {
            for out_x in 0..28 {
                // Map output pixel back to input coordinates
                // out = (in - com) * scale + target_center
                // in = (out - target_center) / scale + com
                let in_x = (out_x as f64 - target_cx) / scale + com_x;
                let in_y = (out_y as f64 - target_cy) / scale + com_y;

                // Bilinear interpolation
                let x0 = in_x.floor() as i32;
                let y0 = in_y.floor() as i32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let fx = in_x - x0 as f64;
                let fy = in_y - y0 as f64;

                let get_pixel = |x: i32, y: i32| -> f64 {
                    if x >= 0 && x < 28 && y >= 0 && y < 28 {
                        self.drawn_pixels[(y * 28 + x) as usize] as f64
                    } else {
                        0.0
                    }
                };

                let v00 = get_pixel(x0, y0);
                let v10 = get_pixel(x1, y0);
                let v01 = get_pixel(x0, y1);
                let v11 = get_pixel(x1, y1);

                let value = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                result[out_y * 28 + out_x] = (value.min(255.0).max(0.0)) as u8;
            }
        }

        result
    }

    /// Classify the drawn pixels and update activations
    fn classify_drawn(&mut self) {
        // Preprocess to center and scale like MNIST
        let preprocessed = self.preprocess_drawn();
        let input: Vec<f32> = preprocessed.iter().map(|&p| p as f32 / 255.0).collect();
        let output = self.network.forward(&input);

        // Store activations for each layer (using preprocessed input)
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

    /// Clear the drawing canvas
    fn clear_drawing(&mut self) {
        self.drawn_pixels = [0u8; 784];
        self.current_activations.clear();
        self.current_prediction = None;
    }

    /// Draw at a position on the 28x28 grid with MNIST-like brush
    /// MNIST digits have strokes ~2-3 pixels wide with Gaussian-like anti-aliasing
    fn draw_at(&mut self, canvas_x: f64, canvas_y: f64, canvas_size: f64) {
        // Convert canvas position to 28x28 grid (floating point for sub-pixel accuracy)
        let scale = 28.0 / canvas_size;
        let grid_x = canvas_x * scale;
        let grid_y = canvas_y * scale;

        // MNIST-like brush: ~2-3 pixel stroke width with smooth Gaussian falloff
        // Brush radius in grid space - MNIST strokes are typically 2-3 pixels wide
        let brush_radius: f64 = 1.8;
        let sigma: f64 = 0.9; // Controls the falloff steepness

        // Check pixels in a 5x5 area around the brush center
        let center_x = grid_x.floor() as i32;
        let center_y = grid_y.floor() as i32;

        for dy in -2..=2 {
            for dx in -2..=2 {
                let px = center_x + dx;
                let py = center_y + dy;
                if px >= 0 && px < 28 && py >= 0 && py < 28 {
                    let idx = (py * 28 + px) as usize;

                    // Calculate distance from brush center to pixel center
                    let pixel_center_x = px as f64 + 0.5;
                    let pixel_center_y = py as f64 + 0.5;
                    let dist = ((grid_x - pixel_center_x).powi(2) + (grid_y - pixel_center_y).powi(2)).sqrt();

                    // Gaussian falloff matching MNIST anti-aliasing
                    // MNIST has bright cores (255) that fall off smoothly to edges
                    let intensity = if dist < brush_radius {
                        // Gaussian: exp(-dist^2 / (2 * sigma^2))
                        let falloff = (-dist.powi(2) / (2.0 * sigma.powi(2))).exp();
                        (255.0 * falloff) as u8
                    } else {
                        0
                    };

                    // Max blend for overlapping strokes (like real pen strokes)
                    if intensity > 0 {
                        self.drawn_pixels[idx] = self.drawn_pixels[idx].max(intensity);
                    }
                }
            }
        }
    }

    /// Draw a line between two points (for smooth strokes)
    fn draw_line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, canvas_size: f64) {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist = (dx * dx + dy * dy).sqrt();
        // Step more frequently for smoother lines (every 1.5 canvas pixels)
        let steps = (dist / 1.5).max(1.0) as i32;

        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let x = x1 + dx * t;
            let y = y1 + dy * t;
            self.draw_at(x, y, canvas_size);
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
                // Output layer: column layout (only 10 neurons, larger size)
                let layer_x = INPUT_PANEL_WIDTH + 100.0 + (layer_idx as f64) * LAYER_SPACING;

                // Calculate spacing to fit all neurons with room (use larger output radius)
                let neuron_spacing = (OUTPUT_NEURON_RADIUS * 2.5).min(available_height / actual_size as f64);
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
        let num_layers = self.network.layers.len();

        for pos in &self.neuron_positions {
            // Use larger radius for output layer
            let is_output = pos.layer_idx == num_layers;
            let base_radius = if is_output { OUTPUT_NEURON_RADIUS } else { NEURON_RADIUS };
            let hit_radius = base_radius / self.zoom.sqrt();

            let dx = world_x - pos.x;
            let dy = world_y - pos.y;
            if dx * dx + dy * dy <= (base_radius + hit_radius) * (base_radius + hit_radius) {
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
        // Use validation set for Random Image - these are never trained on
        let next_idx = rng.gen_range(0..self.validation_digits.len());
        self.current_digit_idx = Some(next_idx);

        // Run forward pass using the validation digit
        // This tests true generalization since these images were never trained on
        let digit = &self.validation_digits[next_idx];
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
        // Reset metrics but preserve flops_per_sample (network architecture unchanged)
        let flops_per_sample = self.metrics.flops_per_sample;
        self.metrics = TrainingMetrics::default();
        self.metrics.flops_per_sample = flops_per_sample;
        self.current_batch_idx = 0;
        self.current_activations.clear();
        self.current_prediction = None;
        // Reset GPU state - need to reinitialize after reset
        self.gpu_init_state = GpuInitState::NotStarted;
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
                    self.training_state = TrainingState::Completed;
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
            // Use validation_digits - these are NEVER trained on
            // This gives us a true measure of generalization
            if self.current_batch_idx % 10 == 0 {
                // Use first 500 of validation set for speed (still never trained on)
                let val_size = 500.min(self.validation_digits.len());
                let eval_inputs: Vec<Vec<f32>> = self.validation_digits[..val_size]
                    .iter()
                    .map(|d| d.pixels().iter().map(|&p| p as f32 / 255.0).collect())
                    .collect();
                let eval_labels: Vec<u8> = self.validation_digits[..val_size]
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

thread_local! {
    static GPU_NETWORK: std::cell::RefCell<Option<GpuNetwork>> = const { std::cell::RefCell::new(None) };
}

pub fn init(csv_data: &[u8]) -> Result<(), JsValue> {
    let digits = parse_digits_from_bytes(csv_data).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let network = Network::mnist_default();

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;

    let canvas_width = window.inner_width()?.as_f64().unwrap_or(1200.0) - INPUT_PANEL_WIDTH;
    let canvas_height = window.inner_height()?.as_f64().unwrap_or(800.0);

    // Check if WebGPU is available
    let gpu_available = check_webgpu_available(&window);

    STATE.with(|state| {
        *state.borrow_mut() = Some(State::new(network, digits, canvas_width, canvas_height, gpu_available));
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

    // Auto-initialize GPU if available and GPU mode is default
    if gpu_available {
        init_gpu_async(document.clone());
    }

    Ok(())
}

/// Check if WebGPU is available in the browser
fn check_webgpu_available(_window: &web_sys::Window) -> bool {
    // WebGPU availability check - the navigator.gpu property exists when WebGPU is supported
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

/// Initialize GPU network asynchronously
fn init_gpu_async(document: Document) {
    use wasm_bindgen_futures::spawn_local;

    // Mark as initializing
    STATE.with(|state| {
        if let Some(ref mut s) = *state.borrow_mut() {
            s.gpu_init_state = GpuInitState::Initializing;
        }
    });
    let _ = update_gpu_button(&document);

    // Clone network weights to initialize GPU network with same state
    let network_clone = STATE.with(|state| {
        state.borrow().as_ref().map(|s| s.network.clone())
    });

    let doc = document.clone();
    spawn_local(async move {
        if let Some(network) = network_clone {
            match GpuNetwork::from_network_async(network).await {
                Ok(gpu_net) => {
                    // Store GPU network
                    GPU_NETWORK.with(|gpu| {
                        *gpu.borrow_mut() = Some(gpu_net);
                    });

                    // Mark as ready
                    STATE.with(|state| {
                        if let Some(ref mut s) = *state.borrow_mut() {
                            s.gpu_init_state = GpuInitState::Ready;
                        }
                    });
                    web_sys::console::log_1(&"GPU network initialized successfully".into());
                }
                Err(e) => {
                    // Mark as failed and auto-fallback to CPU
                    STATE.with(|state| {
                        if let Some(ref mut s) = *state.borrow_mut() {
                            s.gpu_init_state = GpuInitState::Failed;
                            s.accelerator_mode = AcceleratorMode::Cpu;
                        }
                    });
                    web_sys::console::error_1(&format!("GPU initialization failed, falling back to CPU: {}", e).into());
                }
            }
        }
        let _ = update_gpu_button(&doc);
    });
}

/// Sync CPU network weights to GPU network (if initialized)
#[allow(dead_code)]
fn sync_network_to_gpu() {
    STATE.with(|state| {
        if let Some(ref s) = *state.borrow() {
            GPU_NETWORK.with(|gpu| {
                if let Some(ref mut gpu_net) = *gpu.borrow_mut() {
                    // Copy weights from CPU network to GPU network
                    let cpu_net = &s.network;
                    for (gpu_layer, cpu_layer) in gpu_net.network_mut().layers.iter_mut().zip(cpu_net.layers.iter()) {
                        gpu_layer.weights.copy_from_slice(&cpu_layer.weights);
                        gpu_layer.biases.copy_from_slice(&cpu_layer.biases);
                    }
                }
            });
        }
    });
}

/// Sync GPU network weights back to CPU network
#[allow(dead_code)]
fn sync_network_from_gpu() {
    GPU_NETWORK.with(|gpu| {
        if let Some(ref gpu_net) = *gpu.borrow() {
            STATE.with(|state| {
                if let Some(ref mut s) = *state.borrow_mut() {
                    for (cpu_layer, gpu_layer) in s.network.layers.iter_mut().zip(gpu_net.network().layers.iter()) {
                        cpu_layer.weights.copy_from_slice(&gpu_layer.weights);
                        cpu_layer.biases.copy_from_slice(&gpu_layer.biases);
                    }
                }
            });
        }
    });
}

/// Train batches using GPU acceleration
///
/// Full GPU training - works in both WASM and native
///
/// This uses `train_batch_gpu_full()` which runs the entire training loop on GPU:
/// - Forward pass: matmul -> add_bias -> activation (relu/softmax)
/// - Backward pass: output_delta -> matmul for gradients -> relu_backward
/// - Weight updates: saxpy for SGD
///
/// No CPU readback during training - only sync weights periodically for metrics.
fn train_batches_gpu(state: &mut State, num_batches: usize) {
    let total_samples = state.samples.len();
    let num_total_batches = total_samples / BATCH_SIZE;

    GPU_NETWORK.with(|gpu| {
        let mut gpu_borrow = gpu.borrow_mut();
        let gpu_net = match gpu_borrow.as_mut() {
            Some(g) => g,
            None => {
                web_sys::console::error_1(&"GPU network not initialized".into());
                return;
            }
        };

        // Initialize persistent buffers if not already done
        if !gpu_net.has_persistent_buffers() {
            gpu_net.init_persistent_buffers();
        }

        for _ in 0..num_batches {
            if state.training_state != TrainingState::Training {
                break;
            }

            // Get batch
            let start = state.current_batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(total_samples);

            if start >= total_samples {
                // End of epoch - sync weights back to CPU and update metrics
                state.metrics.epoch += 1;
                state.current_batch_idx = 0;

                // Sync weights from GPU to CPU for evaluation
                #[cfg(not(target_arch = "wasm32"))]
                {
                    gpu_net.sync_weights_to_cpu();
                    // Copy synced weights to visualization state
                    for (cpu_layer, gpu_layer) in state.network.layers.iter_mut().zip(gpu_net.network().layers.iter()) {
                        cpu_layer.weights.copy_from_slice(&gpu_layer.weights);
                        cpu_layer.biases.copy_from_slice(&gpu_layer.biases);
                    }
                }

                // Evaluate accuracy at epoch end using validation set (never trained on)
                let val_size = 500.min(state.validation_digits.len());
                let eval_inputs: Vec<Vec<f32>> = state.validation_digits[..val_size]
                    .iter()
                    .map(|d| d.pixels().iter().map(|&p| p as f32 / 255.0).collect())
                    .collect();
                let eval_labels: Vec<u8> = state.validation_digits[..val_size]
                    .iter()
                    .map(|d| d.label())
                    .collect();

                #[cfg(not(target_arch = "wasm32"))]
                {
                    let (loss, acc) = state.network.evaluate(&eval_inputs, &eval_labels);
                    state.metrics.accuracy = acc;
                    state.metrics.loss = loss;
                }

                // For WASM, start async weight sync - completion handled at next frame start
                #[cfg(target_arch = "wasm32")]
                {
                    // Always start a sync at epoch boundary for accurate metrics
                    if gpu_net.weight_sync_state() == WeightSyncState::Idle {
                        gpu_net.start_weight_sync();
                    }
                }

                // Stop training if accuracy is good enough or max epochs reached
                if state.metrics.accuracy >= 0.99 || state.metrics.epoch >= 10 {
                    state.training_state = TrainingState::Completed;
                    break;
                }

                // Shuffle samples for next epoch
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                state.samples.shuffle(&mut rng);
                continue;
            }

            let inputs: Vec<Vec<f32>> = state.samples[start..end]
                .iter()
                .map(|s| s.normalized_pixels_f32())
                .collect();
            let labels: Vec<u8> = state.samples[start..end].iter().map(|s| s.label()).collect();

            // Train batch on GPU using full GPU training (no CPU readback)
            let samples_trained = gpu_net.train_batch_gpu_full(&inputs, &labels, LEARNING_RATE);

            // Update metrics
            state.metrics.batch = state.current_batch_idx;
            state.metrics.total_batches = num_total_batches;
            state.metrics.samples_seen += samples_trained;

            // For native, sync weights and evaluate periodically
            #[cfg(not(target_arch = "wasm32"))]
            if state.current_batch_idx % 50 == 0 {
                // Sync GPU weights to CPU for evaluation (blocking)
                gpu_net.sync_weights_to_cpu();
                for (cpu_layer, gpu_layer) in state.network.layers.iter_mut().zip(gpu_net.network().layers.iter()) {
                    cpu_layer.weights.copy_from_slice(&gpu_layer.weights);
                    cpu_layer.biases.copy_from_slice(&gpu_layer.biases);
                }

                // Evaluate on validation set (never trained on)
                let val_size = 500.min(state.validation_digits.len());
                let eval_inputs: Vec<Vec<f32>> = state.validation_digits[..val_size]
                    .iter()
                    .map(|d| d.pixels().iter().map(|&p| p as f32 / 255.0).collect())
                    .collect();
                let eval_labels: Vec<u8> = state.validation_digits[..val_size]
                    .iter()
                    .map(|d| d.label())
                    .collect();
                let (loss, acc) = state.network.evaluate(&eval_inputs, &eval_labels);
                state.metrics.accuracy = acc;
                state.metrics.loss = loss;
                state.metrics.loss_history.push(loss);
            }

            // For WASM, start weight sync periodically (completion is handled at frame start)
            #[cfg(target_arch = "wasm32")]
            {
                // Start a new sync periodically (every 50 batches) if not already syncing
                if state.current_batch_idx % 50 == 0 && state.current_batch_idx > 0 {
                    if gpu_net.weight_sync_state() == WeightSyncState::Idle {
                        gpu_net.start_weight_sync();
                    }
                }
            }

            state.current_batch_idx += 1;
        }
    });
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
            "width: {}px; background: {}; padding: 10px; box-sizing: border-box; \
             display: flex; flex-direction: column; gap: 6px; border-right: 1px solid #1a1a2e; \
             overflow-y: auto;",
            INPUT_PANEL_WIDTH as u32, PANEL_BG
        ),
    )?;

    // Title
    let title = document.create_element("div")?;
    title.set_attribute(
        "style",
        &format!("color: {}; font-size: 16px; font-weight: 600;", ACCENT_COLOR),
    )?;
    title.set_text_content(Some("MNIST Trainer"));
    left_panel.append_child(&title)?;

    // Training Controls Section
    let controls_section = document.create_element("div")?;
    controls_section.set_attribute("style", "display: flex; flex-direction: column; gap: 6px;")?;

    // Start/Stop Button
    let train_btn = document.create_element("button")?;
    train_btn.set_id("train-btn");
    train_btn.set_text_content(Some("Start Training"));
    train_btn.set_attribute(
        "style",
        &format!(
            "width: 100%; padding: 8px; font-size: 12px; cursor: pointer; \
             background: {}; color: {}; border: none; border-radius: 6px; \
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
            "width: 100%; padding: 8px; font-size: 12px; cursor: pointer; \
             background: transparent; color: {}; border: 1px solid {}; border-radius: 6px; \
             transition: all 0.2s;",
            ACCENT_SECONDARY, ACCENT_SECONDARY
        ),
    )?;
    controls_section.append_child(&reset_btn)?;

    // GPU/CPU toggle buttons container
    let accel_container = document.create_element("div")?;
    accel_container.set_attribute(
        "style",
        "display: flex; gap: 4px;",
    )?;

    // GPU button (on left, default when available)
    let gpu_btn = document.create_element("button")?;
    gpu_btn.set_id("gpu-btn");
    gpu_btn.set_text_content(Some("GPU"));
    gpu_btn.set_attribute(
        "style",
        &format!(
            "flex: 1; padding: 6px 4px; font-size: 11px; cursor: pointer; \
             background: #1a2a3a; color: {}; border: 1px solid {}; border-radius: 4px; \
             font-weight: 500; transition: all 0.2s;",
            TEXT_COLOR, NEURON_STROKE
        ),
    )?;
    accel_container.append_child(&gpu_btn)?;

    // CPU button (on right)
    let cpu_btn = document.create_element("button")?;
    cpu_btn.set_id("cpu-btn");
    cpu_btn.set_text_content(Some("CPU"));
    cpu_btn.set_attribute(
        "style",
        &format!(
            "flex: 1; padding: 6px 4px; font-size: 11px; cursor: pointer; \
             background: #1a2a3a; color: {}; border: 1px solid {}; border-radius: 4px; \
             font-weight: 500; transition: all 0.2s;",
            TEXT_COLOR, NEURON_STROKE
        ),
    )?;
    accel_container.append_child(&cpu_btn)?;

    controls_section.append_child(&accel_container)?;

    // GPU Status indicator
    let gpu_status = document.create_element("div")?;
    gpu_status.set_id("gpu-status");
    gpu_status.set_attribute(
        "style",
        &format!(
            "color: {}; font-size: 10px; text-align: center; padding: 2px;",
            TEXT_DIM
        ),
    )?;
    controls_section.append_child(&gpu_status)?;

    left_panel.append_child(&controls_section)?;

    // Metrics Section
    let metrics_section = document.create_element("div")?;
    metrics_section.set_id("metrics-section");
    metrics_section.set_attribute(
        "style",
        &format!(
            "background: {}; border-radius: 6px; padding: 8px; \
             border: 1px solid #1a1a2e;",
            BG_COLOR
        ),
    )?;
    metrics_section.set_inner_html(&format!(
        "<div style='color: {}; font-size: 11px; margin-bottom: 4px; font-weight: 600;'>TRAINING METRICS</div>\
         <div id='metric-epoch' style='color: {}; font-size: 12px; margin-bottom: 2px;'>Epoch: 0</div>\
         <div id='metric-batch' style='color: {}; font-size: 12px; margin-bottom: 2px;'>Batch: 0/0</div>\
         <div id='metric-loss' style='color: {}; font-size: 12px; margin-bottom: 2px;'>Loss: -</div>\
         <div id='metric-accuracy' style='color: {}; font-size: 12px; margin-bottom: 2px;'>Accuracy: -</div>\
         <div id='metric-flops' style='color: {}; font-size: 12px;'>FLOPS: -</div>",
        TEXT_DIM, TEXT_COLOR, TEXT_COLOR, TEXT_COLOR, TEXT_COLOR, TEXT_COLOR
    ));
    left_panel.append_child(&metrics_section)?;

    // Loss Chart
    let chart_canvas = document.create_element("canvas")?.dyn_into::<HtmlCanvasElement>()?;
    chart_canvas.set_id("loss-chart");
    chart_canvas.set_width(160);
    chart_canvas.set_height(70);
    chart_canvas.set_attribute(
        "style",
        &format!("background: {}; border-radius: 8px; border: 1px solid #1a1a2e;", BG_COLOR),
    )?;
    left_panel.append_child(&chart_canvas)?;

    // Scale buttons container
    let scale_container = document.create_element("div")?;
    scale_container.set_attribute(
        "style",
        "display: flex; gap: 4px; margin-top: 4px;",
    )?;

    // Linear scale button
    let linear_btn = document.create_element("button")?;
    linear_btn.set_id("scale-linear-btn");
    linear_btn.set_text_content(Some("Linear"));
    linear_btn.set_attribute(
        "style",
        &format!(
            "flex: 1; padding: 6px 4px; font-size: 11px; cursor: pointer; \
             background: {}; color: {}; border: none; border-radius: 4px; \
             font-weight: 600; transition: all 0.2s;",
            ACCENT_COLOR, BG_COLOR
        ),
    )?;
    scale_container.append_child(&linear_btn)?;

    // Log scale button
    let log_btn = document.create_element("button")?;
    log_btn.set_id("scale-log-btn");
    log_btn.set_text_content(Some("Log"));
    log_btn.set_attribute(
        "style",
        &format!(
            "flex: 1; padding: 6px 4px; font-size: 11px; cursor: pointer; \
             background: #1a2a3a; color: {}; border: 1px solid {}; border-radius: 4px; \
             font-weight: 500; transition: all 0.2s;",
            TEXT_COLOR, NEURON_STROKE
        ),
    )?;
    scale_container.append_child(&log_btn)?;

    // Log-Log scale button
    let loglog_btn = document.create_element("button")?;
    loglog_btn.set_id("scale-loglog-btn");
    loglog_btn.set_text_content(Some("Log-Log"));
    loglog_btn.set_attribute(
        "style",
        &format!(
            "flex: 1; padding: 6px 4px; font-size: 11px; cursor: pointer; \
             background: #1a2a3a; color: {}; border: 1px solid {}; border-radius: 4px; \
             font-weight: 500; transition: all 0.2s;",
            TEXT_COLOR, NEURON_STROKE
        ),
    )?;
    scale_container.append_child(&loglog_btn)?;

    left_panel.append_child(&scale_container)?;

    // Divider
    let divider = document.create_element("div")?;
    divider.set_attribute("style", "height: 1px; background: #1a1a2e; margin: 4px 0;")?;
    left_panel.append_child(&divider)?;

    // Sample Section
    let sample_label = document.create_element("div")?;
    sample_label.set_attribute(
        "style",
        &format!("color: {}; font-size: 11px; font-weight: 600;", TEXT_DIM),
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
            "width: 100%; padding: 8px; font-size: 12px; cursor: pointer; \
             background: #1a2a3a; color: {}; border: 1px solid {}; border-radius: 6px; \
             transition: all 0.2s;",
            TEXT_COLOR, NEURON_STROKE
        ),
    )?;
    left_panel.append_child(&load_btn)?;

    // Draw Your Own Button
    let draw_btn = document.create_element("button")?;
    draw_btn.set_id("draw-btn");
    draw_btn.set_text_content(Some("Draw Your Own"));
    draw_btn.set_attribute(
        "style",
        &format!(
            "width: 100%; padding: 8px; font-size: 12px; cursor: pointer; \
             background: #1a2a3a; color: {}; border: 1px solid {}; border-radius: 6px; \
             transition: all 0.2s;",
            TEXT_COLOR, NEURON_STROKE
        ),
    )?;
    left_panel.append_child(&draw_btn)?;

    // Drawing controls container (hidden by default)
    let draw_controls = document.create_element("div")?;
    draw_controls.set_id("draw-controls");
    draw_controls.set_attribute(
        "style",
        "display: none; flex-direction: row; gap: 4px;",
    )?;

    // Clear button
    let clear_btn = document.create_element("button")?;
    clear_btn.set_id("clear-btn");
    clear_btn.set_text_content(Some("Clear"));
    clear_btn.set_attribute(
        "style",
        &format!(
            "flex: 1; padding: 6px; font-size: 11px; cursor: pointer; \
             background: transparent; color: {}; border: 1px solid {}; border-radius: 4px;",
            TEXT_DIM, NEURON_STROKE
        ),
    )?;
    draw_controls.append_child(&clear_btn)?;

    // Done button (exits drawing mode)
    let done_btn = document.create_element("button")?;
    done_btn.set_id("done-btn");
    done_btn.set_text_content(Some("Done"));
    done_btn.set_attribute(
        "style",
        &format!(
            "flex: 1; padding: 6px; font-size: 11px; cursor: pointer; \
             background: {}; color: {}; border: none; border-radius: 4px;",
            ACCENT_COLOR, BG_COLOR
        ),
    )?;
    draw_controls.append_child(&done_btn)?;

    left_panel.append_child(&draw_controls)?;

    // Drawing mode label
    let draw_label = document.create_element("div")?;
    draw_label.set_id("draw-label");
    draw_label.set_attribute(
        "style",
        &format!(
            "display: none; color: {}; font-size: 10px; text-align: center;",
            ACCENT_SECONDARY
        ),
    )?;
    draw_label.set_text_content(Some("Draw a digit (0-9)"));
    left_panel.append_child(&draw_label)?;

    // Digit display
    let digit_canvas = document.create_element("canvas")?.dyn_into::<HtmlCanvasElement>()?;
    digit_canvas.set_id("digit-canvas");
    digit_canvas.set_width(120);
    digit_canvas.set_height(120);
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
            "color: {}; text-align: center; font-size: 12px; padding: 6px; \
             background: {}; border-radius: 6px;",
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

    // Training Complete Modal
    let modal_overlay = document.create_element("div")?;
    modal_overlay.set_id("completion-modal");
    modal_overlay.set_attribute(
        "style",
        "position: fixed; \
         top: 0; left: 0; right: 0; bottom: 0; \
         background: rgba(0, 0, 0, 0.7); \
         display: none; \
         justify-content: center; \
         align-items: center; \
         z-index: 2000;",
    )?;

    let modal_content = document.create_element("div")?;
    modal_content.set_attribute(
        "style",
        &format!(
            "background: {}; \
             border-radius: 16px; \
             padding: 32px 40px; \
             text-align: center; \
             border: 2px solid {}; \
             box-shadow: 0 20px 60px rgba(0,0,0,0.5); \
             animation: modalPop 0.3s ease-out;",
            PANEL_BG, ACCENT_COLOR
        ),
    )?;

    // Add CSS animation
    let style = document.create_element("style")?;
    style.set_text_content(Some(
        "@keyframes modalPop { \
            0% { transform: scale(0.8); opacity: 0; } \
            100% { transform: scale(1); opacity: 1; } \
        } \
        @keyframes confetti { \
            0% { transform: translateY(0) rotate(0deg); opacity: 1; } \
            100% { transform: translateY(100px) rotate(720deg); opacity: 0; } \
        }"
    ));
    body.append_child(&style)?;

    let modal_title = document.create_element("div")?;
    modal_title.set_id("modal-title");
    modal_title.set_attribute(
        "style",
        &format!(
            "color: {}; font-size: 28px; font-weight: 700; margin-bottom: 8px;",
            ACCENT_COLOR
        ),
    )?;
    modal_title.set_text_content(Some("🎉 Training Complete!"));
    modal_content.append_child(&modal_title)?;

    let modal_accuracy = document.create_element("div")?;
    modal_accuracy.set_id("modal-accuracy");
    modal_accuracy.set_attribute(
        "style",
        &format!(
            "color: {}; font-size: 48px; font-weight: 800; margin: 16px 0;",
            TEXT_COLOR
        ),
    )?;
    modal_accuracy.set_text_content(Some("99.0%"));
    modal_content.append_child(&modal_accuracy)?;

    let modal_subtitle = document.create_element("div")?;
    modal_subtitle.set_attribute(
        "style",
        &format!(
            "color: {}; font-size: 14px; margin-bottom: 24px;",
            TEXT_DIM
        ),
    )?;
    modal_subtitle.set_text_content(Some("accuracy achieved"));
    modal_content.append_child(&modal_subtitle)?;

    let button_container = document.create_element("div")?;
    button_container.set_attribute("style", "display: flex; gap: 12px; justify-content: center;")?;

    let ok_btn = document.create_element("button")?;
    ok_btn.set_id("modal-ok-btn");
    ok_btn.set_text_content(Some("Ok"));
    ok_btn.set_attribute(
        "style",
        &format!(
            "padding: 12px 32px; font-size: 14px; cursor: pointer; \
             background: {}; color: {}; border: none; border-radius: 8px; \
             font-weight: 600; transition: all 0.2s;",
            ACCENT_COLOR, BG_COLOR
        ),
    )?;
    button_container.append_child(&ok_btn)?;

    let more_btn = document.create_element("button")?;
    more_btn.set_id("modal-more-btn");
    more_btn.set_text_content(Some("One...more...epoch"));
    more_btn.set_attribute(
        "style",
        &format!(
            "padding: 12px 24px; font-size: 14px; cursor: pointer; \
             background: transparent; color: {}; border: 1px solid {}; border-radius: 8px; \
             font-weight: 600; transition: all 0.2s;",
            ACCENT_SECONDARY, ACCENT_SECONDARY
        ),
    )?;
    button_container.append_child(&more_btn)?;

    modal_content.append_child(&button_container)?;
    modal_overlay.append_child(&modal_content)?;
    body.append_child(&modal_overlay)?;

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
                // Exit drawing mode if active
                s.drawing_mode = false;
                s.is_drawing = false;
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

    // Draw button - enter drawing mode
    let doc_clone = document.clone();
    let draw_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.drawing_mode = true;
                s.current_digit_idx = None; // Clear current sample
                s.drawn_pixels = [0u8; 784];
                s.current_activations.clear();
                s.current_prediction = None;
            }
        });
        // Show drawing controls, hide draw button
        if let Some(controls) = doc_clone.get_element_by_id("draw-controls") {
            let _ = controls.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "flex");
        }
        if let Some(label) = doc_clone.get_element_by_id("draw-label") {
            let _ = label.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "block");
        }
        if let Some(btn) = doc_clone.get_element_by_id("draw-btn") {
            let _ = btn.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "none");
        }
        if let Some(btn) = doc_clone.get_element_by_id("load-btn") {
            let _ = btn.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "none");
        }
        // Update canvas cursor
        if let Some(canvas) = doc_clone.get_element_by_id("digit-canvas") {
            let _ = canvas.dyn_into::<HtmlElement>().unwrap().style().set_property("cursor", "crosshair");
        }
        let _ = render_digit(&doc_clone);
        let _ = render(&doc_clone);
        let _ = update_prediction(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("draw-btn")
        .ok_or("no draw button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(draw_closure.as_ref().unchecked_ref()));
    draw_closure.forget();

    // Clear button - clear drawing
    let doc_clone = document.clone();
    let clear_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.clear_drawing();
            }
        });
        let _ = render_digit(&doc_clone);
        let _ = render(&doc_clone);
        let _ = update_prediction(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("clear-btn")
        .ok_or("no clear button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(clear_closure.as_ref().unchecked_ref()));
    clear_closure.forget();

    // Done button - exit drawing mode
    let doc_clone = document.clone();
    let done_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.drawing_mode = false;
                s.is_drawing = false;
            }
        });
        // Hide drawing controls, show draw button
        if let Some(controls) = doc_clone.get_element_by_id("draw-controls") {
            let _ = controls.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "none");
        }
        if let Some(label) = doc_clone.get_element_by_id("draw-label") {
            let _ = label.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "none");
        }
        if let Some(btn) = doc_clone.get_element_by_id("draw-btn") {
            let _ = btn.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "block");
        }
        if let Some(btn) = doc_clone.get_element_by_id("load-btn") {
            let _ = btn.dyn_into::<HtmlElement>().unwrap().style().set_property("display", "block");
        }
        // Restore canvas cursor
        if let Some(canvas) = doc_clone.get_element_by_id("digit-canvas") {
            let _ = canvas.dyn_into::<HtmlElement>().unwrap().style().set_property("cursor", "default");
        }
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("done-btn")
        .ok_or("no done button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(done_closure.as_ref().unchecked_ref()));
    done_closure.forget();

    // Digit canvas drawing events
    let digit_canvas = document
        .get_element_by_id("digit-canvas")
        .ok_or("no digit canvas")?
        .dyn_into::<HtmlCanvasElement>()?;

    // Mouse down on digit canvas - start drawing
    let doc_clone = document.clone();
    let digit_mousedown = Closure::wrap(Box::new(move |event: MouseEvent| {
        let canvas_size = 120.0;
        let x = event.offset_x() as f64;
        let y = event.offset_y() as f64;

        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                if s.drawing_mode {
                    s.is_drawing = true;
                    s.last_draw_pos = Some((x, y));
                    s.draw_at(x, y, canvas_size);
                    s.classify_drawn();
                }
            }
        });
        let _ = render_digit(&doc_clone);
        let _ = render(&doc_clone);
        let _ = update_prediction(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    digit_canvas.add_event_listener_with_callback("mousedown", digit_mousedown.as_ref().unchecked_ref())?;
    digit_mousedown.forget();

    // Mouse move on digit canvas - continue drawing
    let doc_clone = document.clone();
    let digit_mousemove = Closure::wrap(Box::new(move |event: MouseEvent| {
        let canvas_size = 120.0;
        let x = event.offset_x() as f64;
        let y = event.offset_y() as f64;

        let should_render = STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                if s.drawing_mode && s.is_drawing {
                    if let Some((last_x, last_y)) = s.last_draw_pos {
                        s.draw_line(last_x, last_y, x, y, canvas_size);
                    } else {
                        s.draw_at(x, y, canvas_size);
                    }
                    s.last_draw_pos = Some((x, y));
                    s.classify_drawn();
                    return true;
                }
            }
            false
        });

        if should_render {
            let _ = render_digit(&doc_clone);
            let _ = render(&doc_clone);
            let _ = update_prediction(&doc_clone);
        }
    }) as Box<dyn Fn(MouseEvent)>);
    digit_canvas.add_event_listener_with_callback("mousemove", digit_mousemove.as_ref().unchecked_ref())?;
    digit_mousemove.forget();

    // Mouse up on digit canvas - stop drawing
    let digit_mouseup = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.is_drawing = false;
                s.last_draw_pos = None;
            }
        });
    }) as Box<dyn Fn(MouseEvent)>);
    digit_canvas.add_event_listener_with_callback("mouseup", digit_mouseup.as_ref().unchecked_ref())?;
    digit_mouseup.forget();

    // Mouse leave on digit canvas - stop drawing
    let digit_mouseleave = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.is_drawing = false;
                s.last_draw_pos = None;
            }
        });
    }) as Box<dyn Fn(MouseEvent)>);
    digit_canvas.add_event_listener_with_callback("mouseleave", digit_mouseleave.as_ref().unchecked_ref())?;
    digit_mouseleave.forget();

    // Train button
    let doc_clone = document.clone();
    let train_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        let should_start = STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                match s.training_state {
                    TrainingState::Idle | TrainingState::Paused => {
                        s.training_state = TrainingState::Training;
                        // Start timing if this is the first start (Idle)
                        if s.metrics.training_start_time.is_none() {
                            s.metrics.training_start_time = Some(js_sys::Date::now());
                        }
                        true
                    }
                    TrainingState::Training => {
                        s.training_state = TrainingState::Paused;
                        false
                    }
                    TrainingState::Completed => {
                        // Do nothing when completed - use modal buttons or reset
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
        // Get GPU availability before reset
        let (gpu_available, accel_mode) = STATE.with(|state| {
            if let Some(ref s) = *state.borrow() {
                (s.gpu_available, s.accelerator_mode)
            } else {
                (false, AcceleratorMode::Cpu)
            }
        });

        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.reset_network();
            }
        });
        // Clear GPU network on reset
        GPU_NETWORK.with(|gpu| {
            *gpu.borrow_mut() = None;
        });
        let _ = hide_completion_modal(&doc_clone);
        let _ = update_train_button(&doc_clone);
        let _ = update_gpu_button(&doc_clone);
        let _ = update_metrics(&doc_clone);
        let _ = render_loss_chart(&doc_clone);
        let _ = render(&doc_clone);

        // Re-initialize GPU if it was available and in GPU mode
        if gpu_available && accel_mode == AcceleratorMode::Gpu {
            init_gpu_async(doc_clone.clone());
        }
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("reset-btn")
        .ok_or("no reset button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(reset_closure.as_ref().unchecked_ref()));
    reset_closure.forget();

    // Linear scale button
    let doc_clone = document.clone();
    let linear_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.metrics.chart_scale = ChartScale::Linear;
            }
        });
        let _ = update_scale_buttons(&doc_clone);
        let _ = render_loss_chart(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("scale-linear-btn")
        .ok_or("no linear scale button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(linear_closure.as_ref().unchecked_ref()));
    linear_closure.forget();

    // Log scale button
    let doc_clone = document.clone();
    let log_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.metrics.chart_scale = ChartScale::Log;
            }
        });
        let _ = update_scale_buttons(&doc_clone);
        let _ = render_loss_chart(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("scale-log-btn")
        .ok_or("no log scale button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(log_closure.as_ref().unchecked_ref()));
    log_closure.forget();

    // Log-Log scale button
    let doc_clone = document.clone();
    let loglog_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.metrics.chart_scale = ChartScale::LogLog;
            }
        });
        let _ = update_scale_buttons(&doc_clone);
        let _ = render_loss_chart(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("scale-loglog-btn")
        .ok_or("no log-log scale button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(loglog_closure.as_ref().unchecked_ref()));
    loglog_closure.forget();

    // GPU button click handler
    let doc_clone = document.clone();
    let gpu_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        let (is_training, gpu_available, gpu_init_state) = STATE.with(|state| {
            if let Some(ref s) = *state.borrow() {
                (s.training_state == TrainingState::Training, s.gpu_available, s.gpu_init_state)
            } else {
                (false, false, GpuInitState::NotStarted)
            }
        });

        // Don't allow switching while training
        if is_training {
            return;
        }

        // Don't allow selecting GPU if not available or init failed
        if !gpu_available || gpu_init_state == GpuInitState::Failed {
            return;
        }

        // Only allow GPU if ready (init happens automatically on startup)
        if gpu_init_state == GpuInitState::Ready {
            STATE.with(|state| {
                if let Some(ref mut s) = *state.borrow_mut() {
                    s.accelerator_mode = AcceleratorMode::Gpu;
                }
            });
        }

        let _ = update_gpu_button(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("gpu-btn")
        .ok_or("no gpu button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(gpu_closure.as_ref().unchecked_ref()));
    gpu_closure.forget();

    // CPU button click handler
    let doc_clone = document.clone();
    let cpu_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        let is_training = STATE.with(|state| {
            if let Some(ref s) = *state.borrow() {
                s.training_state == TrainingState::Training
            } else {
                false
            }
        });

        // Don't allow switching while training
        if is_training {
            return;
        }

        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.accelerator_mode = AcceleratorMode::Cpu;
            }
        });

        let _ = update_gpu_button(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("cpu-btn")
        .ok_or("no cpu button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(cpu_closure.as_ref().unchecked_ref()));
    cpu_closure.forget();

    // Initialize GPU button state
    update_gpu_button(document)?;

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

        // Convert canvas coords (120x120) to pixel index (28x28)
        let scale = 120.0 / 28.0;
        let col = (x / scale).floor() as usize;
        let row = (y / scale).floor() as usize;

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

    // Modal "Ok" button - close modal
    let doc_clone = document.clone();
    let modal_ok_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        let _ = hide_completion_modal(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("modal-ok-btn")
        .ok_or("no modal ok button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(modal_ok_closure.as_ref().unchecked_ref()));
    modal_ok_closure.forget();

    // Modal "One more epoch" button - close modal and continue training
    let doc_clone = document.clone();
    let modal_more_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        let _ = hide_completion_modal(&doc_clone);

        // Set state back to training and continue for one more epoch
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.training_state = TrainingState::Training;
            }
        });

        let _ = update_train_button(&doc_clone);
        start_training_loop(doc_clone.clone());
    }) as Box<dyn Fn(MouseEvent)>);
    document
        .get_element_by_id("modal-more-btn")
        .ok_or("no modal more button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(modal_more_closure.as_ref().unchecked_ref()));
    modal_more_closure.forget();

    Ok(())
}

/// Poll for async weight sync completion at the start of each frame
/// This is called before training new batches to check if previous sync completed
#[cfg(target_arch = "wasm32")]
fn poll_gpu_weight_sync(state: &mut State) {
    GPU_NETWORK.with(|gpu| {
        let mut gpu_borrow = gpu.borrow_mut();
        if let Some(gpu_net) = gpu_borrow.as_mut() {
            // Only poll if we're actually using GPU mode
            if state.accelerator_mode != AcceleratorMode::Gpu || state.gpu_init_state != GpuInitState::Ready {
                return;
            }

            let sync_state = gpu_net.poll_weight_sync();

            if sync_state == WeightSyncState::Complete {

                // Weights synced! Copy to visualization state and evaluate
                for (cpu_layer, gpu_layer) in state.network.layers.iter_mut().zip(gpu_net.network().layers.iter()) {
                    cpu_layer.weights.copy_from_slice(&gpu_layer.weights);
                    cpu_layer.biases.copy_from_slice(&gpu_layer.biases);
                }

                // Evaluate with real weights on validation set (never trained on)
                let val_size = 500.min(state.validation_digits.len());
                let eval_inputs: Vec<Vec<f32>> = state.validation_digits[..val_size]
                    .iter()
                    .map(|d| d.pixels().iter().map(|&p| p as f32 / 255.0).collect())
                    .collect();
                let eval_labels: Vec<u8> = state.validation_digits[..val_size]
                    .iter()
                    .map(|d| d.label())
                    .collect();
                let (loss, acc) = state.network.evaluate(&eval_inputs, &eval_labels);
                state.metrics.accuracy = acc;
                state.metrics.loss = loss;
                state.metrics.loss_history.push(loss);

                // Reset sync state for next evaluation
                gpu_net.reset_weight_sync();
            }
        }
    });
}

fn start_training_loop(document: Document) {
    let window = web_sys::window().unwrap();

    let closure = Closure::wrap(Box::new(move || {
        let (is_training, just_completed, final_accuracy) = STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                // Poll for async weight sync completion at the START of each frame
                // This is when the async callbacks will have fired from the previous frame
                #[cfg(target_arch = "wasm32")]
                poll_gpu_weight_sync(s);

                if s.training_state == TrainingState::Training {
                    // Check if we should use GPU training
                    let use_gpu = s.accelerator_mode == AcceleratorMode::Gpu
                        && s.gpu_init_state == GpuInitState::Ready;

                    // In WASM with GPU: don't train while weight sync is in progress
                    // Training commands would queue up and delay the sync completion
                    #[cfg(target_arch = "wasm32")]
                    let skip_training = {
                        if use_gpu {
                            GPU_NETWORK.with(|gpu| {
                                gpu.borrow().as_ref().map(|g| g.is_syncing_weights()).unwrap_or(false)
                            })
                        } else {
                            false
                        }
                    };
                    #[cfg(not(target_arch = "wasm32"))]
                    let skip_training = false;

                    if skip_training {
                        // Skip training this frame to let weight sync complete
                        (true, false, 0.0)
                    } else if use_gpu {
                        train_batches_gpu(s, BATCHES_PER_FRAME);
                        // Check if training just completed
                        let completed = s.training_state == TrainingState::Completed;
                        (true, completed, s.metrics.accuracy)
                    } else {
                        s.train_batches(BATCHES_PER_FRAME);
                        // Check if training just completed
                        let completed = s.training_state == TrainingState::Completed;
                        (true, completed, s.metrics.accuracy)
                    }
                } else {
                    (false, false, 0.0)
                }
            } else {
                (false, false, 0.0)
            }
        });

        // Update UI
        let _ = update_metrics(&document);
        let _ = render_loss_chart(&document);
        let _ = render(&document);

        if just_completed {
            // Training just finished - show the completion modal
            let _ = update_train_button(&document);
            let _ = show_completion_modal(&document, final_accuracy);
        } else if is_training {
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
                TrainingState::Completed => ("Training Complete!", "#4a4a5a"),
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

fn show_completion_modal(document: &Document, accuracy: f32) -> Result<(), JsValue> {
    let modal = document.get_element_by_id("completion-modal").ok_or("no modal")?;
    let accuracy_el = document.get_element_by_id("modal-accuracy").ok_or("no modal accuracy")?;

    // Update accuracy display
    accuracy_el.set_text_content(Some(&format!("{:.1}%", accuracy * 100.0)));

    // Show modal with flex display
    let _ = modal.dyn_ref::<HtmlElement>().unwrap().style().set_property("display", "flex");

    Ok(())
}

fn hide_completion_modal(document: &Document) -> Result<(), JsValue> {
    let modal = document.get_element_by_id("completion-modal").ok_or("no modal")?;
    let _ = modal.dyn_ref::<HtmlElement>().unwrap().style().set_property("display", "none");
    Ok(())
}

fn update_scale_buttons(document: &Document) -> Result<(), JsValue> {
    let linear_btn = document.get_element_by_id("scale-linear-btn").ok_or("no linear btn")?;
    let log_btn = document.get_element_by_id("scale-log-btn").ok_or("no log btn")?;
    let loglog_btn = document.get_element_by_id("scale-loglog-btn").ok_or("no loglog btn")?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            let active_style = format!(
                "flex: 1; padding: 6px 4px; font-size: 11px; cursor: pointer; \
                 background: {}; color: {}; border: none; border-radius: 4px; \
                 font-weight: 600; transition: all 0.2s;",
                ACCENT_COLOR, BG_COLOR
            );
            let inactive_style = format!(
                "flex: 1; padding: 6px 4px; font-size: 11px; cursor: pointer; \
                 background: #1a2a3a; color: {}; border: 1px solid {}; border-radius: 4px; \
                 font-weight: 500; transition: all 0.2s;",
                TEXT_COLOR, NEURON_STROKE
            );

            let (linear_style, log_style, loglog_style) = match s.metrics.chart_scale {
                ChartScale::Linear => (&active_style, &inactive_style, &inactive_style),
                ChartScale::Log => (&inactive_style, &active_style, &inactive_style),
                ChartScale::LogLog => (&inactive_style, &inactive_style, &active_style),
            };

            let _ = linear_btn.set_attribute("style", linear_style);
            let _ = log_btn.set_attribute("style", log_style);
            let _ = loglog_btn.set_attribute("style", loglog_style);
        }
    });

    Ok(())
}

fn update_gpu_button(document: &Document) -> Result<(), JsValue> {
    let gpu_btn = document.get_element_by_id("gpu-btn").ok_or("no gpu button")?;
    let cpu_btn = document.get_element_by_id("cpu-btn").ok_or("no cpu button")?;
    let status = document.get_element_by_id("gpu-status").ok_or("no gpu status")?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            let gpu_btn_el = gpu_btn.dyn_ref::<HtmlElement>().unwrap();
            let cpu_btn_el = cpu_btn.dyn_ref::<HtmlElement>().unwrap();

            // Check if GPU is usable (available and not failed)
            let gpu_usable = s.gpu_available && s.gpu_init_state != GpuInitState::Failed;

            // Style buttons based on current mode and GPU availability
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

                    // GPU button - disabled style if not usable
                    if gpu_usable {
                        let _ = gpu_btn_el.style().set_property("background", "#1a2a3a");
                        let _ = gpu_btn_el.style().set_property("border", &format!("1px solid {}", NEURON_STROKE));
                        let _ = gpu_btn_el.style().set_property("color", TEXT_COLOR);
                        let _ = gpu_btn_el.style().set_property("font-weight", "500");
                        let _ = gpu_btn_el.style().set_property("opacity", "1");
                        let _ = gpu_btn_el.style().set_property("cursor", "pointer");
                    } else {
                        // Disabled style - grayed out
                        let _ = gpu_btn_el.style().set_property("background", "#1a1a2a");
                        let _ = gpu_btn_el.style().set_property("border", "1px solid #2a2a3a");
                        let _ = gpu_btn_el.style().set_property("color", "#4a4a5a");
                        let _ = gpu_btn_el.style().set_property("font-weight", "500");
                        let _ = gpu_btn_el.style().set_property("opacity", "0.5");
                        let _ = gpu_btn_el.style().set_property("cursor", "not-allowed");
                    }
                }
            };

            // Update status text based on GPU state
            let (status_text, status_color) = if !s.gpu_available {
                ("WebGPU unavailable".to_string(), ACCENT_SECONDARY)
            } else {
                match s.gpu_init_state {
                    GpuInitState::NotStarted => ("Initializing GPU...".to_string(), TEXT_DIM),
                    GpuInitState::Initializing => ("Initializing GPU...".to_string(), "#f0ad4e"),
                    GpuInitState::Ready => {
                        if s.accelerator_mode == AcceleratorMode::Gpu {
                            ("WebGPU active".to_string(), ACCENT_COLOR)
                        } else {
                            ("WebGPU ready".to_string(), TEXT_DIM)
                        }
                    }
                    GpuInitState::Failed => ("GPU failed, using CPU".to_string(), ACCENT_SECONDARY),
                }
            };
            status.set_text_content(Some(&status_text));
            let _ = status.dyn_ref::<HtmlElement>().unwrap().style().set_property("color", status_color);
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
            if let Some(el) = document.get_element_by_id("metric-flops") {
                // Calculate FLOPS: (samples_seen * flops_per_sample) / elapsed_seconds
                if let Some(start_time) = s.metrics.training_start_time {
                    let elapsed_ms = js_sys::Date::now() - start_time;
                    if elapsed_ms > 0.0 && s.metrics.samples_seen > 0 {
                        let total_flops = s.metrics.samples_seen as f64 * s.metrics.flops_per_sample as f64;
                        let flops_per_sec = total_flops / (elapsed_ms / 1000.0);

                        // Format with appropriate suffix
                        let (value, suffix) = if flops_per_sec >= 1e12 {
                            (flops_per_sec / 1e12, "TFLOPS")
                        } else if flops_per_sec >= 1e9 {
                            (flops_per_sec / 1e9, "GFLOPS")
                        } else if flops_per_sec >= 1e6 {
                            (flops_per_sec / 1e6, "MFLOPS")
                        } else if flops_per_sec >= 1e3 {
                            (flops_per_sec / 1e3, "KFLOPS")
                        } else {
                            (flops_per_sec, "FLOPS")
                        };

                        el.set_text_content(Some(&format!("{:.2} {}", value, suffix)));
                    } else {
                        el.set_text_content(Some("FLOPS: -"));
                    }
                } else {
                    el.set_text_content(Some("FLOPS: -"));
                }
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

            let chart_scale = s.metrics.chart_scale;
            let log_y = matches!(chart_scale, ChartScale::Log | ChartScale::LogLog);
            let log_x = matches!(chart_scale, ChartScale::LogLog);

            // Transform Y values if log scale
            let y_values: Vec<f64> = if log_y {
                history.iter().map(|&v| (v.max(1e-7) as f64).ln()).collect()
            } else {
                history.iter().map(|&v| v as f64).collect()
            };

            // Transform X values if log-log scale
            let x_values: Vec<f64> = if log_x {
                (0..history.len()).map(|i| ((i + 1) as f64).ln()).collect()
            } else {
                (0..history.len()).map(|i| i as f64).collect()
            };

            // Find min/max for scaling
            let max_y = y_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_y = y_values.iter().cloned().fold(f64::INFINITY, f64::min);
            let range_y = (max_y - min_y).max(0.1);

            let max_x = x_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_x = x_values.iter().cloned().fold(f64::INFINITY, f64::min);
            let range_x = (max_x - min_x).max(0.1);

            // Draw line
            ctx.set_stroke_style_str(ACCENT_COLOR);
            ctx.set_line_width(1.5);
            ctx.begin_path();

            let padding = 10.0;
            let chart_width = width - 2.0 * padding;
            let chart_height = height - 2.0 * padding;

            for i in 0..y_values.len() {
                let x = padding + ((x_values[i] - min_x) / range_x) * chart_width;
                let y = padding + (1.0 - (y_values[i] - min_y) / range_y) * chart_height;

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

            let max_display = if log_y { max_y.exp() } else { max_y };
            let min_display = if log_y { min_y.exp() } else { min_y };

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
                let actual = s.validation_digits[digit_idx].label();
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

                // Get pixel value from validation digit
                let pixel_val = s.current_digit_idx
                    .and_then(|idx| s.validation_digits.get(idx))
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
            // Clear canvas
            let canvas_size = 120.0;
            let scale = canvas_size / 28.0;
            ctx.set_fill_style_str(BG_COLOR);
            ctx.fill_rect(0.0, 0.0, canvas_size, canvas_size);

            // Get pixels to render - either from drawing or from loaded validation digit
            let pixels: Option<&[u8]> = if s.drawing_mode {
                Some(&s.drawn_pixels)
            } else if let Some(digit_idx) = s.current_digit_idx {
                Some(s.validation_digits[digit_idx].pixels())
            } else {
                None
            };

            if let Some(pixels) = pixels {
                // Draw digit scaled to fit canvas
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
                            ctx.fill_rect(col as f64 * scale, row as f64 * scale, scale, scale);
                        }

                        // Draw highlight border for hovered pixel
                        if is_hovered {
                            ctx.set_stroke_style_str(ACCENT_COLOR);
                            ctx.set_line_width(2.0);
                            ctx.stroke_rect(col as f64 * scale, row as f64 * scale, scale, scale);
                        }
                    }
                }

                // Draw a subtle grid in drawing mode for guidance
                if s.drawing_mode {
                    ctx.set_stroke_style_str("rgba(61, 90, 128, 0.2)");
                    ctx.set_line_width(0.5);
                    // Vertical lines every scale pixels (1 MNIST pixel)
                    for i in 0..=28 {
                        ctx.begin_path();
                        ctx.move_to(i as f64 * scale, 0.0);
                        ctx.line_to(i as f64 * scale, canvas_size);
                        ctx.stroke();
                    }
                    // Horizontal lines
                    for i in 0..=28 {
                        ctx.begin_path();
                        ctx.move_to(0.0, i as f64 * scale);
                        ctx.line_to(canvas_size, i as f64 * scale);
                        ctx.stroke();
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
                // Output layer: draw as circles with labels (larger for visibility)

                // Draw glow for active neurons
                if activation > 0.1 {
                    let glow_alpha = (activation * 0.5).min(0.5) as f64;
                    let glow_radius = OUTPUT_NEURON_RADIUS * (1.3 + activation as f64 * 0.4);

                    ctx.begin_path();
                    let _ = ctx.arc(pos.x, pos.y, glow_radius, 0.0, std::f64::consts::TAU);
                    ctx.set_fill_style_str(ACCENT_COLOR);
                    ctx.set_global_alpha(glow_alpha);
                    ctx.fill();
                    ctx.set_global_alpha(1.0);
                }

                // Neuron circle (larger for output layer)
                ctx.begin_path();
                let _ = ctx.arc(pos.x, pos.y, OUTPUT_NEURON_RADIUS, 0.0, std::f64::consts::TAU);

                if is_hovered {
                    ctx.set_fill_style_str(NEURON_HOVER);
                    ctx.set_stroke_style_str("#ffffff");
                    ctx.set_line_width(3.0 / state.zoom);
                } else {
                    ctx.set_fill_style_str(&fill_color);
                    ctx.set_stroke_style_str(NEURON_STROKE);
                    ctx.set_line_width(2.0 / state.zoom);
                }

                ctx.fill();
                ctx.stroke();

                // Draw output layer labels (larger font)
                ctx.set_fill_style_str(TEXT_COLOR);
                ctx.set_font(&format!("bold {}px 'Inter', sans-serif", (16.0 / state.zoom.sqrt()).max(12.0)));
                ctx.set_text_align("center");
                ctx.set_text_baseline("middle");
                let _ = ctx.fill_text(&pos.neuron_idx.to_string(), pos.x, pos.y);
            }
        }

        ctx.restore();

        Ok(())
    })
}

/// Returns the default accelerator mode based on GPU availability
fn get_default_accelerator_mode(gpu_available: bool) -> AcceleratorMode {
    if gpu_available {
        AcceleratorMode::Gpu
    } else {
        AcceleratorMode::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_accelerator_gpu_when_available() {
        let mode = get_default_accelerator_mode(true);
        assert_eq!(mode, AcceleratorMode::Gpu, "GPU should be default when available");
    }

    #[test]
    fn test_default_accelerator_cpu_when_gpu_unavailable() {
        let mode = get_default_accelerator_mode(false);
        assert_eq!(mode, AcceleratorMode::Cpu, "CPU should be default when GPU unavailable");
    }
}
