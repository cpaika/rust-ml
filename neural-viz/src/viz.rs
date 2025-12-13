use common::{parse_digits_from_bytes, Digit};
use neural::NeuralNet;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, MouseEvent, WheelEvent};

const INPUT_PANEL_WIDTH: f64 = 150.0;
const NEURON_RADIUS: f64 = 10.0;
const LAYER_SPACING: f64 = 300.0;

// Softer color palette
const BG_COLOR: &str = "#0f0f1a";
const PANEL_BG: &str = "#1a1a2e";
const NEURON_FILL: &str = "#2d4a6e";
const NEURON_STROKE: &str = "#5a9fd4";
const NEURON_HOVER: &str = "#7ec8e3";
const WEIGHT_COLOR: &str = "#3a4a5c";
const WEIGHT_HIGHLIGHT: &str = "#7ec8e3";
const ACCENT_COLOR: &str = "#5a9fd4";
const TEXT_COLOR: &str = "#c8d6e5";

struct NeuronPosition {
    layer_idx: usize,
    neuron_idx: usize,
    x: f64,
    y: f64,
}

struct State {
    network: NeuralNet,
    digits: Vec<Digit>,
    current_digit_idx: Option<usize>,
    neuron_positions: Vec<NeuronPosition>,
    hovered_neuron: Option<(usize, usize)>, // (layer_idx, neuron_idx)
    mouse_x: f64,
    mouse_y: f64,
    // Canvas dimensions
    canvas_width: f64,
    canvas_height: f64,
    // Pan and zoom
    offset_x: f64,
    offset_y: f64,
    zoom: f64,
    is_dragging: bool,
    drag_start_x: f64,
    drag_start_y: f64,
}

impl State {
    fn new(network: NeuralNet, digits: Vec<Digit>, canvas_width: f64, canvas_height: f64) -> Self {
        let neuron_positions = Self::calculate_positions(&network, canvas_width, canvas_height);
        Self {
            network,
            digits,
            current_digit_idx: None,
            neuron_positions,
            hovered_neuron: None,
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
        }
    }

    fn calculate_positions(network: &NeuralNet, canvas_width: f64, canvas_height: f64) -> Vec<NeuronPosition> {
        let mut positions = Vec::new();

        // Collect all layers: input, hidden layers, output
        let all_layers = Self::get_all_layers(network);
        let num_layers = all_layers.len();

        // Find max neurons in any layer for scaling
        let max_neurons = all_layers.iter().map(|l| l.neurons.len()).max().unwrap_or(1);

        // Calculate spacing to fit all neurons with padding
        let available_height = canvas_height - 100.0; // padding top and bottom
        let neuron_spacing = (available_height / max_neurons as f64).min(NEURON_RADIUS * 3.0);

        for (layer_idx, layer) in all_layers.iter().enumerate() {
            let num_neurons = layer.neurons.len();
            let layer_x = INPUT_PANEL_WIDTH + 100.0 + (layer_idx as f64) * LAYER_SPACING;

            // Center neurons vertically
            let total_height = (num_neurons - 1) as f64 * neuron_spacing;
            let start_y = (canvas_height - total_height) / 2.0;

            for neuron_idx in 0..num_neurons {
                let y = start_y + (neuron_idx as f64) * neuron_spacing;
                positions.push(NeuronPosition {
                    layer_idx,
                    neuron_idx,
                    x: layer_x,
                    y,
                });
            }
        }

        positions
    }

    fn get_all_layers(network: &NeuralNet) -> Vec<&neural::Layer> {
        let mut layers = vec![&network.input_layer];
        layers.extend(network.hidden_layers.iter());
        layers.push(&network.output_layer);
        layers
    }

    fn screen_to_world(&self, screen_x: f64, screen_y: f64) -> (f64, f64) {
        let world_x = (screen_x - self.offset_x) / self.zoom;
        let world_y = (screen_y - self.offset_y) / self.zoom;
        (world_x, world_y)
    }

    fn get_neuron_at(&self, screen_x: f64, screen_y: f64) -> Option<(usize, usize)> {
        let (world_x, world_y) = self.screen_to_world(screen_x, screen_y);
        let hit_radius = NEURON_RADIUS / self.zoom.sqrt(); // Adjust hit area with zoom

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

    fn zoom_at(&mut self, screen_x: f64, screen_y: f64, delta: f64) {
        let zoom_factor = if delta > 0.0 { 0.9 } else { 1.1 };
        let new_zoom = (self.zoom * zoom_factor).clamp(0.3, 3.0);

        // Zoom toward mouse position
        let (world_x, world_y) = self.screen_to_world(screen_x, screen_y);

        self.zoom = new_zoom;

        // Adjust offset to keep world point under mouse
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

    fn load_next_digit(&mut self) {
        let next_idx = match self.current_digit_idx {
            Some(idx) => (idx + 1) % self.digits.len(),
            None => 0,
        };
        self.current_digit_idx = Some(next_idx);
    }
}

thread_local! {
    static STATE: std::cell::RefCell<Option<State>> = const { std::cell::RefCell::new(None) };
}

pub fn init(csv_data: &[u8]) -> Result<(), JsValue> {
    let digits =
        parse_digits_from_bytes(csv_data).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let network = NeuralNet::new(784, 3, 28, 10);

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;

    let canvas_width = window.inner_width()?.as_f64().unwrap_or(1200.0) - INPUT_PANEL_WIDTH;
    let canvas_height = window.inner_height()?.as_f64().unwrap_or(800.0);

    STATE.with(|state| {
        *state.borrow_mut() = Some(State::new(network, digits, canvas_width, canvas_height));
    });

    setup_ui(&document, canvas_width as u32, canvas_height as u32)?;
    setup_handlers(&document)?;

    // Load first digit by default
    STATE.with(|state| {
        if let Some(ref mut s) = *state.borrow_mut() {
            s.load_next_digit();
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
        &format!("margin: 0; overflow: hidden; background: {}; font-family: 'Segoe UI', sans-serif;", BG_COLOR),
    )?;

    // Container
    let container = document.create_element("div")?;
    container.set_attribute("style", "display: flex; height: 100vh;")?;

    // Left panel with button and digit display
    let left_panel = document.create_element("div")?;
    left_panel.set_id("left-panel");
    left_panel.set_attribute(
        "style",
        &format!(
            "width: {}px; background: {}; padding: 20px; box-sizing: border-box;",
            INPUT_PANEL_WIDTH as u32, PANEL_BG
        ),
    )?;

    // Load digit button
    let load_btn = document.create_element("button")?;
    load_btn.set_id("load-btn");
    load_btn.set_text_content(Some("Random Image"));
    load_btn.set_attribute(
        "style",
        &format!(
            "width: 100%; padding: 10px; font-size: 14px; cursor: pointer; \
             background: {}; color: white; border: none; border-radius: 5px; \
             margin-bottom: 20px; transition: background 0.2s;",
            ACCENT_COLOR
        ),
    )?;
    left_panel.append_child(&load_btn)?;

    // Digit display canvas
    let digit_canvas = document
        .create_element("canvas")?
        .dyn_into::<HtmlCanvasElement>()?;
    digit_canvas.set_id("digit-canvas");
    digit_canvas.set_width(112);
    digit_canvas.set_height(112);
    digit_canvas.set_attribute(
        "style",
        &format!("background: {}; border: 2px solid {}; border-radius: 5px;", BG_COLOR, ACCENT_COLOR),
    )?;
    left_panel.append_child(&digit_canvas)?;

    // Digit label
    let digit_label = document.create_element("div")?;
    digit_label.set_id("digit-label");
    digit_label.set_attribute(
        "style",
        &format!("color: {}; text-align: center; margin-top: 10px; font-size: 14px;", TEXT_COLOR),
    )?;
    digit_label.set_text_content(Some("No digit loaded"));
    left_panel.append_child(&digit_label)?;

    // Instructions
    let instructions = document.create_element("div")?;
    instructions.set_attribute(
        "style",
        &format!(
            "color: {}; font-size: 11px; margin-top: 30px; line-height: 1.6; opacity: 0.7;",
            TEXT_COLOR
        ),
    )?;
    instructions.set_inner_html("Scroll to zoom<br>Drag to pan<br>Hover for details");
    left_panel.append_child(&instructions)?;

    container.append_child(&left_panel)?;

    // Main canvas
    let canvas = document
        .create_element("canvas")?
        .dyn_into::<HtmlCanvasElement>()?;
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
             background: rgba(26, 26, 46, 0.95); \
             color: {}; \
             padding: 12px 16px; \
             border-radius: 8px; \
             font-size: 13px; \
             pointer-events: none; \
             display: none; \
             z-index: 1000; \
             border: 1px solid {}; \
             max-width: 280px; \
             box-shadow: 0 4px 20px rgba(0,0,0,0.3);",
            TEXT_COLOR, ACCENT_COLOR
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

    // Mouse down for drag start
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
        canvas_clone
            .style()
            .set_property("cursor", "grabbing")
            .unwrap();
        let _ = render(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);

    canvas.add_event_listener_with_callback("mousedown", mousedown_closure.as_ref().unchecked_ref())?;
    mousedown_closure.forget();

    // Mouse move for drag and hover
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

    canvas.add_event_listener_with_callback(
        "mousemove",
        mousemove_closure.as_ref().unchecked_ref(),
    )?;
    mousemove_closure.forget();

    // Mouse up for drag end
    let doc_clone = document.clone();
    let canvas_clone = canvas.clone();
    let mouseup_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.end_drag();
            }
        });
        canvas_clone
            .style()
            .set_property("cursor", "grab")
            .unwrap();
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
        canvas_clone
            .style()
            .set_property("cursor", "grab")
            .unwrap();

        if let Some(tooltip) = doc_clone.get_element_by_id("tooltip") {
            let _ = tooltip
                .dyn_into::<HtmlElement>()
                .unwrap()
                .style()
                .set_property("display", "none");
        }

        let _ = render(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);

    canvas.add_event_listener_with_callback(
        "mouseleave",
        mouseleave_closure.as_ref().unchecked_ref(),
    )?;
    mouseleave_closure.forget();

    // Load button click
    let doc_clone = document.clone();
    let load_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.load_next_digit();
            }
        });

        let _ = render_digit(&doc_clone);
        let _ = render(&doc_clone);
    }) as Box<dyn Fn(MouseEvent)>);

    document
        .get_element_by_id("load-btn")
        .ok_or("no load button")?
        .dyn_into::<HtmlElement>()?
        .set_onclick(Some(load_closure.as_ref().unchecked_ref()));
    load_closure.forget();

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
                let all_layers = State::get_all_layers(&s.network);
                let neuron = &all_layers[layer_idx].neurons[neuron_idx];

                // Count weights connected to this neuron
                let mut incoming_weights = Vec::new();
                let mut outgoing_weights = Vec::new();

                for weight in s.network.weights() {
                    if Rc::ptr_eq(weight.destination(), neuron) {
                        incoming_weights.push(weight.weight());
                    }
                    if Rc::ptr_eq(weight.source(), neuron) {
                        outgoing_weights.push(weight.weight());
                    }
                }

                let num_layers = all_layers.len();
                let layer_name = if layer_idx == 0 {
                    "Input Layer"
                } else if layer_idx == num_layers - 1 {
                    "Output Layer"
                } else {
                    "Hidden Layer"
                };

                let mut html = format!(
                    "<div style='margin-bottom: 8px; font-weight: 600; color: {};'>{}</div>\
                     <div style='margin-bottom: 4px;'>Neuron <b>#{}</b></div>\
                     <div style='margin-bottom: 4px;'>Bias: <b>{}</b></div>",
                    NEURON_HOVER, layer_name, neuron_idx, neuron.bias()
                );

                if !incoming_weights.is_empty() {
                    html.push_str(&format!(
                        "<div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #3a4a5c;'>\
                         <b>Incoming:</b> {} weights<br>\
                         Range: [{:.3}, {:.3}]</div>",
                        incoming_weights.len(),
                        incoming_weights.iter().cloned().fold(f64::INFINITY, f64::min),
                        incoming_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    ));
                }

                if !outgoing_weights.is_empty() {
                    html.push_str(&format!(
                        "<div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #3a4a5c;'>\
                         <b>Outgoing:</b> {} weights<br>\
                         Range: [{:.3}, {:.3}]</div>",
                        outgoing_weights.len(),
                        outgoing_weights.iter().cloned().fold(f64::INFINITY, f64::min),
                        outgoing_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    ));
                }

                tooltip.set_inner_html(&html);
                tooltip.style().set_property("display", "block").unwrap();
                tooltip
                    .style()
                    .set_property("left", &format!("{}px", mouse_x + 15.0))
                    .unwrap();
                tooltip
                    .style()
                    .set_property("top", &format!("{}px", mouse_y + 15.0))
                    .unwrap();
            } else {
                tooltip.style().set_property("display", "none").unwrap();
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

    let label_div = document
        .get_element_by_id("digit-label")
        .ok_or("no label div")?;

    STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref s) = *state {
            if let Some(digit_idx) = s.current_digit_idx {
                let digit = &s.digits[digit_idx];

                // Clear
                ctx.set_fill_style_str(BG_COLOR);
                ctx.fill_rect(0.0, 0.0, 112.0, 112.0);

                // Draw digit scaled 4x
                let pixels = digit.pixels();
                for row in 0..28 {
                    for col in 0..28 {
                        let pixel = pixels[row * 28 + col];
                        if pixel > 10 {
                            let gray = pixel;
                            ctx.set_fill_style_str(&format!("rgb({},{},{})", gray, gray, gray));
                            ctx.fill_rect(col as f64 * 4.0, row as f64 * 4.0, 4.0, 4.0);
                        }
                    }
                }

                label_div.set_text_content(Some(&format!(
                    "Label: {} (#{}/{})",
                    digit.label(),
                    digit_idx + 1,
                    s.digits.len()
                )));
            }
        }
    });

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
        ctx.set_fill_style_str(TEXT_COLOR);
        ctx.set_font("16px 'Segoe UI', sans-serif");
        let all_layers = State::get_all_layers(&state.network);
        let num_layers = all_layers.len();
        for i in 0..num_layers {
            let label = if i == 0 {
                "Input Layer"
            } else if i == num_layers - 1 {
                "Output Layer"
            } else {
                "Hidden Layer"
            };
            if let Some((x, _)) = state.get_position(i, 0) {
                ctx.set_text_align("center");
                let _ = ctx.fill_text(label, x, 30.0);
            }
        }

        // Draw weights first (behind neurons)
        let hovered = state.hovered_neuron;

        for weight in state.network.weights() {
            // Find source and dest positions by matching Rc pointers
            let mut source_pos = None;
            let mut dest_pos = None;

            for (layer_idx, layer) in all_layers.iter().enumerate() {
                for (neuron_idx, neuron) in layer.neurons.iter().enumerate() {
                    if Rc::ptr_eq(neuron, weight.source()) {
                        source_pos = state.get_position(layer_idx, neuron_idx);
                    }
                    if Rc::ptr_eq(neuron, weight.destination()) {
                        dest_pos = state.get_position(layer_idx, neuron_idx);
                    }
                }
            }

            if let (Some((x1, y1)), Some((x2, y2))) = (source_pos, dest_pos) {
                // Check if this weight connects to hovered neuron
                let is_connected = hovered.map_or(false, |(layer_idx, neuron_idx)| {
                    let hovered_neuron = &all_layers[layer_idx].neurons[neuron_idx];
                    Rc::ptr_eq(weight.source(), hovered_neuron)
                        || Rc::ptr_eq(weight.destination(), hovered_neuron)
                });

                if is_connected {
                    ctx.set_stroke_style_str(WEIGHT_HIGHLIGHT);
                    ctx.set_line_width(2.0 / state.zoom);
                    ctx.set_global_alpha(0.9);
                } else {
                    ctx.set_stroke_style_str(WEIGHT_COLOR);
                    ctx.set_line_width(0.5 / state.zoom);
                    ctx.set_global_alpha(0.2);
                }

                ctx.begin_path();
                ctx.move_to(x1, y1);
                ctx.line_to(x2, y2);
                ctx.stroke();
            }
        }

        ctx.set_global_alpha(1.0);

        // Draw neurons
        for pos in &state.neuron_positions {
            let is_hovered =
                hovered.map_or(false, |(l, n)| l == pos.layer_idx && n == pos.neuron_idx);

            // Neuron circle
            ctx.begin_path();
            let _ = ctx.arc(pos.x, pos.y, NEURON_RADIUS, 0.0, std::f64::consts::TAU);

            if is_hovered {
                ctx.set_fill_style_str(NEURON_HOVER);
                ctx.set_stroke_style_str("#ffffff");
                ctx.set_line_width(2.0 / state.zoom);
            } else {
                ctx.set_fill_style_str(NEURON_FILL);
                ctx.set_stroke_style_str(NEURON_STROKE);
                ctx.set_line_width(1.5 / state.zoom);
            }

            ctx.fill();
            ctx.stroke();
        }

        ctx.restore();

        Ok(())
    })
}
