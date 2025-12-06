use common::{parse_digits_from_bytes, Digit};
use wasm_bindgen::prelude::*;
use web_sys::{
    CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, MouseEvent, WheelEvent,
};

const DIGIT_SIZE: u32 = 28;
const TILE_SIZE: u32 = 32; // Slightly larger than digit for padding
const COLS: u32 = 200; // 200 columns = 42000 / 200 = 210 rows

struct State {
    digits: Vec<Digit>,
    // View transform
    offset_x: f64,
    offset_y: f64,
    zoom: f64,
    // Dragging state
    is_dragging: bool,
    drag_start_x: f64,
    drag_start_y: f64,
    // Canvas dimensions
    canvas_width: u32,
    canvas_height: u32,
    // Hover state
    hover_index: Option<usize>,
}

impl State {
    fn new(digits: Vec<Digit>, canvas_width: u32, canvas_height: u32) -> Self {
        Self {
            digits,
            offset_x: 0.0,
            offset_y: 0.0,
            zoom: 1.0,
            is_dragging: false,
            drag_start_x: 0.0,
            drag_start_y: 0.0,
            canvas_width,
            canvas_height,
            hover_index: None,
        }
    }

    fn screen_to_world(&self, screen_x: f64, screen_y: f64) -> (f64, f64) {
        let world_x = (screen_x - self.offset_x) / self.zoom;
        let world_y = (screen_y - self.offset_y) / self.zoom;
        (world_x, world_y)
    }

    fn get_digit_at_screen_pos(&self, screen_x: f64, screen_y: f64) -> Option<usize> {
        let (world_x, world_y) = self.screen_to_world(screen_x, screen_y);

        if world_x < 0.0 || world_y < 0.0 {
            return None;
        }

        let col = (world_x / TILE_SIZE as f64) as u32;
        let row = (world_y / TILE_SIZE as f64) as u32;

        if col >= COLS {
            return None;
        }

        let index = (row * COLS + col) as usize;
        if index < self.digits.len() {
            Some(index)
        } else {
            None
        }
    }

    fn zoom_at(&mut self, screen_x: f64, screen_y: f64, delta: f64) {
        let zoom_factor = if delta > 0.0 { 0.9 } else { 1.1 };
        let new_zoom = (self.zoom * zoom_factor).clamp(0.1, 5.0);

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
}

thread_local! {
    static STATE: std::cell::RefCell<Option<State>> = const { std::cell::RefCell::new(None) };
}

pub fn init(csv_data: &[u8]) -> Result<(), JsValue> {
    let digits =
        parse_digits_from_bytes(csv_data).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;

    let canvas_width = window.inner_width()?.as_f64().unwrap_or(800.0) as u32;
    let canvas_height = window.inner_height()?.as_f64().unwrap_or(600.0) as u32;

    STATE.with(|state| {
        *state.borrow_mut() = Some(State::new(digits, canvas_width, canvas_height));
    });

    setup_ui(&document, canvas_width, canvas_height)?;
    setup_handlers(&document)?;
    render(&document)?;

    Ok(())
}

fn setup_ui(document: &Document, width: u32, height: u32) -> Result<(), JsValue> {
    let body = document.body().ok_or("no body")?;
    body.set_attribute("style", "margin: 0; overflow: hidden;")?;

    // Canvas
    let canvas = document
        .create_element("canvas")?
        .dyn_into::<HtmlCanvasElement>()?;
    canvas.set_id("main-canvas");
    canvas.set_width(width);
    canvas.set_height(height);
    canvas.set_attribute("style", "display: block; cursor: grab;")?;
    body.append_child(&canvas)?;

    // Tooltip
    let tooltip = document.create_element("div")?;
    tooltip.set_id("tooltip");
    tooltip.set_attribute(
        "style",
        "position: fixed; \
         background: rgba(0,0,0,0.8); \
         color: white; \
         padding: 8px 12px; \
         border-radius: 4px; \
         font-family: sans-serif; \
         font-size: 14px; \
         pointer-events: none; \
         display: none; \
         z-index: 1000;",
    )?;
    body.append_child(&tooltip)?;

    // Instructions
    let instructions = document.create_element("div")?;
    instructions.set_attribute(
        "style",
        "position: fixed; \
         bottom: 10px; \
         left: 10px; \
         background: rgba(0,0,0,0.7); \
         color: white; \
         padding: 10px; \
         border-radius: 4px; \
         font-family: sans-serif; \
         font-size: 12px;",
    )?;
    instructions.set_inner_html("Scroll to zoom • Drag to pan • Hover for label");
    body.append_child(&instructions)?;

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

    canvas
        .add_event_listener_with_callback("mousedown", mousedown_closure.as_ref().unchecked_ref())?;
    mousedown_closure.forget();

    // Mouse move for drag and hover
    let doc_clone = document.clone();
    let mousemove_closure = Closure::wrap(Box::new(move |event: MouseEvent| {
        let x = event.offset_x() as f64;
        let y = event.offset_y() as f64;

        let needs_render = STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                let was_dragging = s.is_dragging;
                s.drag(x, y);

                // Update hover
                let old_hover = s.hover_index;
                s.hover_index = s.get_digit_at_screen_pos(x, y);

                was_dragging || old_hover != s.hover_index
            } else {
                false
            }
        });

        // Update tooltip
        let _ = update_tooltip(&doc_clone, x, y);

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

    canvas
        .add_event_listener_with_callback("mouseup", mouseup_closure.as_ref().unchecked_ref())?;
    mouseup_closure.forget();

    // Mouse leave
    let doc_clone = document.clone();
    let canvas_clone = canvas.clone();
    let mouseleave_closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
        STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.end_drag();
                s.hover_index = None;
            }
        });
        canvas_clone
            .style()
            .set_property("cursor", "grab")
            .unwrap();

        // Hide tooltip
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
            if let Some(index) = s.hover_index {
                let digit = &s.digits[index];
                tooltip.set_inner_html(&format!(
                    "<b>Label:</b> {}<br><b>Index:</b> {}",
                    digit.label(),
                    index
                ));
                tooltip
                    .style()
                    .set_property("display", "block")
                    .unwrap();
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
        ctx.set_fill_style_str("#1a1a2e");
        ctx.fill_rect(0.0, 0.0, state.canvas_width as f64, state.canvas_height as f64);

        // Calculate visible range
        let (min_world_x, min_world_y) = state.screen_to_world(0.0, 0.0);
        let (max_world_x, max_world_y) =
            state.screen_to_world(state.canvas_width as f64, state.canvas_height as f64);

        let start_col = ((min_world_x / TILE_SIZE as f64).floor() as i32).max(0) as u32;
        let end_col = ((max_world_x / TILE_SIZE as f64).ceil() as u32).min(COLS);
        let start_row = ((min_world_y / TILE_SIZE as f64).floor() as i32).max(0) as u32;
        let total_rows = (state.digits.len() as u32 + COLS - 1) / COLS;
        let end_row = ((max_world_y / TILE_SIZE as f64).ceil() as u32).min(total_rows);

        // Draw visible tiles
        for row in start_row..end_row {
            for col in start_col..end_col {
                let index = (row * COLS + col) as usize;
                if index >= state.digits.len() {
                    continue;
                }

                let digit = &state.digits[index];

                // Calculate screen position
                let world_x = col as f64 * TILE_SIZE as f64;
                let world_y = row as f64 * TILE_SIZE as f64;
                let screen_x = world_x * state.zoom + state.offset_x;
                let screen_y = world_y * state.zoom + state.offset_y;

                let tile_screen_size = TILE_SIZE as f64 * state.zoom;
                let digit_screen_size = DIGIT_SIZE as f64 * state.zoom;
                let padding = (tile_screen_size - digit_screen_size) / 2.0;

                // Draw digit pixels
                let pixels = digit.pixels();
                let pixel_size = state.zoom;

                // Only draw individual pixels if zoomed in enough
                if pixel_size >= 0.5 {
                    for py in 0..DIGIT_SIZE {
                        for px in 0..DIGIT_SIZE {
                            let pixel = pixels[(py * DIGIT_SIZE + px) as usize];
                            if pixel > 10 {
                                // Skip very dark pixels for performance
                                let gray = pixel;
                                ctx.set_fill_style_str(&format!("rgb({},{},{})", gray, gray, gray));
                                ctx.fill_rect(
                                    screen_x + padding + px as f64 * pixel_size,
                                    screen_y + padding + py as f64 * pixel_size,
                                    pixel_size.max(1.0),
                                    pixel_size.max(1.0),
                                );
                            }
                        }
                    }
                } else {
                    // When zoomed out, draw a simple representation
                    ctx.set_fill_style_str("#666");
                    ctx.fill_rect(
                        screen_x + padding,
                        screen_y + padding,
                        digit_screen_size,
                        digit_screen_size,
                    );
                }

                // Highlight hovered tile
                if state.hover_index == Some(index) {
                    ctx.set_stroke_style_str("#00ff00");
                    ctx.set_line_width(2.0);
                    ctx.stroke_rect(screen_x, screen_y, tile_screen_size, tile_screen_size);
                }
            }
        }

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_navigation() {
        let digits = vec![
            Digit::new(0, vec![0; 784]),
            Digit::new(1, vec![0; 784]),
            Digit::new(2, vec![0; 784]),
        ];
        let state = State::new(digits, 800, 600);

        assert_eq!(state.zoom, 1.0);
        assert_eq!(state.offset_x, 0.0);
        assert_eq!(state.offset_y, 0.0);
    }

    #[test]
    fn test_screen_to_world() {
        let digits = vec![Digit::new(0, vec![0; 784])];
        let mut state = State::new(digits, 800, 600);

        // At zoom 1, offset 0
        let (wx, wy) = state.screen_to_world(100.0, 200.0);
        assert_eq!(wx, 100.0);
        assert_eq!(wy, 200.0);

        // At zoom 2
        state.zoom = 2.0;
        let (wx, wy) = state.screen_to_world(100.0, 200.0);
        assert_eq!(wx, 50.0);
        assert_eq!(wy, 100.0);

        // With offset
        state.offset_x = 50.0;
        state.offset_y = 100.0;
        let (wx, wy) = state.screen_to_world(100.0, 200.0);
        assert_eq!(wx, 25.0);
        assert_eq!(wy, 50.0);
    }

    #[test]
    fn test_get_digit_at_pos() {
        let digits = vec![
            Digit::new(0, vec![0; 784]),
            Digit::new(1, vec![0; 784]),
            Digit::new(2, vec![0; 784]),
        ];
        let state = State::new(digits, 800, 600);

        // First tile (0,0)
        let idx = state.get_digit_at_screen_pos(5.0, 5.0);
        assert_eq!(idx, Some(0));

        // Second tile (1,0)
        let idx = state.get_digit_at_screen_pos(TILE_SIZE as f64 + 5.0, 5.0);
        assert_eq!(idx, Some(1));

        // Out of bounds negative
        let idx = state.get_digit_at_screen_pos(-5.0, 5.0);
        assert_eq!(idx, None);
    }

    #[test]
    fn test_zoom_clamp() {
        let digits = vec![Digit::new(0, vec![0; 784])];
        let mut state = State::new(digits, 800, 600);

        // Zoom in a lot
        for _ in 0..100 {
            state.zoom_at(400.0, 300.0, -1.0);
        }
        assert!(state.zoom <= 5.0);

        // Zoom out a lot
        for _ in 0..100 {
            state.zoom_at(400.0, 300.0, 1.0);
        }
        assert!(state.zoom >= 0.1);
    }

    #[test]
    fn test_drag() {
        let digits = vec![Digit::new(0, vec![0; 784])];
        let mut state = State::new(digits, 800, 600);

        state.start_drag(100.0, 100.0);
        assert!(state.is_dragging);

        state.drag(150.0, 120.0);
        assert_eq!(state.offset_x, 50.0);
        assert_eq!(state.offset_y, 20.0);

        state.end_drag();
        assert!(!state.is_dragging);
    }
}
