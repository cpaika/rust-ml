//! Language Model Visualization
//!
//! WASM-compatible visualization for training and interacting with
//! transformer language models.

mod viz;
mod completion;
mod metrics;

#[cfg(test)]
mod tests;

use wasm_bindgen::prelude::*;

/// Initialize the visualization app
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    // Set up panic hook for better error messages in WASM
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    viz::init()
}

/// Re-export for JavaScript interop
pub use viz::*;
pub use completion::*;
pub use metrics::*;
