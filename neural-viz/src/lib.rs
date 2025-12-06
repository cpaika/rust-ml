mod viz;

use wasm_bindgen::prelude::*;

// Embed the training data at compile time
const TRAIN_CSV: &[u8] = include_bytes!("../../perceptron/digit-recognizer/train.csv");

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    viz::init(TRAIN_CSV)
}
