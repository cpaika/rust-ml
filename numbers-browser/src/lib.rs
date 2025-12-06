mod browser;

use wasm_bindgen::prelude::*;

// Embed the training data at compile time
const TRAIN_CSV: &[u8] = include_bytes!("../../perceptron/digit-recognizer/train.csv");

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    browser::init(TRAIN_CSV)
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::parse_digits_from_bytes;

    #[test]
    fn test_embedded_csv_parses() {
        let digits = parse_digits_from_bytes(TRAIN_CSV).unwrap();
        assert!(!digits.is_empty());
        // MNIST training set has 42000 samples
        assert_eq!(digits.len(), 42000);
    }

    #[test]
    fn test_first_digit_is_valid() {
        let digits = parse_digits_from_bytes(TRAIN_CSV).unwrap();
        let first = &digits[0];

        // Label should be 0-9
        assert!(first.label() <= 9);
        // MNIST images are 28x28 = 784 pixels
        assert_eq!(first.pixels().len(), 784);
    }

    #[test]
    fn test_all_labels_in_range() {
        let digits = parse_digits_from_bytes(TRAIN_CSV).unwrap();
        for digit in digits.iter().take(100) {
            assert!(digit.label() <= 9, "Label {} is out of range", digit.label());
        }
    }
}
