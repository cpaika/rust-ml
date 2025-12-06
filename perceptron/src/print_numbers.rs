use common::parse_digits;
use std::error::Error;
use std::path::PathBuf;

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

pub fn print_numbers(n: usize) -> Result<(), Box<dyn Error>> {
    let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("digit-recognizer")
        .join("train.csv");

    let digits = parse_digits(&csv_path)?;

    for (i, digit) in digits.iter().take(n).enumerate() {
        println!("=== Digit {} (sample {}) ===", digit.label(), i + 1);
        print!("{}", digit.to_ascii_art(WIDTH, HEIGHT));
        println!();
    }

    Ok(())
}
