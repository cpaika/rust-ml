use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

pub fn print_numbers(n: usize) -> Result<(), Box<dyn Error>> {
    let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("digit-recognizer")
        .join("train.csv");

    let file = File::open(&csv_path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    for (i, result) in reader.records().enumerate() {
        if i >= n {
            break;
        }

        let record = result?;
        let label: u8 = record.get(0).unwrap().parse()?;
        let pixels: Vec<u8> = record
            .iter()
            .skip(1)
            .map(|s| s.parse().unwrap_or(0))
            .collect();

        println!("=== Digit {} (sample {}) ===", label, i + 1);
        print_digit(&pixels);
        println!();
    }

    Ok(())
}

fn print_digit(pixels: &[u8]) {
    const WIDTH: usize = 28;
    const HEIGHT: usize = 28;

    let chars = [' ', '░', '▒', '▓', '█'];

    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            let pixel = pixels[row * WIDTH + col];
            let char_idx = (pixel as usize * (chars.len() - 1)) / 255;
            print!("{}", chars[char_idx]);
        }
        println!();
    }
}
