use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Digit {
    label: u8,
    pixels: Vec<u8>,
}

impl Digit {
    pub fn new(label: u8, pixels: Vec<u8>) -> Self {
        Self { label, pixels }
    }

    pub fn label(&self) -> u8 {
        self.label
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Get pixel value at (row, col) given a specific width
    pub fn pixel_at(&self, row: usize, col: usize, width: usize) -> u8 {
        self.pixels[row * width + col]
    }

    /// Convert digit to ASCII art representation
    pub fn to_ascii_art(&self, width: usize, height: usize) -> String {
        const CHARS: [char; 5] = [' ', '░', '▒', '▓', '█'];

        let mut result = String::new();
        for row in 0..height {
            for col in 0..width {
                let pixel = self.pixel_at(row, col, width);
                let char_idx = (pixel as usize * (CHARS.len() - 1)) / 255;
                result.push(CHARS[char_idx]);
            }
            result.push('\n');
        }
        result
    }
}

/// Parse digits from a file path
pub fn parse_digits<P: AsRef<Path>>(path: P) -> Result<Vec<Digit>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    parse_digits_from_reader(reader)
}

/// Parse digits from raw bytes (useful for embedded data)
pub fn parse_digits_from_bytes(data: &[u8]) -> Result<Vec<Digit>, Box<dyn Error>> {
    let reader = BufReader::new(data);
    parse_digits_from_reader(reader)
}

fn parse_digits_from_reader<R: Read + BufRead>(reader: R) -> Result<Vec<Digit>, Box<dyn Error>> {
    let mut digits = Vec::new();
    let mut lines = reader.lines();

    // Skip header
    lines.next();

    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split(',');

        let label: u8 = parts
            .next()
            .ok_or("missing label")?
            .parse()?;

        let pixels: Vec<u8> = parts
            .map(|s| s.parse().unwrap_or(0))
            .collect();

        digits.push(Digit::new(label, pixels));
    }

    Ok(digits)
}
