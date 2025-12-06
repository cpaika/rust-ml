mod digit;

pub use digit::{Digit, parse_digits, parse_digits_from_bytes};

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_CSV: &str = "label,pixel0,pixel1,pixel2,pixel3
1,0,128,255,64
7,255,0,128,32";

    #[test]
    fn test_parse_digits_from_csv() {
        let digits = parse_digits_from_bytes(TEST_CSV.as_bytes()).unwrap();

        assert_eq!(digits.len(), 2);

        assert_eq!(digits[0].label(), 1);
        assert_eq!(digits[0].pixels(), &[0, 128, 255, 64]);

        assert_eq!(digits[1].label(), 7);
        assert_eq!(digits[1].pixels(), &[255, 0, 128, 32]);
    }

    #[test]
    fn test_digit_pixel_access() {
        let digits = parse_digits_from_bytes(TEST_CSV.as_bytes()).unwrap();
        let digit = &digits[0];

        assert_eq!(digit.pixel_at(0, 0, 2), 0);   // row 0, col 0, width 2
        assert_eq!(digit.pixel_at(0, 1, 2), 128); // row 0, col 1, width 2
        assert_eq!(digit.pixel_at(1, 0, 2), 255); // row 1, col 0, width 2
        assert_eq!(digit.pixel_at(1, 1, 2), 64);  // row 1, col 1, width 2
    }

    #[test]
    fn test_empty_csv_returns_empty_vec() {
        let csv = "label,pixel0\n";
        let digits = parse_digits_from_bytes(csv.as_bytes()).unwrap();
        assert!(digits.is_empty());
    }

    #[test]
    fn test_digit_to_ascii_art() {
        let digit = Digit::new(5, vec![0, 64, 128, 192, 255, 32, 96, 160, 224]);
        let art = digit.to_ascii_art(3, 3);

        // Each row should have 3 characters + newline
        let lines: Vec<&str> = art.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].chars().count(), 3);
    }
}
