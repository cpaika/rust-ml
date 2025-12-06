mod commands;
mod print_numbers;

use clap::Parser;
use commands::{Cli, Commands};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::PrintNumbers { n } => {
            print_numbers::print_numbers(n)?;
        }
    }

    Ok(())
}
