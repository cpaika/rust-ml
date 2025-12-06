use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "nx")]
#[command(about = "Neural network tools for digit recognition")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Print n sample digit images from the training data as ASCII art
    PrintNumbers {
        /// Number of digits to print
        n: usize,
    },
}
