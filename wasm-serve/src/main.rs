//! A simple HTTP server for serving WASM applications
//!
//! Serves static files with correct MIME types, especially for WASM.

use clap::Parser;
use mime_guess::MimeGuess;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use tiny_http::{Header, Response, Server, StatusCode};

#[derive(Parser, Debug)]
#[command(name = "wasm-serve")]
#[command(about = "A simple HTTP server for serving WASM applications")]
struct Args {
    /// Directory to serve files from
    #[arg(default_value = ".")]
    directory: PathBuf,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

fn main() {
    let args = Args::parse();

    let root_dir = args.directory.canonicalize().unwrap_or_else(|_| {
        eprintln!("Error: Directory '{}' not found", args.directory.display());
        std::process::exit(1);
    });

    let addr = format!("{}:{}", args.host, args.port);
    let server = Server::http(&addr).unwrap_or_else(|e| {
        eprintln!("Error starting server: {}", e);
        std::process::exit(1);
    });

    println!("Serving '{}' at http://{}", root_dir.display(), addr);
    println!("Press Ctrl+C to stop");

    for request in server.incoming_requests() {
        let url_path = request.url().to_string();
        let url_path = url_path.split('?').next().unwrap_or(&url_path);

        // Decode URL-encoded characters
        let decoded_path = urlencoded_decode(url_path);

        // Determine file path
        let relative_path = decoded_path.trim_start_matches('/');
        let file_path = if relative_path.is_empty() {
            root_dir.join("index.html")
        } else {
            root_dir.join(relative_path)
        };

        // Security: prevent directory traversal
        let canonical = match file_path.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                let _ = request.respond(not_found());
                continue;
            }
        };

        if !canonical.starts_with(&root_dir) {
            let _ = request.respond(forbidden());
            continue;
        }

        // If directory, try index.html
        let final_path = if canonical.is_dir() {
            canonical.join("index.html")
        } else {
            canonical
        };

        // Serve the file
        match serve_file(&final_path) {
            Ok(response) => {
                println!("{} {} -> 200", request.method(), url_path);
                let _ = request.respond(response);
            }
            Err(_) => {
                println!("{} {} -> 404", request.method(), url_path);
                let _ = request.respond(not_found());
            }
        }
    }
}

fn serve_file(path: &Path) -> Result<Response<std::io::Cursor<Vec<u8>>>, std::io::Error> {
    let mut file = fs::File::open(path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;

    let mime = get_mime_type(path);
    let content_type = Header::from_bytes("Content-Type", mime).unwrap();

    // Add CORS headers for WASM
    let cors = Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap();

    // Special header for SharedArrayBuffer support (needed by some WASM apps)
    let coop = Header::from_bytes("Cross-Origin-Opener-Policy", "same-origin").unwrap();
    let coep = Header::from_bytes("Cross-Origin-Embedder-Policy", "require-corp").unwrap();

    Ok(Response::from_data(contents)
        .with_header(content_type)
        .with_header(cors)
        .with_header(coop)
        .with_header(coep))
}

fn get_mime_type(path: &Path) -> &'static str {
    // Handle WASM specially since mime_guess might not have it
    if let Some(ext) = path.extension() {
        match ext.to_str() {
            Some("wasm") => return "application/wasm",
            Some("js") => return "application/javascript",
            Some("mjs") => return "application/javascript",
            _ => {}
        }
    }

    MimeGuess::from_path(path)
        .first_raw()
        .unwrap_or("application/octet-stream")
}

fn not_found() -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string("404 Not Found")
        .with_status_code(StatusCode(404))
}

fn forbidden() -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string("403 Forbidden")
        .with_status_code(StatusCode(403))
}

fn urlencoded_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            } else {
                result.push('%');
                result.push_str(&hex);
            }
        } else if c == '+' {
            result.push(' ');
        } else {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_get_mime_type_wasm() {
        let path = Path::new("test.wasm");
        assert_eq!(get_mime_type(path), "application/wasm");
    }

    #[test]
    fn test_get_mime_type_js() {
        let path = Path::new("test.js");
        assert_eq!(get_mime_type(path), "application/javascript");
    }

    #[test]
    fn test_get_mime_type_html() {
        let path = Path::new("test.html");
        assert_eq!(get_mime_type(path), "text/html");
    }

    #[test]
    fn test_get_mime_type_css() {
        let path = Path::new("style.css");
        assert_eq!(get_mime_type(path), "text/css");
    }

    #[test]
    fn test_urlencoded_decode() {
        assert_eq!(urlencoded_decode("hello%20world"), "hello world");
        assert_eq!(urlencoded_decode("foo+bar"), "foo bar");
        assert_eq!(urlencoded_decode("test%2Fpath"), "test/path");
        assert_eq!(urlencoded_decode("normal"), "normal");
    }

    #[test]
    fn test_serve_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.html");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"<html>test</html>").unwrap();

        let response = serve_file(&file_path).unwrap();
        assert_eq!(response.status_code().0, 200);
    }

    #[test]
    fn test_serve_file_not_found() {
        let result = serve_file(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }
}
