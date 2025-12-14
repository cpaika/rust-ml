# wasm-serve

A simple, fast HTTP server for serving WASM applications during development.

## Features

- Serves static files with correct MIME types
- Proper `application/wasm` content type for `.wasm` files
- CORS headers enabled for cross-origin requests
- Cross-Origin isolation headers for SharedArrayBuffer support
- Directory traversal protection
- URL decoding support
- Automatic `index.html` fallback

## Installation

From the workspace root:

```bash
cargo install --path wasm-serve
```

Or build in release mode:

```bash
cargo build --release -p wasm-serve
```

The binary will be at `target/release/wasm-serve`.

## Usage

```bash
# Serve current directory on port 8080
wasm-serve

# Serve a specific directory
wasm-serve /path/to/your/wasm/app

# Serve on a different port
wasm-serve -p 3000 /path/to/your/wasm/app

# Bind to all interfaces (for network access)
wasm-serve --host 0.0.0.0 -p 8080 .
```

### Options

```
Usage: wasm-serve [OPTIONS] [DIRECTORY]

Arguments:
  [DIRECTORY]  Directory to serve files from [default: .]

Options:
  -p, --port <PORT>  Port to listen on [default: 8080]
      --host <HOST>  Host to bind to [default: 127.0.0.1]
  -h, --help         Print help
```

## Example: Serving neural-viz

```bash
# From the rust-ml workspace root
wasm-serve neural-viz
```

Then open http://127.0.0.1:8080/www/ in your browser.

## Why not Python?

- Faster startup
- Correct WASM MIME type out of the box
- CORS and Cross-Origin isolation headers included
- No Python installation required
- Single static binary
