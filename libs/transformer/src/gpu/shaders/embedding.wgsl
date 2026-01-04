// Embedding lookup: gather rows from embedding table
// Input: token_ids [seq_len] (as u32)
// Embedding table: [vocab_size, d_model]
// Output: [seq_len, d_model]

struct Dimensions {
    seq_len: u32,
    d_model: u32,
    vocab_size: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;     // [seq_len]
@group(0) @binding(2) var<storage, read> embeddings: array<f32>;    // [vocab_size * d_model]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;  // [seq_len * d_model]

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let seq_len = dims.seq_len;
    let d_model = dims.d_model;

    if (idx >= seq_len * d_model) {
        return;
    }

    let pos = idx / d_model;
    let dim = idx % d_model;

    let token_id = token_ids[pos];

    // Bounds check
    if (token_id < dims.vocab_size) {
        output[idx] = embeddings[token_id * d_model + dim];
    } else {
        output[idx] = 0.0;
    }
}
