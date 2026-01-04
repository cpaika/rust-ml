// Attention QK^T computation with causal masking
// Computes: scores = Q @ K^T / sqrt(d_head), with causal mask applied
// Input: Q [seq_len, d_head], K [seq_len, d_head]
// Output: scores [seq_len, seq_len]

struct Dimensions {
    seq_len: u32,
    d_head: u32,
    scale: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [seq_len * d_head]
@group(0) @binding(2) var<storage, read> k: array<f32>;        // [seq_len * d_head]
@group(0) @binding(3) var<storage, read_write> scores: array<f32>; // [seq_len * seq_len]

const TILE_SIZE: u32 = 16u;
const NEG_INF: f32 = -1e9;

var<workgroup> q_tile: array<f32, 256>;  // 16x16 tile for Q
var<workgroup> k_tile: array<f32, 256>;  // 16x16 tile for K

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let seq_len = dims.seq_len;
    let d_head = dims.d_head;
    let scale = dims.scale;

    let row = workgroup_id.y * TILE_SIZE + local_id.y;
    let col = workgroup_id.x * TILE_SIZE + local_id.x;

    var sum: f32 = 0.0;

    // Tiled matrix multiplication
    let num_tiles = (d_head + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load Q tile
        let q_col = t * TILE_SIZE + local_id.x;
        if (row < seq_len && q_col < d_head) {
            q_tile[local_id.y * TILE_SIZE + local_id.x] = q[row * d_head + q_col];
        } else {
            q_tile[local_id.y * TILE_SIZE + local_id.x] = 0.0;
        }

        // Load K tile (transposed access - we want K^T)
        let k_row = t * TILE_SIZE + local_id.y;
        if (col < seq_len && k_row < d_head) {
            k_tile[local_id.y * TILE_SIZE + local_id.x] = k[col * d_head + k_row];
        } else {
            k_tile[local_id.y * TILE_SIZE + local_id.x] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + q_tile[local_id.y * TILE_SIZE + i] * k_tile[i * TILE_SIZE + local_id.x];
        }

        workgroupBarrier();
    }

    // Apply scaling and causal mask
    if (row < seq_len && col < seq_len) {
        var score = sum * scale;

        // Causal masking: positions can only attend to previous positions
        if (col > row) {
            score = NEG_INF;
        }

        scores[row * seq_len + col] = score;
    }
}
