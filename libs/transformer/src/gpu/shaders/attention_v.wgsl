// Attention output computation: attention_weights @ V
// Input: attn [seq_len, seq_len] (after softmax)
// Input: V [seq_len, d_head]
// Output: out [seq_len, d_head]

struct Dimensions {
    seq_len: u32,
    d_head: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> attn: array<f32>;    // [seq_len * seq_len]
@group(0) @binding(2) var<storage, read> v: array<f32>;       // [seq_len * d_head]
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [seq_len * d_head]

const TILE_SIZE: u32 = 16u;

var<workgroup> attn_tile: array<f32, 256>;  // 16x16 tile
var<workgroup> v_tile: array<f32, 256>;     // 16x16 tile

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let seq_len = dims.seq_len;
    let d_head = dims.d_head;

    let row = workgroup_id.y * TILE_SIZE + local_id.y;
    let col = workgroup_id.x * TILE_SIZE + local_id.x;

    var sum: f32 = 0.0;

    // Tiled matrix multiplication: attn @ V
    let num_tiles = (seq_len + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load attention tile
        let attn_col = t * TILE_SIZE + local_id.x;
        if (row < seq_len && attn_col < seq_len) {
            attn_tile[local_id.y * TILE_SIZE + local_id.x] = attn[row * seq_len + attn_col];
        } else {
            attn_tile[local_id.y * TILE_SIZE + local_id.x] = 0.0;
        }

        // Load V tile
        let v_row = t * TILE_SIZE + local_id.y;
        if (v_row < seq_len && col < d_head) {
            v_tile[local_id.y * TILE_SIZE + local_id.x] = v[v_row * d_head + col];
        } else {
            v_tile[local_id.y * TILE_SIZE + local_id.x] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + attn_tile[local_id.y * TILE_SIZE + i] * v_tile[i * TILE_SIZE + local_id.x];
        }

        workgroupBarrier();
    }

    // Write output
    if (row < seq_len && col < d_head) {
        out[row * d_head + col] = sum;
    }
}
