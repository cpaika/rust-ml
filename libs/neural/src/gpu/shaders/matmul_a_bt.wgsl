// Tiled Matrix multiplication with B transposed: C = A * B^T
// A is (m x k), B is (n x k), B^T is (k x n), C is (m x n)
// Used for backpropagating delta: delta_prev = delta * W^T
//
// Uses 16x16 tiles loaded into workgroup shared memory.

const TILE_SIZE: u32 = 16u;

struct Dims {
    m: u32,  // rows of A, rows of C
    k: u32,  // cols of A, cols of B (rows of B^T)
    n: u32,  // rows of B (cols of B^T), cols of C
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

// Shared memory tiles
var<workgroup> tile_a: array<f32, 256>;  // 16x16
var<workgroup> tile_b: array<f32, 256>;  // 16x16 - stores B^T tile

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.x;  // Row in C
    let col = global_id.y;  // Col in C
    let local_row = local_id.x;
    let local_col = local_id.y;
    let local_idx = local_row * TILE_SIZE + local_col;

    var sum: f32 = 0.0;
    let num_tiles = (dims.k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load A tile
        let a_row = workgroup_id.x * TILE_SIZE + local_row;
        let a_col = t * TILE_SIZE + local_col;

        if (a_row < dims.m && a_col < dims.k) {
            tile_a[local_idx] = a[a_row * dims.k + a_col];
        } else {
            tile_a[local_idx] = 0.0;
        }

        // Load B^T tile: B^T[t*TILE+local_row, col] = B[col, t*TILE+local_row]
        let b_src_row = workgroup_id.y * TILE_SIZE + local_col;  // Row in B
        let b_src_col = t * TILE_SIZE + local_row;  // Col in B

        if (b_src_row < dims.n && b_src_col < dims.k) {
            tile_b[local_idx] = b[b_src_row * dims.k + b_src_col];
        } else {
            tile_b[local_idx] = 0.0;
        }

        workgroupBarrier();

        // Compute partial sum
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + i] * tile_b[i * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    if (row < dims.m && col < dims.n) {
        c[row * dims.n + col] = sum;
    }
}
