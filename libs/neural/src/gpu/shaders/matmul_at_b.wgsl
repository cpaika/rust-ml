// Tiled Matrix multiplication with A transposed: C = A^T * B
// A is (k x m), A^T is (m x k), B is (k x n), C is (m x n)
// Used for computing weight gradients: dW = input^T * delta
//
// Uses 16x16 tiles loaded into workgroup shared memory.

const TILE_SIZE: u32 = 16u;

struct Dims {
    m: u32,  // cols of A (rows of A^T), rows of C
    k: u32,  // rows of A (cols of A^T), rows of B
    n: u32,  // cols of B, cols of C
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

// Shared memory tiles
var<workgroup> tile_a: array<f32, 256>;  // 16x16 - stores A^T tile
var<workgroup> tile_b: array<f32, 256>;  // 16x16

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
        // Load A^T tile: A^T[row, t*TILE+local_col] = A[t*TILE+local_col, row]
        let a_src_row = t * TILE_SIZE + local_col;  // Row in A
        let a_src_col = workgroup_id.x * TILE_SIZE + local_row;  // Col in A

        if (a_src_row < dims.k && a_src_col < dims.m) {
            tile_a[local_idx] = a[a_src_row * dims.m + a_src_col];
        } else {
            tile_a[local_idx] = 0.0;
        }

        // Load B tile
        let b_row = t * TILE_SIZE + local_row;
        let b_col = workgroup_id.y * TILE_SIZE + local_col;

        if (b_row < dims.k && b_col < dims.n) {
            tile_b[local_idx] = b[b_row * dims.n + b_col];
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
