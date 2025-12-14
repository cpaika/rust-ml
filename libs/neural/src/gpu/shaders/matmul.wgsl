// Tiled Matrix multiplication compute shader with shared memory
// C = A * B where A is (m x k), B is (k x n), C is (m x n)
//
// Uses 16x16 tiles loaded into workgroup shared memory to improve
// memory access patterns and reduce global memory bandwidth.

const TILE_SIZE: u32 = 16u;

struct Dims {
    m: u32,  // rows of A
    k: u32,  // cols of A / rows of B
    n: u32,  // cols of B
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

// Shared memory tiles for A and B
var<workgroup> tile_a: array<f32, 256>;  // 16x16
var<workgroup> tile_b: array<f32, 256>;  // 16x16

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;

    // Linear index within workgroup for shared memory access
    let local_idx = local_row * TILE_SIZE + local_col;

    // Accumulator for this thread's output element
    var sum: f32 = 0.0;

    // Number of tiles we need to process along the k dimension
    let num_tiles = (dims.k + TILE_SIZE - 1u) / TILE_SIZE;

    // Process tiles
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Collaborative loading of tile_a
        // Each thread loads one element of the tile
        let a_row = workgroup_id.x * TILE_SIZE + local_row;
        let a_col = t * TILE_SIZE + local_col;

        if (a_row < dims.m && a_col < dims.k) {
            tile_a[local_idx] = a[a_row * dims.k + a_col];
        } else {
            tile_a[local_idx] = 0.0;
        }

        // Collaborative loading of tile_b
        let b_row = t * TILE_SIZE + local_row;
        let b_col = workgroup_id.y * TILE_SIZE + local_col;

        if (b_row < dims.k && b_col < dims.n) {
            tile_b[local_idx] = b[b_row * dims.n + b_col];
        } else {
            tile_b[local_idx] = 0.0;
        }

        // Ensure all threads have loaded their data
        workgroupBarrier();

        // Compute partial dot product using shared memory
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + i] * tile_b[i * TILE_SIZE + local_col];
        }

        // Ensure all threads are done before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < dims.m && col < dims.n) {
        c[row * dims.n + col] = sum;
    }
}
