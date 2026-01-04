// General matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;         // [M * K]
@group(0) @binding(2) var<storage, read> b: array<f32>;         // [K * N]
@group(0) @binding(3) var<storage, read_write> c: array<f32>;   // [M * N]

const TILE_SIZE: u32 = 16u;

var<workgroup> a_tile: array<f32, 256>;  // 16x16 tile
var<workgroup> b_tile: array<f32, 256>;  // 16x16 tile

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let M = dims.M;
    let K = dims.K;
    let N = dims.N;

    let row = workgroup_id.y * TILE_SIZE + local_id.y;
    let col = workgroup_id.x * TILE_SIZE + local_id.x;

    var sum: f32 = 0.0;

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load A tile
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            a_tile[local_id.y * TILE_SIZE + local_id.x] = a[row * K + a_col];
        } else {
            a_tile[local_id.y * TILE_SIZE + local_id.x] = 0.0;
        }

        // Load B tile
        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            b_tile[local_id.y * TILE_SIZE + local_id.x] = b[b_row * N + col];
        } else {
            b_tile[local_id.y * TILE_SIZE + local_id.x] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + a_tile[local_id.y * TILE_SIZE + i] * b_tile[i * TILE_SIZE + local_id.x];
        }

        workgroupBarrier();
    }

    // Write output
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}
