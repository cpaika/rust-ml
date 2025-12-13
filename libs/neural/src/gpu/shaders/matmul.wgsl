// Matrix multiplication compute shader
// C = A * B where A is (m x k), B is (k x n), C is (m x n)

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    // Bounds check
    if (row >= dims.m || col >= dims.n) {
        return;
    }

    // Compute dot product for this cell
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        let a_idx = row * dims.k + i;
        let b_idx = i * dims.n + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    let c_idx = row * dims.n + col;
    c[c_idx] = sum;
}
