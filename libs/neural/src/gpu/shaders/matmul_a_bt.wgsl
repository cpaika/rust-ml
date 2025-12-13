// Matrix multiplication with B transposed: C = A * B^T
// A is (m x k), B is (n x k), B^T is (k x n), C is (m x n)
// Used for backpropagating delta: delta_prev = delta * W^T

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;  // Row in C (and in A)
    let col = global_id.y;  // Col in C (and row in B, col in B^T)

    if (row >= dims.m || col >= dims.n) {
        return;
    }

    // C[row, col] = sum over i of A[row, i] * B^T[i, col]
    //             = sum over i of A[row, i] * B[col, i]
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        let a_idx = row * dims.k + i;  // A[row, i] in row-major
        let b_idx = col * dims.k + i;  // B[col, i] in row-major
        sum = sum + a[a_idx] * b[b_idx];
    }

    let c_idx = row * dims.n + col;
    c[c_idx] = sum;
}
