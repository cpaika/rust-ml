// Matrix multiplication with A transposed: C = A^T * B
// A is (k x m), A^T is (m x k), B is (k x n), C is (m x n)
// Used for computing weight gradients: dW = input^T * delta

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;  // Row in C (and in A^T, which is col in A)
    let col = global_id.y;  // Col in C (and in B)

    if (row >= dims.m || col >= dims.n) {
        return;
    }

    // C[row, col] = sum over i of A^T[row, i] * B[i, col]
    //             = sum over i of A[i, row] * B[i, col]
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        let a_idx = i * dims.m + row;  // A[i, row] in row-major
        let b_idx = i * dims.n + col;  // B[i, col] in row-major
        sum = sum + a[a_idx] * b[b_idx];
    }

    let c_idx = row * dims.n + col;
    c[c_idx] = sum;
}
