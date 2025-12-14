// Layer normalization
// For each row (position): normalize to mean=0, var=1, then scale by gamma and shift by beta
// Input: x [seq_len, d_model]
// Parameters: gamma [d_model], beta [d_model]
// Output: out [seq_len, d_model]

struct Dimensions {
    seq_len: u32,
    d_model: u32,
    eps: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> input: array<f32>;      // [seq_len * d_model]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;      // [d_model]
@group(0) @binding(3) var<storage, read> beta: array<f32>;       // [d_model]
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [seq_len * d_model]

const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> shared_sum: array<f32, 64>;
var<workgroup> shared_sq_sum: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let pos = workgroup_id.x;
    let tid = local_id.x;
    let d_model = dims.d_model;
    let eps = dims.eps;

    if (pos >= dims.seq_len) {
        return;
    }

    let row_start = pos * d_model;

    // Step 1: Compute sum and squared sum for mean and variance
    var local_sum: f32 = 0.0;
    var local_sq_sum: f32 = 0.0;

    for (var i = tid; i < d_model; i = i + WORKGROUP_SIZE) {
        let val = input[row_start + i];
        local_sum = local_sum + val;
        local_sq_sum = local_sq_sum + val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sq_sum[tid] = local_sq_sum;
    workgroupBarrier();

    // Parallel reduction
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
            shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let mean = shared_sum[0] / f32(d_model);
    let variance = shared_sq_sum[0] / f32(d_model) - mean * mean;
    let inv_std = 1.0 / sqrt(variance + eps);

    workgroupBarrier();

    // Step 2: Normalize, scale, and shift
    for (var i = tid; i < d_model; i = i + WORKGROUP_SIZE) {
        let val = input[row_start + i];
        let normalized = (val - mean) * inv_std;
        output[row_start + i] = normalized * gamma[i] + beta[i];
    }
}
