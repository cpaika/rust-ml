// Softmax activation compute shader
// Computes numerically stable softmax: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// Designed for small output sizes (e.g., 10 classes for MNIST)
// Uses a single workgroup with shared memory for reductions

struct Params {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

// Shared memory for reductions (max 64 elements supported)
var<workgroup> shared_data: array<f32, 64>;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let tid = local_idx;
    let size = params.size;

    // Load data into shared memory (pad with -inf for threads beyond size)
    if (tid < size) {
        shared_data[tid] = data[tid];
    } else {
        shared_data[tid] = -3.402823e+38; // -FLT_MAX for max reduction
    }
    workgroupBarrier();

    // Step 1: Find maximum (parallel reduction)
    // For small sizes like 10, this is overkill but correct
    var stride: u32 = 32u;
    while (stride > 0u) {
        if (tid < stride && tid + stride < 64u) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 has the max
    if (tid == 0u) {
        shared_max = shared_data[0];
    }
    workgroupBarrier();

    // Step 2: Compute exp(x - max) and store
    if (tid < size) {
        shared_data[tid] = exp(data[tid] - shared_max);
    } else {
        shared_data[tid] = 0.0;
    }
    workgroupBarrier();

    // Step 3: Sum reduction
    stride = 32u;
    while (stride > 0u) {
        if (tid < stride && tid + stride < 64u) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 has the sum
    if (tid == 0u) {
        shared_sum = shared_data[0];
    }
    workgroupBarrier();

    // Step 4: Normalize - write back exp(x - max) / sum
    if (tid < size) {
        data[tid] = exp(data[tid] - shared_max) / shared_sum;
    }
}
