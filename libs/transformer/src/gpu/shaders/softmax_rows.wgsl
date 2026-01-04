// Row-wise softmax for attention scores
// Input: data [rows, cols]
// Output: data [rows, cols] (in-place, softmax applied to each row)
// Each workgroup handles one row

struct Dimensions {
    rows: u32,
    cols: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> shared_max: array<f32, 64>;
var<workgroup> shared_sum: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = workgroup_id.x;
    let tid = local_id.x;
    let cols = dims.cols;

    if (row >= dims.rows) {
        return;
    }

    let row_start = row * cols;

    // Step 1: Find max in this row (parallel reduction)
    var local_max: f32 = -1e9;
    for (var i = tid; i < cols; i = i + WORKGROUP_SIZE) {
        local_max = max(local_max, data[row_start + i]);
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Reduce to find global max
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
    }

    let max_val = shared_max[0];
    workgroupBarrier();

    // Step 2: Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    for (var i = tid; i < cols; i = i + WORKGROUP_SIZE) {
        let exp_val = exp(data[row_start + i] - max_val);
        data[row_start + i] = exp_val;
        local_sum = local_sum + exp_val;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Reduce to find total sum
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let sum_val = shared_sum[0];
    workgroupBarrier();

    // Step 3: Normalize
    for (var i = tid; i < cols; i = i + WORKGROUP_SIZE) {
        data[row_start + i] = data[row_start + i] / sum_val;
    }
}
