// SAXPY compute shader: y = alpha * x + y
// Used for SGD weight updates: weights = -learning_rate * gradients + weights

struct Params {
    size: u32,
    _pad1: u32,
    alpha: f32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    y[idx] = params.alpha * x[idx] + y[idx];
}
