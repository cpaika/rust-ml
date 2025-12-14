// ReLU activation compute shader
// Applies max(0, x) element-wise in-place

struct Dims {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Bounds check
    if (idx >= dims.size) {
        return;
    }

    // ReLU: max(0, x)
    data[idx] = max(0.0, data[idx]);
}
