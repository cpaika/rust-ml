// ReLU backward pass compute shader
// Computes: grad_output * (input > 0 ? 1 : 0)
// grad is modified in-place based on the original pre-activation values

struct Dims {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> grad: array<f32>;      // gradient (modified in-place)
@group(0) @binding(2) var<storage, read> pre_activation: array<f32>;  // z values before ReLU

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= dims.size) {
        return;
    }

    // ReLU derivative: 1 if x > 0, else 0
    if (pre_activation[idx] <= 0.0) {
        grad[idx] = 0.0;
    }
    // else: grad stays the same (multiply by 1)
}
