// Output layer delta compute shader
// Computes delta = softmax_output - one_hot(label)
// For softmax with cross-entropy loss, this is the gradient of the loss w.r.t. pre-softmax logits

struct Params {
    size: u32,
    label: u32,  // The correct class label (0-indexed)
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> output: array<f32>;  // Softmax output
@group(0) @binding(2) var<storage, read_write> delta: array<f32>;  // Output gradient

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // delta = output - one_hot(label)
    // one_hot(label)[i] = 1 if i == label, else 0
    let expected = select(0.0, 1.0, idx == params.label);
    delta[idx] = output[idx] - expected;
}
