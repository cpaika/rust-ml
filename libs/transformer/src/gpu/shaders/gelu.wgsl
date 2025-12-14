// GELU activation function (approximate)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// In-place operation

struct Dimensions {
    size: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

const SQRT_2_PI: f32 = 0.7978845608028654;
const COEFF: f32 = 0.044715;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= dims.size) {
        return;
    }

    let x = data[idx];
    let x3 = x * x * x;
    let inner = SQRT_2_PI * (x + COEFF * x3);

    // tanh approximation using exp
    let exp_2inner = exp(2.0 * inner);
    let tanh_val = (exp_2inner - 1.0) / (exp_2inner + 1.0);

    let cdf = 0.5 * (1.0 + tanh_val);
    data[idx] = x * cdf;
}
