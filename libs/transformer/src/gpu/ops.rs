//! GPU compute operations for transformer inference
//!
//! Provides high-level operations that dispatch GPU compute shaders

use super::GpuContext;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu;

/// Dimensions for matrix multiplication: C = A @ B
/// A: [M, K], B: [K, N], C: [M, N]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MatmulDims {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub _padding: u32,
}

/// Dimensions for attention QK computation
/// Q: [seq_len, head_dim], K: [seq_len, head_dim]
/// Output: [seq_len, seq_len] attention scores
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AttentionQKDims {
    pub seq_len: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub _padding: u32,
}

/// Dimensions for attention V computation
/// Attention: [seq_len, seq_len], V: [seq_len, head_dim]
/// Output: [seq_len, head_dim]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AttentionVDims {
    pub seq_len: u32,
    pub head_dim: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// Dimensions for softmax over rows
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SoftmaxDims {
    pub rows: u32,
    pub cols: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// Dimensions for layer normalization
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LayerNormDims {
    pub seq_len: u32,
    pub d_model: u32,
    pub eps: f32,
    pub _padding: u32,
}

/// Dimensions for GELU activation
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GeluDims {
    pub size: u32,
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
}

/// Dimensions for element-wise addition
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AddDims {
    pub size: u32,
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
}

/// Dimensions for embedding lookup
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EmbeddingDims {
    pub seq_len: u32,
    pub d_model: u32,
    pub vocab_size: u32,
    pub _padding: u32,
}

/// GPU operations runner
pub struct GpuOps {
    ctx: Arc<GpuContext>,
}

impl GpuOps {
    /// Create GPU operations runner from context
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        Self { ctx }
    }

    /// Create a uniform buffer with data
    pub fn create_uniform_buffer<T: Pod>(&self, data: &T, label: &str) -> wgpu::Buffer {
        self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Create a storage buffer with data (read-only)
    pub fn create_storage_buffer(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// Create a storage buffer for u32 data (for token IDs)
    pub fn create_storage_buffer_u32(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// Create a storage buffer for read-write (output)
    pub fn create_output_buffer(&self, size: usize, label: &str) -> wgpu::Buffer {
        self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for reading results back to CPU
    pub fn create_staging_buffer(&self, size: usize, label: &str) -> wgpu::Buffer {
        self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Run matrix multiplication: C = A @ B
    /// A: [M, K], B: [K, N], C: [M, N]
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Vec<f32> {
        let dims = MatmulDims { m, k, n, _padding: 0 };

        let dims_buffer = self.create_uniform_buffer(&dims, "matmul_dims");
        let a_buffer = self.create_storage_buffer(a, "matmul_a");
        let b_buffer = self.create_storage_buffer(b, "matmul_b");
        let c_buffer = self.create_output_buffer((m * n) as usize, "matmul_c");
        let staging = self.create_staging_buffer((m * n) as usize, "matmul_staging");

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: self.ctx.matmul_layout(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.matmul_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            // Tile size is 16x16
            let workgroups_x = (n + 15) / 16;
            let workgroups_y = (m + 15) / 16;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging, 0, (m * n * 4) as u64);
        self.ctx.queue.submit(Some(encoder.finish()));

        self.read_buffer(&staging, (m * n) as usize)
    }

    /// Run layer normalization
    /// Input: [seq_len, d_model]
    /// Output: [seq_len, d_model] normalized
    pub fn layer_norm(
        &self,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        seq_len: u32,
        d_model: u32,
        eps: f32,
    ) -> Vec<f32> {
        let dims = LayerNormDims { seq_len, d_model, eps, _padding: 0 };

        let dims_buffer = self.create_uniform_buffer(&dims, "ln_dims");
        let input_buffer = self.create_storage_buffer(input, "ln_input");
        let gamma_buffer = self.create_storage_buffer(gamma, "ln_gamma");
        let beta_buffer = self.create_storage_buffer(beta, "ln_beta");
        let output_buffer = self.create_output_buffer((seq_len * d_model) as usize, "ln_output");
        let staging = self.create_staging_buffer((seq_len * d_model) as usize, "ln_staging");

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ln_bind_group"),
            layout: self.ctx.layer_norm_layout(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gamma_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: beta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ln_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ln_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.layer_norm_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(seq_len, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_buffer, 0, &staging, 0,
            (seq_len * d_model * 4) as u64
        );
        self.ctx.queue.submit(Some(encoder.finish()));

        self.read_buffer(&staging, (seq_len * d_model) as usize)
    }

    /// Run GELU activation in-place
    pub fn gelu(&self, data: &[f32]) -> Vec<f32> {
        let size = data.len();
        let dims = GeluDims {
            size: size as u32,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
        };

        let dims_buffer = self.create_uniform_buffer(&dims, "gelu_dims");
        // For in-place operation, we need a read-write buffer
        let data_buffer = self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gelu_data"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = self.create_staging_buffer(size, "gelu_staging");

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gelu_bind_group"),
            layout: self.ctx.gelu_layout(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gelu_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gelu_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.gelu_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&data_buffer, 0, &staging, 0, (size * 4) as u64);
        self.ctx.queue.submit(Some(encoder.finish()));

        self.read_buffer(&staging, size)
    }

    /// Run element-wise addition: out = a + b
    pub fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        let size = a.len();
        let dims = AddDims {
            size: size as u32,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
        };

        let dims_buffer = self.create_uniform_buffer(&dims, "add_dims");
        let a_buffer = self.create_storage_buffer(a, "add_a");
        let b_buffer = self.create_storage_buffer(b, "add_b");
        let out_buffer = self.create_output_buffer(size, "add_out");
        let staging = self.create_staging_buffer(size, "add_staging");

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add_bind_group"),
            layout: self.ctx.add_layout(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("add_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("add_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.add_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging, 0, (size * 4) as u64);
        self.ctx.queue.submit(Some(encoder.finish()));

        self.read_buffer(&staging, size)
    }

    /// Run embedding lookup
    /// token_ids: [seq_len], embeddings: [vocab_size * d_model]
    /// Output: [seq_len * d_model]
    pub fn embedding_lookup(
        &self,
        token_ids: &[u32],
        embeddings: &[f32],
        seq_len: u32,
        d_model: u32,
        vocab_size: u32,
    ) -> Vec<f32> {
        let dims = EmbeddingDims {
            seq_len,
            d_model,
            vocab_size,
            _padding: 0,
        };

        let dims_buffer = self.create_uniform_buffer(&dims, "emb_dims");
        let ids_buffer = self.create_storage_buffer_u32(token_ids, "emb_ids");
        let emb_buffer = self.create_storage_buffer(embeddings, "emb_table");
        let out_buffer = self.create_output_buffer((seq_len * d_model) as usize, "emb_out");
        let staging = self.create_staging_buffer((seq_len * d_model) as usize, "emb_staging");

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("emb_bind_group"),
            layout: self.ctx.embedding_layout(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ids_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: emb_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("emb_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("emb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.embedding_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            let total = seq_len * d_model;
            let workgroups = (total + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging, 0, (seq_len * d_model * 4) as u64);
        self.ctx.queue.submit(Some(encoder.finish()));

        self.read_buffer(&staging, (seq_len * d_model) as usize)
    }

    /// Run softmax over rows of a matrix
    /// Input: [rows, cols], Output: [rows, cols] with softmax applied per row
    pub fn softmax_rows(&self, data: &[f32], rows: u32, cols: u32) -> Vec<f32> {
        let dims = SoftmaxDims {
            rows,
            cols,
            _padding1: 0,
            _padding2: 0,
        };

        let dims_buffer = self.create_uniform_buffer(&dims, "softmax_dims");
        // Need a read-write buffer for in-place operation
        let data_buffer = self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("softmax_data"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = self.create_staging_buffer((rows * cols) as usize, "softmax_staging");

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_bind_group"),
            layout: self.ctx.softmax_layout(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("softmax_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.softmax_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per row
            pass.dispatch_workgroups(rows, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&data_buffer, 0, &staging, 0, (rows * cols * 4) as u64);
        self.ctx.queue.submit(Some(encoder.finish()));

        self.read_buffer(&staging, (rows * cols) as usize)
    }

    /// Compute attention scores: Q @ K^T with scaling and causal masking
    /// Q: [seq_len, head_dim], K: [seq_len, head_dim]
    /// Output: [seq_len, seq_len]
    pub fn attention_qk(
        &self,
        q: &[f32],
        k: &[f32],
        seq_len: u32,
        head_dim: u32,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let dims = AttentionQKDims {
            seq_len,
            head_dim,
            scale,
            _padding: 0,
        };

        let dims_buffer = self.create_uniform_buffer(&dims, "attn_qk_dims");
        let q_buffer = self.create_storage_buffer(q, "attn_q");
        let k_buffer = self.create_storage_buffer(k, "attn_k");
        let out_buffer = self.create_output_buffer((seq_len * seq_len) as usize, "attn_scores");
        let staging = self.create_staging_buffer((seq_len * seq_len) as usize, "attn_qk_staging");

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_qk_bind_group"),
            layout: self.ctx.attention_qk_layout(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn_qk_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attn_qk_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.attention_qk_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output element
            let total = seq_len * seq_len;
            let workgroups = (total + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging, 0, (seq_len * seq_len * 4) as u64);
        self.ctx.queue.submit(Some(encoder.finish()));

        self.read_buffer(&staging, (seq_len * seq_len) as usize)
    }

    /// Compute attention output: attention_weights @ V
    /// Attention: [seq_len, seq_len], V: [seq_len, head_dim]
    /// Output: [seq_len, head_dim]
    pub fn attention_v(
        &self,
        attention: &[f32],
        v: &[f32],
        seq_len: u32,
        head_dim: u32,
    ) -> Vec<f32> {
        let dims = AttentionVDims {
            seq_len,
            head_dim,
            _padding1: 0,
            _padding2: 0,
        };

        let dims_buffer = self.create_uniform_buffer(&dims, "attn_v_dims");
        let attn_buffer = self.create_storage_buffer(attention, "attn_weights");
        let v_buffer = self.create_storage_buffer(v, "attn_v");
        let out_buffer = self.create_output_buffer((seq_len * head_dim) as usize, "attn_output");
        let staging = self.create_staging_buffer((seq_len * head_dim) as usize, "attn_v_staging");

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_v_bind_group"),
            layout: self.ctx.attention_v_layout(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: attn_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn_v_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attn_v_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.attention_v_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output element
            let total = seq_len * head_dim;
            let workgroups = (total + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging, 0, (seq_len * head_dim * 4) as u64);
        self.ctx.queue.submit(Some(encoder.finish()));

        self.read_buffer(&staging, (seq_len * head_dim) as usize)
    }

    /// Full single-head attention: softmax(Q @ K^T / sqrt(d)) @ V
    /// Q, K, V: [seq_len, head_dim]
    /// Output: [seq_len, head_dim]
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: u32,
        head_dim: u32,
    ) -> Vec<f32> {
        // Step 1: Compute Q @ K^T with scaling and causal mask
        let scores = self.attention_qk(q, k, seq_len, head_dim);

        // Step 2: Apply softmax to each row
        let weights = self.softmax_rows(&scores, seq_len, seq_len);

        // Step 3: Multiply by V
        self.attention_v(&weights, v, seq_len, head_dim)
    }

    /// Read buffer back to CPU (blocking)
    fn read_buffer(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("Failed to map buffer");

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();
        result[..len].to_vec()
    }
}

// Re-export buffer initialization
use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<GpuOps> {
        let ctx = GpuContext::new().ok()?;
        Some(GpuOps::new(Arc::new(ctx)))
    }

    #[test]
    fn test_matmul_2x2() {
        let Some(ops) = setup() else { return };

        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let c = ops.matmul(&a, &b, 2, 2, 2);

        assert_eq!(c.len(), 4);
        assert!((c[0] - 19.0).abs() < 0.001);
        assert!((c[1] - 22.0).abs() < 0.001);
        assert!((c[2] - 43.0).abs() < 0.001);
        assert!((c[3] - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_matmul_non_square() {
        let Some(ops) = setup() else { return };

        // A = [[1, 2, 3]], B = [[1], [2], [3]]
        // C = [[14]]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let c = ops.matmul(&a, &b, 1, 3, 1);

        assert_eq!(c.len(), 1);
        assert!((c[0] - 14.0).abs() < 0.001);
    }

    #[test]
    fn test_gelu_activation() {
        let Some(ops) = setup() else { return };

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let output = ops.gelu(&input);

        assert_eq!(output.len(), 5);
        // GELU(0) = 0
        assert!(output[2].abs() < 0.001);
        // GELU is approximately identity for large positive
        assert!((output[4] - 2.0).abs() < 0.1);
        // GELU is approximately 0 for large negative
        assert!(output[0].abs() < 0.1);
    }

    #[test]
    fn test_add_vectors() {
        let Some(ops) = setup() else { return };

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let c = ops.add(&a, &b);

        assert_eq!(c.len(), 4);
        assert!((c[0] - 6.0).abs() < 0.001);
        assert!((c[1] - 8.0).abs() < 0.001);
        assert!((c[2] - 10.0).abs() < 0.001);
        assert!((c[3] - 12.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_lookup() {
        let Some(ops) = setup() else { return };

        // Vocabulary of 3 tokens, embedding dim 2
        // Token 0 -> [1, 2], Token 1 -> [3, 4], Token 2 -> [5, 6]
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let token_ids = vec![2u32, 0u32, 1u32];

        let output = ops.embedding_lookup(&token_ids, &embeddings, 3, 2, 3);

        assert_eq!(output.len(), 6);
        // Token 2: [5, 6]
        assert!((output[0] - 5.0).abs() < 0.001);
        assert!((output[1] - 6.0).abs() < 0.001);
        // Token 0: [1, 2]
        assert!((output[2] - 1.0).abs() < 0.001);
        assert!((output[3] - 2.0).abs() < 0.001);
        // Token 1: [3, 4]
        assert!((output[4] - 3.0).abs() < 0.001);
        assert!((output[5] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_layer_norm() {
        let Some(ops) = setup() else { return };

        // Single row, dim 4
        // After normalization with gamma=1, beta=0, should have mean~0 and std~1
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];

        let output = ops.layer_norm(&input, &gamma, &beta, 1, 4, 1e-5);

        assert_eq!(output.len(), 4);

        // Mean should be approximately 0
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 0.01, "Mean should be near 0, got {}", mean);

        // Variance should be approximately 1
        let variance: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((variance - 1.0).abs() < 0.1, "Variance should be near 1, got {}", variance);
    }

    #[test]
    fn test_layer_norm_with_scale_shift() {
        let Some(ops) = setup() else { return };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0, 2.0, 2.0, 2.0]; // Scale by 2
        let beta = vec![1.0, 1.0, 1.0, 1.0];  // Shift by 1

        let output = ops.layer_norm(&input, &gamma, &beta, 1, 4, 1e-5);

        // Mean should be shifted to 1 (beta)
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!((mean - 1.0).abs() < 0.1, "Mean should be near 1, got {}", mean);
    }

    #[test]
    fn test_softmax_rows() {
        let Some(ops) = setup() else { return };

        // 2 rows, 3 cols
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = ops.softmax_rows(&data, 2, 3);

        assert_eq!(output.len(), 6);

        // Each row should sum to 1
        let row1_sum: f32 = output[0..3].iter().sum();
        let row2_sum: f32 = output[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 0.001, "Row 1 sum: {}", row1_sum);
        assert!((row2_sum - 1.0).abs() < 0.001, "Row 2 sum: {}", row2_sum);

        // Larger values should have higher probabilities
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_attention_qk_causal_mask() {
        let Some(ops) = setup() else { return };

        // Simple Q and K vectors
        let seq_len = 3u32;
        let head_dim = 2u32;

        // Q = [[1, 0], [0, 1], [1, 1]]
        // K = [[1, 0], [0, 1], [1, 1]]
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let scores = ops.attention_qk(&q, &k, seq_len, head_dim);

        assert_eq!(scores.len(), 9);

        // Check causal mask: upper triangle should be -inf (or very negative)
        // Position (0,1), (0,2), (1,2) should be masked
        assert!(scores[1] < -1e6, "Position (0,1) should be masked: {}", scores[1]);
        assert!(scores[2] < -1e6, "Position (0,2) should be masked: {}", scores[2]);
        assert!(scores[5] < -1e6, "Position (1,2) should be masked: {}", scores[5]);

        // Diagonal and lower triangle should have real values
        // Q[0] @ K[0]^T = 1*1 + 0*0 = 1, scaled by 1/sqrt(2)
        let scale = 1.0 / (head_dim as f32).sqrt();
        assert!((scores[0] - 1.0 * scale).abs() < 0.1, "Position (0,0): {}", scores[0]);
    }

    #[test]
    fn test_attention_v() {
        let Some(ops) = setup() else { return };

        let seq_len = 2u32;
        let head_dim = 2u32;

        // Simple attention weights (already softmaxed)
        // Row 0: attend only to position 0 (causal)
        // Row 1: attend 50/50 to positions 0 and 1
        let attention = vec![1.0, 0.0, 0.5, 0.5];

        // V = [[1, 2], [3, 4]]
        let v = vec![1.0, 2.0, 3.0, 4.0];

        let output = ops.attention_v(&attention, &v, seq_len, head_dim);

        assert_eq!(output.len(), 4);

        // Position 0: 1.0 * [1, 2] = [1, 2]
        assert!((output[0] - 1.0).abs() < 0.001);
        assert!((output[1] - 2.0).abs() < 0.001);

        // Position 1: 0.5 * [1, 2] + 0.5 * [3, 4] = [2, 3]
        assert!((output[2] - 2.0).abs() < 0.001);
        assert!((output[3] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_full_attention() {
        let Some(ops) = setup() else { return };

        let seq_len = 2u32;
        let head_dim = 2u32;

        // Q, K, V all the same for simplicity
        let qkv = vec![1.0, 0.0, 0.0, 1.0];

        let output = ops.attention(&qkv, &qkv, &qkv, seq_len, head_dim);

        assert_eq!(output.len(), 4);

        // Position 0 can only attend to itself
        // softmax([q0 @ k0^T / sqrt(2)]) = [1.0]
        // output[0] = 1.0 * v0 = [1, 0]
        assert!((output[0] - 1.0).abs() < 0.1);
        assert!(output[1].abs() < 0.1);

        // Position 1 can attend to both
        // Q[1] = [0, 1], K = [[1, 0], [0, 1]]
        // Q[1] @ K[0]^T = 0, Q[1] @ K[1]^T = 1
        // After scaling and softmax, position 1 attends more to itself
    }
}
